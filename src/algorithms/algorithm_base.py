import logging
from collections import defaultdict
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.flop_counter import FlopCounterMode

from src.utils import CompileParams

log = logging.getLogger(__name__)


class AlgorithmBase(pl.LightningModule):
    
    def __init__(
        self,
        compiled: CompileParams,
        log_every_n_steps: int,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        validation_names: list[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        if validation_names is None:
            raise ValueError("validation_names must be provided (e.g. ['rt'], ['rts_0.02', 'rts_0.5'])")
        self.validation_names = validation_names
        
        self._log_every_n_steps = log_every_n_steps
        
        self._compile = compiled
        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf
        log.info(
            f"[algorithm] init compile={self._compile}, has_scheduler={self._scheduler_conf is not None}, "
            f"log_every_n_steps={self._log_every_n_steps}"
        )
        
        if network is None:
            self._network: nn.Module = hydra.utils.instantiate(
                OmegaConf.create(network_conf)
            )
        else:
            self._network: nn.Module = network
        
        self._gpu = gpu
        if self._gpu is not None:
            self._network.cuda(gpu)
        
        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(lambda: defaultdict(list))
        self.test_step_outputs = defaultdict(lambda: defaultdict(list))
        
        # FLOPs and timing tracking state
        self._should_count_flops = True  # Only count once, during first training step
        self._has_logged_flop_counts = False
        self._num_flops_train = None
        self._num_flops_backward = None
        self._last_elapsed_ms = None
        self._last_backward_ms = None
        self._is_compiled = False
        self._compiled_forward = None
        self._original_forward = None
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start_backward = torch.cuda.Event(enable_timing=True)
        self._end_backward = torch.cuda.Event(enable_timing=True)

        # torch.utils.flop_counter does not fully support SDPA enable_gqa paths.
        # Disable initial FLOPs counting for GQA runs to avoid first-step crashes.
        try:
            gqa_group_size = int(getattr(self._network, "nsa_gqa_group_size", 1))
            if gqa_group_size > 1:
                self._should_count_flops = False
        except Exception:
            pass

    def _disable_flop_counting(self, reason: str) -> None:
        """Disable FLOPs bookkeeping after unsupported shape/operator is detected."""
        self._should_count_flops = False
        self._num_flops_train = None
        self._num_flops_backward = None
        if self.global_rank == 0:
            log.warning(f"[FLOPs] disabled: {reason}")

    @staticmethod
    def _is_known_flop_counter_issue(exc: Exception) -> bool:
        msg = str(exc)
        if "sdpa_flop_count" in msg or "query/key/value shapes are incompatible" in msg:
            return True

        tb = exc.__traceback__
        while tb is not None:
            frame_path = tb.tb_frame.f_code.co_filename.replace("\\", "/")
            if "torch/utils/flop_counter.py" in frame_path:
                return True
            tb = tb.tb_next
        return False
    
    @property
    def network(self) -> nn.Module:
        return self._network
    
    def configure_optimizers(self):
        optimizer_conf = OmegaConf.create(self._optimizer_conf)
        optimizer_conf.pop("name", None)
        optimizer = hydra.utils.instantiate(
            optimizer_conf,
            params=filter(lambda p: p.requires_grad, self.parameters()),
        )
        try:
            opt_cls = type(optimizer).__name__
            lr = optimizer.param_groups[0].get("lr", None)
            log.info(f"[optimizer] {opt_cls} lr={lr}")
        except Exception:
            pass
        
        ret_opt = {"optimizer": optimizer}
        if self._scheduler_conf is not None:
            scheduler_conf = OmegaConf.create(self._scheduler_conf)
            scheduler_conf.pop("name", None)
            # Get monitor if exists, else None
            monitor = scheduler_conf.get("monitor", None)
            if "monitor" in scheduler_conf:
                del scheduler_conf["monitor"]
            
            if "_target_" not in scheduler_conf:
                scheduler_keys = list(scheduler_conf.keys())
                if len(scheduler_keys) != 1:
                    raise RuntimeError(
                        "Scheduler configuration must contain exactly one target definition."
                    )
                scheduler_conf = scheduler_conf[scheduler_keys[0]]
            
            interval = scheduler_conf.get("interval", None)
            if "interval" in scheduler_conf:
                del scheduler_conf["interval"]
            frequency = scheduler_conf.get("frequency", None)
            if "frequency" in scheduler_conf:
                del scheduler_conf["frequency"]
            
            scheduler_conf["batch_size"] = self.trainer.datamodule._batch_size
            scheduler: LRScheduler = hydra.utils.instantiate(
                scheduler_conf,
                optimizer=optimizer,
            )
            sch_opt = {"scheduler": scheduler}
            
            if monitor:
                sch_opt["monitor"] = monitor
            if interval is not None:
                sch_opt["interval"] = interval
            if frequency is not None:
                sch_opt["frequency"] = frequency
            
            ret_opt.update({"lr_scheduler": sch_opt})
            try:
                log.info(f"[scheduler] {type(scheduler).__name__} monitor={monitor}")
            except Exception:
                pass
        
        return ret_opt
    
    def _wrap_network_for_flops(self) -> None:
        """Wrap the network's forward method with FLOPs counting and timing."""
        # Store original forward for later compilation
        self._original_forward = self._network.forward
        
        def forward_with_flops_and_timing(*args, **kwargs):
            self._start.record()
            
            # Use compiled forward if available, otherwise original
            fwd = self._compiled_forward if self._compiled_forward else self._original_forward
            
            # Count both train and val FLOPs during first training step only
            if self._should_count_flops and self._network.training:
                if self.global_rank == 0:
                    # 1. Measure training FLOPs (already in training mode)
                    try:
                        flop_counter_train = FlopCounterMode(display=False, depth=1)
                        with flop_counter_train:
                            result = fwd(*args, **kwargs)
                        self._num_flops_train = flop_counter_train.get_total_flops()
                        log.info(f"[FLOPs] Measured network FLOPs (train): {self._num_flops_train:.2e}")
                    except Exception as e:
                        if self._is_known_flop_counter_issue(e):
                            self._disable_flop_counting(f"unsupported SDPA/GQA flop path ({e})")
                            result = fwd(*args, **kwargs)
                        else:
                            raise
                else:
                    result = fwd(*args, **kwargs)
                self._should_count_flops = False
            else:
                result = fwd(*args, **kwargs)
            
            self._end.record()
            torch.cuda.synchronize()
            self._last_elapsed_ms = self._start.elapsed_time(self._end)
            return result
        
        self._network.forward = forward_with_flops_and_timing
    
    def _compile_network(self) -> None:
        """Compile the network's forward pass (called after FLOPs measurement)."""
        if self._compile.disable or self._is_compiled:
            return
        
        log.info("Compiling the network...")
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        # Compile the original forward function, not the wrapper
        # This allows the wrapper to do timing outside the compiled graph
        self._compiled_forward = torch.compile(
            self._original_forward,
            fullgraph=self._compile.fullgraph,
            dynamic=self._compile.dynamic,
            backend=self._compile.backend,
            mode=self._compile.mode,
            options=self._compile.options,
        )
        t1.record()
        torch.cuda.synchronize()
        self._is_compiled = True
        log.info(
            f"[compile] done; elapsed={t0.elapsed_time(t1) / 1000.0:.3f}s "
            f"(fullgraph={self._compile.fullgraph}, backend={self._compile.backend}, mode={self._compile.mode})"
        )
    
    def setup(self, stage: str) -> None:
        """Setup hook - wrap network for FLOPs counting."""
        if stage in ("fit", "test"):
            self._wrap_network_for_flops()
    
    def pred(self, batch):
        raise NotImplementedError
    
    def _step(self, batch, *args, **kwargs):
        raise NotImplementedError
    
    def __step(self, batch, split_name, dataloader_idx: int = 0):
        output = self._step(batch, split_name)
        
        # After first forward (FLOPs measured), measure backward FLOPS on first train step
        if (
            split_name == "train"
            and self._num_flops_backward is None
            and self._num_flops_train is not None
            and self.global_rank == 0
        ):
            try:
                flop_counter = FlopCounterMode(display=False, depth=1)
                with flop_counter:
                    output_measure = self._step(batch, split_name)
                    output_measure["loss"].backward()
                total_flops = flop_counter.get_total_flops()
                self._num_flops_backward = total_flops - self._num_flops_train
                log.info(
                    f"[FLOPs] Measured network FLOPs (backward): {self._num_flops_backward:.2e}"
                )
                self.optimizers().zero_grad(set_to_none=True)
            except Exception as e:
                if self._is_known_flop_counter_issue(e):
                    self._disable_flop_counting(f"unsupported SDPA/GQA flop path ({e})")
                else:
                    raise
        
        # After first forward (FLOPs measured), compile the network
        if not self._is_compiled and not self._compile.disable:
            self._compile_network()
        
        # Select the appropriate FLOP count based on phase
        is_training = split_name == "train"
        should_compute_flops = is_training
        num_flops = self._num_flops_train if is_training else None
        
        # Calculate flops_forward; val uses it as flops_overall; train gets overall in on_train_batch_end
        forward_time_s = self._last_elapsed_ms / 1000.0 if self._last_elapsed_ms else None
        if should_compute_flops and num_flops is not None and forward_time_s is not None and forward_time_s > 0:
            flops_forward = num_flops / forward_time_s
            output["flops_forward"] = flops_forward
            if not is_training:
                output["flops_overall"] = flops_forward
            progress_bar_dict = {
                "flops_forward": flops_forward,
                "loss": output["loss"].item(),
            }
            self.trainer.progress_bar_metrics.update(progress_bar_dict)
        
        return output
    
    @staticmethod
    def convert_to_numpy(output_dict: dict[str, torch.Tensor | Any]):
        for key in output_dict:
            if isinstance(output_dict[key], torch.Tensor):
                output_dict[key] = output_dict[key].detach().cpu()
        
        return output_dict
    
    def on_before_backward(self, loss: torch.Tensor) -> None:
        self._start_backward.record()
    
    def on_after_backward(self) -> None:
        self._end_backward.record()
        torch.cuda.synchronize()
        self._last_backward_ms = self._start_backward.elapsed_time(self._end_backward)
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        # Add flops_backward and flops_overall for train (backward time now available)
        if (
            "flops_forward" in outputs
            and self._num_flops_train is not None
            and self._num_flops_backward is not None
        ):
            backward_time_s = self._last_backward_ms / 1000.0
            flops_backward = self._num_flops_backward / backward_time_s
            outputs["flops_backward"] = flops_backward
            forward_time_s = self._last_elapsed_ms / 1000.0
            total_flops = self._num_flops_train + self._num_flops_backward
            total_time_s = forward_time_s + backward_time_s
            outputs["flops_overall"] = total_flops / total_time_s
            # Log FLOP counts as hparams once (same mechanism as log_hyperparameters)
            if not self._has_logged_flop_counts and self.logger:
                orig = self.trainer._original_log_hyperparams
                flop_hparams = {
                    "algorithm/num_flops_forward": self._num_flops_train,
                    "algorithm/num_flops_backward": self._num_flops_backward,
                    "algorithm/num_flops_overall": total_flops,
                }
                orig(flop_hparams)
                self._has_logged_flop_counts = True
        self.training_step_outputs.append(outputs)
        
        if len(self.training_step_outputs) >= self._log_every_n_steps:
            self._log_training_step_metrics()
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.validation_step_outputs[dataloader_idx].append(outputs)
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.test_step_outputs[dataloader_idx].append(outputs)
    
    def training_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="train")
        return output
    
    def validation_step(self, batch, *args, dataloader_idx: int = 0, **kwargs):
        output = self.__step(batch, split_name="val", dataloader_idx=dataloader_idx)
        return output
    
    def test_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="test")
        return output
    
    def _epoch_end(self, outputs: dict[str, list], split_name: str) -> None:
        epoch_metrics = self._calculate_epoch_metrics(outputs)
        
        prefixed_metrics = {}
        wandb_metrics = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        }
        
        for k, v in epoch_metrics.items():
            if "flops" in k:
                prefixed_metrics[k] = v
                wandb_metrics[k] = v
            else:
                original_key = f"{split_name}_{k}"
                prefixed_metrics[original_key] = v
                
                if split_name.startswith("train_"):
                    group_name = split_name.replace("train_", "", 1)
                    wandb_metrics[f"{k}/{group_name}/train"] = v
                elif split_name.startswith("val_"):
                    group_name = split_name.replace("val_", "", 1)
                    wandb_metrics[f"{k}/{group_name}/val"] = v
                elif split_name.startswith("test_"):
                    group_name = split_name.replace("test_", "", 1)
                    wandb_metrics[f"{k}/{group_name}/test"] = v
                else:
                    wandb_metrics[original_key] = v
                
        epoch_metrics = prefixed_metrics
        
        for checkpoint in self.trainer.checkpoint_callbacks:
            if getattr(checkpoint, "monitor", None) in epoch_metrics:
                epoch_metrics[checkpoint.monitor] = torch.Tensor(
                    epoch_metrics[checkpoint.monitor]
                )
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(wandb_metrics, self.global_step)
        else:
            log.info(f"""\n{epoch_metrics}\n""")
    
    def _log_training_step_metrics(self) -> None:
        outputs = self.training_step_outputs
        num_dataloaders = 1
        if isinstance(self.trainer.val_dataloaders, (list, tuple)):
            num_dataloaders = len(self.trainer.val_dataloaders)
        # Log same training metrics with different prefixes for each validation dataloader
        for i in range(num_dataloaders):
            name = self.validation_names[i]
            self._epoch_end(outputs, split_name=f"train_{name}")
        self.training_step_outputs.clear()
    
    def on_train_epoch_end(self) -> None:
        pass
    
    def on_validation_epoch_end(self) -> None:
        # Log metrics for ALL validation dataloaders using named prefixes
        for dataloader_idx in sorted(self.validation_step_outputs.keys()):
            outputs = self.validation_step_outputs[dataloader_idx]
            name = self.validation_names[dataloader_idx]
            self._epoch_end(outputs, split_name=f"val_{name}")
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        for test_num in self.test_step_outputs:
            outputs = self.test_step_outputs[test_num]
            self._epoch_end(outputs, split_name=f"test_{test_num}")
        
        self.test_step_outputs.clear()
