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
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        self._compile = compiled
        self._optimizer_conf = optimizer_conf
        self._scheduler_conf = scheduler_conf
        log.info(f"[algorithm] init compile={self._compile}, has_scheduler={self._scheduler_conf is not None}")
        
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
        
        # FLOPs and timing tracking state (per-phase)
        self._should_count_flops_train = True
        self._should_count_flops_val = True
        self._num_flops_train = None
        self._num_flops_val = None
        self._last_elapsed_ms = None
        self._is_compiled = False
        self._compiled_forward = None
        self._original_forward = None
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
    
    @property
    def network(self) -> nn.Module:
        return self._network
    
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            OmegaConf.create(self._optimizer_conf),
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
            
            # Determine phase from model training state
            is_training = self._network.training
            should_count = self._should_count_flops_train if is_training else self._should_count_flops_val
            
            if should_count:
                if self.global_rank == 0:
                    # Create fresh FlopCounterMode to avoid memory retention
                    flop_counter = FlopCounterMode(display=False, depth=1)
                    with flop_counter:
                        result = fwd(*args, **kwargs)
                    flops = flop_counter.get_total_flops()
                    phase = "train" if is_training else "val"
                    if is_training:
                        self._num_flops_train = flops
                        self._should_count_flops_train = False
                    else:
                        self._num_flops_val = flops
                        self._should_count_flops_val = False
                    log.info(f"[FLOPs] Measured network FLOPs ({phase}): {flops:.2e}")
                else:
                    result = fwd(*args, **kwargs)
                    if is_training:
                        self._should_count_flops_train = False
                    else:
                        self._should_count_flops_val = False
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
    
    def __step(self, batch, split_name):
        output = self._step(batch, split_name)
        
        # After first forward (FLOPs measured), compile the network
        if not self._is_compiled and not self._compile.disable:
            self._compile_network()
        
        # Select the appropriate FLOP count based on phase
        is_training = split_name == "train"
        num_flops = self._num_flops_train if is_training else self._num_flops_val
        
        # Calculate FLOP/s using cached FLOPs and timing from network forward
        if num_flops is not None and self._last_elapsed_ms is not None and self._last_elapsed_ms > 0:
            flops = num_flops / (self._last_elapsed_ms / 1000)
            output["flops"] = flops
            progress_bar_dict = {
                "flops": flops,
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
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.training_step_outputs.append(outputs)
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.validation_step_outputs[dataloader_idx].append(outputs)
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.test_step_outputs[dataloader_idx].append(outputs)
    
    def training_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="train")
        return output
    
    def validation_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="val")
        return output
    
    def test_step(self, batch, *args, **kwargs):
        output = self.__step(batch, split_name="test")
        return output
    
    def _epoch_end(self, outputs: dict[str, list], split_name):
        epoch_metrics = self._calculate_epoch_metrics(outputs)
        epoch_metrics = {f"{split_name}_{k}": v for k, v in epoch_metrics.items()}
        
        for checkpoint in self.trainer.checkpoint_callbacks:
            if checkpoint.monitor in epoch_metrics:
                epoch_metrics[checkpoint.monitor] = torch.Tensor(
                    epoch_metrics[checkpoint.monitor]
                )
        
        self.trainer.callback_metrics.update(epoch_metrics)
        if self.logger:
            self.logger.log_metrics(epoch_metrics, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics}\n""")
    
    def on_train_epoch_end(self) -> None:
        outputs = self.training_step_outputs
        num_dataloaders = 1
        if isinstance(self.trainer.val_dataloaders, (list, tuple)):
            num_dataloaders = len(self.trainer.val_dataloaders)
        # Log same training metrics with different prefixes for each validation dataloader
        for i in range(num_dataloaders):
            self._epoch_end(outputs, split_name=f"train_{i}")
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        # Log metrics for ALL validation dataloaders
        for dataloader_idx in sorted(self.validation_step_outputs.keys()):
            outputs = self.validation_step_outputs[dataloader_idx]
            self._epoch_end(outputs, split_name=f"val_{dataloader_idx}")
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self) -> None:
        for test_num in self.test_step_outputs:
            outputs = self.test_step_outputs[test_num]
            self._epoch_end(outputs, split_name=f"test_{test_num}")
        
        self.test_step_outputs.clear()
