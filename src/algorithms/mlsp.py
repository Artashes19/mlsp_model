import logging
import re
from collections import defaultdict
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LRScheduler

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.datasets.mlsp import IMG_TARGET_SIZE
from src.utils import CompileParams
from src.utils.mlsp.augmentations import resize_bilinear
from src.utils.mlsp.loss import create_sip2net_loss, se

log = logging.getLogger(__name__)


class MLSP(AlgorithmBase):
    
    def __init__(
        self,
        out_norm: float,
        use_sip2net: bool,
        sip2net_params: dict[str, int],
        compiled: CompileParams,
        optimizer_conf: DictConfig = None,
        scheduler_conf: DictConfig = None,
        network: nn.Module = None,
        network_conf: DictConfig = None,
        gpu: int = None,
        *args, **kwargs
    ):
        super().__init__(
            compiled=compiled,
            optimizer_conf=optimizer_conf,
            scheduler_conf=scheduler_conf,
            network=network,
            network_conf=network_conf,
            gpu=gpu
        )
        
        self.out_norm = out_norm
        self.use_sip2net = use_sip2net
        if use_sip2net:
            if sip2net_params is None:
                sip2net_params = {}
            self.sip2net_criterion = create_sip2net_loss(
                use_mse=True,
                mse_weight=sip2net_params.get("mse_weight", 1.0),
                alpha1=sip2net_params.get("alpha1", 500.0),
                alpha2=sip2net_params.get("alpha2", 1.0),
                alpha3=sip2net_params.get("alpha3", 0.0)
            )
            log.info(f"Using SIP2Net loss")
        else:
            log.info("Using pure L1/MAE objective")
        
        self.training_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        
        # Finetune configuration (optional)
        self._finetune_conf: DictConfig | dict | None = kwargs.get("finetune", None)
        if self._finetune_conf is None:
            # default disabled
            self._finetune_conf = {
                "enable": False
            }
        try:
            log.info(
                f"[finetune] enable={bool(self._finetune_conf.get('enable', False))}, "
                f"ckpt_path={self._finetune_conf.get('ckpt_path', None)}, "
                f"freeze_encoder_epochs={int(self._finetune_conf.get('freeze_encoder_epochs', 0))}, "
                f"discriminative_lr={self._finetune_conf.get('discriminative_lr', {})}, "
                f"warmup={self._finetune_conf.get('warmup', {})}, "
                f"bn_recalibration={self._finetune_conf.get('bn_recalibration', {})}, "
                f"l2sp={self._finetune_conf.get('l2sp', {})}"
            )
        except Exception:
            pass
        
        # Placeholders for finetune utilities
        self._pretrained_weights: dict[str, torch.Tensor] | None = None
        self._freeze_encoder_epochs: int = int(self._finetune_conf.get("freeze_encoder_epochs", 0))
        self._encoder_frozen: bool = False
    
    def pred(self, batch):
        inputs, targets, masks, sample = batch
        
        # Calculate original dimensions from pixel_size
        original_pixel_size = 0.25
        current_pixel_size = sample["pixel_size"]
        reverse_scale_factor = original_pixel_size / current_pixel_size
        
        # Network output is 256x256, calculate original dimensions
        old_h = int(IMG_TARGET_SIZE * reverse_scale_factor)
        old_w = int(IMG_TARGET_SIZE * reverse_scale_factor)
        
        # Forward pass with bfloat16 autocast
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.network(inputs.cuda(self._gpu).unsqueeze(0)).squeeze(0)
        
        # Resize to original dimensions with bilinear interpolation
        pred_final = resize_bilinear(pred.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
        pred_np = pred_final.detach().float().cpu().numpy()
        
        # Convert inputs, targets, masks to numpy
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy() if targets is not None and targets != "" else None
        masks_np = masks.detach().cpu().numpy()
        
        return {
            "pred": pred_np,
            "inputs": inputs_np,
            "targets": targets_np,
            "masks": masks_np,
            "sample": sample,
        }
    
    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False
    ):
        """
        Override to handle foreign (colleague's) checkpoint format.
        Remaps keys by adding _network. prefix if needed.
        """
        # Check if this is a foreign checkpoint (no _network. prefix in keys)
        has_network_prefix = any(k.startswith("_network.") for k in state_dict.keys())
        
        if not has_network_prefix and len(state_dict) > 0:
            # This is a foreign checkpoint - remap keys
            log.info(f"[load_state_dict] Detected foreign checkpoint, remapping keys...")
            remapped = {}
            for key, value in state_dict.items():
                new_key = f"_network.{key}"
                remapped[new_key] = value
            state_dict = remapped
            # Use strict=False for foreign checkpoints due to potential mismatches
            strict = False
        
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
    
    def _capture_pretrained_reference(self) -> None:
        self._pretrained_weights = {
            name: p.detach().clone().cpu()
            for name, p in self._network.named_parameters()
        }
    
    @staticmethod
    def _name_matches(name: str, include_patterns: list[str] | None, exclude_patterns: list[str] | None) -> bool:
        def any_match(patterns: list[str] | None) -> bool:
            if not patterns:
                return False
            for pat in patterns:
                try:
                    if re.search(pat, name):
                        return True
                except re.error:
                    if pat in name:
                        return True
            return False
        
        if include_patterns and not any_match(include_patterns):
            return False
        if exclude_patterns and any_match(exclude_patterns):
            return False
        return True
    
    def _l2sp_penalty(self) -> torch.Tensor:
        if self._pretrained_weights is None:
            return torch.tensor(0.0, device=next(self._network.parameters()).device)
        l2sp_conf = self._finetune_conf.get("l2sp", {})
        include_patterns = l2sp_conf.get("include_patterns", [])
        exclude_patterns = l2sp_conf.get("exclude_patterns", [])
        penalty = None
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            if not self._name_matches(name, include_patterns, exclude_patterns):
                continue
            ref = self._pretrained_weights.get(name, None)
            if ref is None:
                continue
            ref = ref.to(p.device)
            diff = (p - ref).pow(2).sum()
            penalty = diff if penalty is None else penalty + diff
        if penalty is None:
            penalty = torch.tensor(0.0, device=next(self._network.parameters()).device)
        return penalty
    
    def _set_encoder_trainable(self, requires_grad: bool) -> None:
        net = self._network
        encoder = getattr(net, "unet", None)
        if encoder is not None:
            encoder = encoder.encoder
        else:
            # fallback: search for attribute named 'encoder'
            encoder = getattr(net, "encoder", None)
        if encoder is None:
            return
        for p in encoder.parameters():
            p.requires_grad = requires_grad
        self._encoder_frozen = not requires_grad
        log.info(f"Encoder trainable={requires_grad}")
    
    def on_fit_start(self) -> None:
        # Optional BN recalibration
        bn_conf = self._finetune_conf.get("bn_recalibration", {}) if self._finetune_conf else {}
        if bool(self._finetune_conf.get("enable", False)) and bool(bn_conf.get("enable", False)):
            num_batches = int(bn_conf.get("num_batches", 100))
            try:
                log.info(f"[finetune] BN recalibration begin: num_batches={num_batches}")
                self._network.train()
                dl = self.trainer.datamodule.train_dataloader()
                processed = 0
                device = self._gpu if self._gpu is not None else (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                for batch in dl:
                    with torch.no_grad():
                        x = batch[0].to(device)
                        _ = self._network(x)
                    processed += 1
                    if processed >= num_batches:
                        break
                log.info(f"BN recalibrated on {processed} batches")
            except Exception as ex:
                log.warning(f"BN recalibration skipped due to error: {ex}")
        
        # Initial freeze if configured
        if bool(self._finetune_conf.get("enable", False)) and self._freeze_encoder_epochs > 0:
            self._set_encoder_trainable(False)
    
    def on_train_epoch_start(self) -> None:
        # Unfreeze after configured epochs
        if bool(self._finetune_conf.get("enable", False)) and self._encoder_frozen:
            if self.current_epoch >= self._freeze_encoder_epochs:
                self._set_encoder_trainable(True)
    
    # noinspection PyMethodOverriding
    def _step(self, batch, split_name, *args, **kwargs):
        inputs, targets, masks, sample = batch
        
        # Lightning's bf16-mixed precision handles autocast for both forward AND loss
        # (matches korean-model's AMP setup where autocast wraps forward + loss)
        preds = self._network(inputs)
        
        # Squeeze channel dim: [B, 1, H, W] -> [B, H, W] to match targets shape
        if preds.dim() == 4:
            preds = preds.squeeze(1)
        
        # Clamp predictions to [0, 1] only during validation (matches colleague's approach)
        if split_name in ("val", "valid", "validation", "test"):
            preds = torch.clamp(preds, 0.0, 1.0)
        
        weights = torch.ones_like(inputs[:, -1])
        return self.get_metrics(preds, targets, masks, weights)
    
    def get_metrics(self, preds, targets, masks, weights):
        # DEBUG: Check actual value ranges
        if not hasattr(self, '_debug_logged') or not self._debug_logged:
            log.info(
                f"[DEBUG get_metrics] preds: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}"
            )
            log.info(
                f"[DEBUG get_metrics] targets: min={targets.min():.2f}, max={targets.max():.2f}, mean={targets.mean():.2f}"
            )
            log.info(f"[DEBUG get_metrics] masks: sum={masks.sum():.0f}, shape={masks.shape}")
            self._debug_logged = True
        
        # MSE (internal; used to derive RMSE)
        batch_se = se(preds, targets, masks, weights)
        batch_mse = batch_se / (masks.sum() + 1e-8)
        
        # MAE (L1) – this is the primary training loss
        abs_err = (preds - targets).abs() * masks
        if weights is not None:
            abs_err = abs_err * weights
        batch_mae = abs_err.sum() / (masks.sum() + 1e-8)
        if torch.isnan(preds).any():
            logging.error(f"[Epoch {self.trainer.current_epoch}] NaNs detected in preds")
        if torch.isinf(preds).any():
            logging.error(f"[Epoch {self.trainer.current_epoch}] Infs detected in preds")
        if torch.isnan(targets).any():
            logging.error(f"[Epoch {self.trainer.current_epoch}] NaNs detected in targets")
        if torch.isinf(targets).any():
            logging.error(f"[Epoch {self.trainer.current_epoch}] Infs detected in targets")
        
        # Use SIP2Net loss if requested
        if self.use_sip2net:
            loss, _ = self.sip2net_criterion(preds, targets, masks, weights)
        else:
            # Switch primary objective to L1 (MAE)
            loss = batch_mae
        
        # L2-SP regularization (optional)
        if bool(self._finetune_conf.get("enable", False)) and bool(
            self._finetune_conf.get("l2sp", {}).get("enable", False)
        ):
            alpha = float(self._finetune_conf.get("l2sp", {}).get("alpha", 1e-4))
            l2sp = self._l2sp_penalty()
            loss = loss + alpha * l2sp
        
        # Scale RMSE by out_norm to report in dB units
        rmse_normalized = torch.sqrt(batch_mse)
        rmse_db = rmse_normalized * self.out_norm
        
        return {
            "loss": loss,
            "rmse": rmse_db,
            "mae": batch_mae,
        }
    
    # Override to support discriminative LR and warmup
    def configure_optimizers(
        self,
    ) -> dict:
        if not bool(self._finetune_conf.get("enable", False)):
            return super().configure_optimizers()
        
        optimizer_conf: DictConfig = OmegaConf.create(self._optimizer_conf)
        base_lr: float = float(optimizer_conf["lr"]) if "lr" in optimizer_conf else 3e-4
        
        discr_conf: dict = self._finetune_conf.get("discriminative_lr", {})
        use_discr: bool = bool(discr_conf.get("enable", False))
        enc_factor: float = float(discr_conf.get("encoder_lr_factor", 0.1))
        
        params_encoder: list[nn.Parameter] = []
        params_other: list[nn.Parameter] = []
        for name, parameter in self._network.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".encoder." in name or name.startswith("unet.encoder") or name.startswith("encoder."):
                params_encoder.append(parameter)
            else:
                params_other.append(parameter)
        
        num_enc: int = sum(parameter.numel() for parameter in params_encoder)
        num_oth: int = sum(parameter.numel() for parameter in params_other)
        log.info(
            f"[finetune] param groups: encoder_params={len(params_encoder)} ({num_enc} weights), "
            f"other_params={len(params_other)} ({num_oth} weights), "
            f"discriminative_lr={'on' if use_discr else 'off'} (encoder_lr_factor={enc_factor})"
        )
        
        if use_discr:
            param_groups: list[dict] = [
                {
                    "params": params_encoder,
                    "lr": base_lr,
                    "lr_scale": enc_factor,
                },
                {
                    "params": params_other,
                    "lr": base_lr,
                    "lr_scale": 1.0,
                },
            ]
        else:
            combined_params: list[nn.Parameter] = params_encoder + params_other
            param_groups = [
                {
                    "params": combined_params,
                    "lr": base_lr,
                    "lr_scale": 1.0,
                }
            ]
        
        optimizer = hydra.utils.instantiate(
            optimizer_conf,
            params=param_groups,
            _convert_="all"
        )
        
        ret_opt: dict = {
            "optimizer": optimizer,
        }
        
        if self._scheduler_conf is not None:
            scheduler_conf: DictConfig = OmegaConf.create(self._scheduler_conf)
            monitor = scheduler_conf.get("monitor", None)
            if "monitor" in scheduler_conf:
                del scheduler_conf["monitor"]
            
            if "_target_" not in scheduler_conf:
                scheduler_keys: list[str] = list(scheduler_conf.keys())
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
            sch_opt: dict = {
                "scheduler": scheduler,
            }
            if monitor is not None:
                sch_opt["monitor"] = monitor
            if interval is not None:
                sch_opt["interval"] = interval
            if frequency is not None:
                sch_opt["frequency"] = frequency
            ret_opt["lr_scheduler"] = sch_opt
        
        return ret_opt
    
    def _calculate_epoch_metrics(self, outputs: list[Any]) -> dict:
        # init combined metrics with zero values
        combined_general_metrics = {k: 0 for k in outputs[0].keys()}
        
        # add all output values to combined_group_metrics
        for o in outputs:
            for k in o.keys():
                combined_general_metrics[k] += o[k]
        
        # compute means of metrics
        for k in outputs[0].keys():
            combined_general_metrics[k] /= len(outputs)
        
        # Derive log2 versions of key metrics for logging
        if "rmse" in combined_general_metrics or "mae" in combined_general_metrics:
            import torch as _torch
            
            eps = _torch.tensor(1e-8, dtype=_torch.float32)
            if "rmse" in combined_general_metrics:
                combined_general_metrics["log2_rmse"] = _torch.log2(
                    combined_general_metrics["rmse"].to(dtype=_torch.float32) + eps
                )
            if "mae" in combined_general_metrics:
                combined_general_metrics["log2_mae"] = _torch.log2(
                    combined_general_metrics["mae"].to(dtype=_torch.float32) + eps
                )
        
        # merge all
        epoch_metrics_sep = combined_general_metrics
        
        epoch_metrics_shared = {
            "learning_rate": self.trainer.optimizers[0].param_groups[0]["lr"]
        }
        
        if self.logger:
            self.logger.log_metrics(epoch_metrics_shared, self.trainer.current_epoch)
        else:
            log.info(f"""\n{epoch_metrics_shared}\n""")
        
        return epoch_metrics_sep
