import logging
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from kaggle import KaggleApi
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.algorithms.algorithm_base import AlgorithmBase
from src.datamodules.datasets.mlsp import IMG_TARGET_SIZE
from src.utils import CompileParams
from src.utils.mlsp.augmentations import resize_linear
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
            log.info("Using pure MSE objective")
        
        self.training_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.loss = nn.MSELoss()

        # Finetune configuration (optional)
        self._finetune_conf: DictConfig | dict | None = kwargs.get("finetune", None)
        if self._finetune_conf is None:
            # default disabled
            self._finetune_conf = {
                "enable": False
            }
        try:
            log.info(f"[finetune] enable={bool(self._finetune_conf.get('enable', False))}, "
                     f"ckpt_path={self._finetune_conf.get('ckpt_path', None)}, "
                     f"freeze_encoder_epochs={int(self._finetune_conf.get('freeze_encoder_epochs', 0))}, "
                     f"discriminative_lr={self._finetune_conf.get('discriminative_lr', {})}, "
                     f"warmup={self._finetune_conf.get('warmup', {})}, "
                     f"bn_recalibration={self._finetune_conf.get('bn_recalibration', {})}, "
                     f"l2sp={self._finetune_conf.get('l2sp', {})}")
        except Exception:
            pass

        # Placeholders for finetune utilities
        self._pretrained_weights: dict[str, torch.Tensor] | None = None
        self._freeze_encoder_epochs: int = int(self._finetune_conf.get("freeze_encoder_epochs", 0))
        self._encoder_frozen: bool = False
        
    
    def pred(self, batch):
        inputs, targets, masks, sample = batch

        # Use pixel_size for exact reverse scaling (no floating point errors)
        original_pixel_size = 0.25  # Known constant
        current_pixel_size = sample["pixel_size"]
        reverse_scale_factor = original_pixel_size / current_pixel_size

        # Get exact original dimensions
        current_h, current_w = 640, 640  # Normalized size
        old_h = int(current_h * reverse_scale_factor)
        old_w = int(current_w * reverse_scale_factor)

        # Calculate pre-padding dimensions
        scale_factor_forward = min(IMG_TARGET_SIZE / old_h, IMG_TARGET_SIZE / old_w)
        resized_w = int(old_w * scale_factor_forward)

        pred = self.network(inputs.cuda(self._gpu).unsqueeze(0)).squeeze(0)

        # Cut prediction to remove padding (640x640 → 640xresized_w)
        pred_cut = pred[:, :resized_w]

        # Resize prediction back to exact original dimensions
        pred_final = resize_linear(pred_cut.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
        pred = pred_final.detach().cpu().numpy()

        return {
            "pred": pred
        }

    # -------------------- Finetune helpers --------------------
    def _load_weights_only(self, ckpt_path: str) -> None:
        """Load only network weights from a PL checkpoint, ignore optimizer/scheduler."""
        device = next(self._network.parameters()).device
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        net_state = self._network.state_dict()
        remapped = {}
        matched, total = 0, 0
        for k in net_state.keys():
            total += 1
            for prefix in ("_network.", "network.", ""):
                cand = f"{prefix}{k}" if prefix else k
                if cand in state_dict:
                    remapped[k] = state_dict[cand]
                    matched += 1
                    break
        missing = [k for k in net_state.keys() if k not in remapped]
        if missing:
            log.info(f"Weights-only load: matched {matched}/{total}; missing {len(missing)} params")
        self._network.load_state_dict(remapped, strict=False)
        self._network.to(device)
        log.info(f"Loaded weights-only from {ckpt_path}")

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
                device = self._gpu if self._gpu is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
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
        
        if split_name == "val" and sample["task_idx"][0].item() not in [-1, -2]:
            # No evaluation for sanity check
            if self.trainer.sanity_checking:
                return {
                    "loss": torch.Tensor([float("inf")]),
                    "mse": torch.Tensor([float("inf")]),
                }
            
            # Getting predictions
            log.info("[validation] Kaggle submission branch engaged")
            pred_path = "./task1" if sample["task_idx"][0] == 1 else "./task2"
            if os.path.exists(pred_path):
                shutil.rmtree(pred_path)
            os.makedirs(pred_path, exist_ok=True)
            for i in tqdm(list(range(len(targets)))):
                sample_i = {k: sample[k][i] for k in sample.keys()}
                alg_out = self.pred(
                    (inputs[i], targets[i], masks[i], sample_i)
                )
                pred_img = Image.fromarray(alg_out["pred"]).convert("RGB")
                pred_img.save(os.path.join(pred_path, f"{sample['file_name'][i]}"))
            
            # Creating predictions dataframe
            data = []
            for file_name in os.listdir(pred_path):
                if file_name.endswith(".png"):
                    file_path = os.path.join(pred_path, file_name)
                    image = Image.open(file_path).convert("L")
                    pl_array = np.array(image)
                    
                    flat_pl = pl_array.flatten()
                    for idx, value in enumerate(flat_pl):
                        id_str = f"{file_name.split('.')[0]}_{idx}"
                        data.append((id_str, value))
            
            # Save predictions to CSV
            df = pd.DataFrame(data, columns=["ID", "PL"])
            df = df.groupby("ID", as_index=False).mean()
            pred_file = os.path.join(pred_path, f"epoch_{self.trainer.current_epoch}.csv")
            df.to_csv(pred_file, index=False)
            
            # Submit to Kaggle
            if sample["task_idx"][0] == 1:
                competition = "iprm-task-1"
            else:
                competition = "indoor-pathloss-radio-map-prediction-task-2"
            
            api = KaggleApi()
            api.authenticate()
            submission = self._submit_solution_to_kaggle(
                api, pred_file, competition,
                f"Submission from epoch {self.trainer.current_epoch}"
            )
            kaggle_mse = self._poll_submission_score(api, competition, submission)
            
            return {
                "loss": torch.Tensor([kaggle_mse]),
                "mse": torch.Tensor([kaggle_mse]),
            }
        
        preds = self._network(inputs)

        
        if split_name == "train":
            # weights = (inputs[:, -1] == 0) * 9 + 1
            weights = torch.ones_like(inputs[:, -1])
            return self.get_metrics(preds, targets, masks, weights)
        else:
            mses = []
            for i in range(targets.shape[0]):
                input_i = inputs[i]
                targets_i = targets[i]
                pred_i = preds[i]
                sample_i = {k: sample[k][i] for k in sample.keys()}

                # Use pixel_size for exact reverse scaling (no floating point errors)
                original_pixel_size = 0.25  # Known constant
                current_pixel_size = sample_i["pixel_size"]
                reverse_scale_factor = original_pixel_size / current_pixel_size

                # Get exact original dimensions
                current_h, current_w = 640, 640  # Normalized size
                old_h = int(current_h * reverse_scale_factor)
                old_w = int(current_w * reverse_scale_factor)

                # Calculate pre-padding dimensions (before padding to 640x640)
                scale_factor_forward = min(IMG_TARGET_SIZE / old_h, IMG_TARGET_SIZE / old_w)
                resized_w = int(old_w * scale_factor_forward)

                try:
                    # Cut prediction to remove padding (640x640 → 640xresized_w)
                    pred_cut = pred_i.squeeze(0)[:, :resized_w]

                    # Resize prediction back to exact original dimensions
                    pred_final = resize_linear(pred_cut.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)

                    # Target is also in normalized 640x640 space - apply same processing
                    target_normalized = targets_i.squeeze(0)  # [640, 640]
                    target_cut = target_normalized[:, :resized_w]
                    target_final = resize_linear(target_cut.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)

                    mse = torch.mean((pred_final - target_final) ** 2)
                    mses.append(mse)

                except Exception as ex:
                    log.error(f"Error in validation sample {i}: {ex}")
                    continue

            return {
                "loss": torch.mean(torch.Tensor(mses)) if mses else torch.Tensor([float("inf")]),
                "mse": torch.mean(torch.Tensor(mses)) if mses else torch.Tensor([float("inf")]),
            }
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.training_step_outputs.append(outputs)
    
    @staticmethod
    def _submit_solution_to_kaggle(api: KaggleApi, file_path: str, competition: str, message: str):
        """
        Submits the CSV file to Kaggle and returns the submission object.
        """
        return api.competition_submit(file_path, message, competition)
    
    @staticmethod
    def _poll_submission_score(api: KaggleApi, competition: str, submission) -> float:
        """
        Polls Kaggle until the submission completes and returns the public_score (MSE).
        """
        result = None
        i = 0
        while result is None:
            submission_results = api.competition_submissions(competition=competition)
            latest = sorted(submission_results, key=lambda x: x.date, reverse=True)[0]
            if str(latest.status) == "SubmissionStatus.COMPLETE":
                result = latest
                break
            
            if result is not None:
                break
            time.sleep(5)  # Wait between checks
            i += 1
            if i > 24:
                log.warning("Kaggle submission timed out.")
                break
        
        return float(result.public_score) if result is not None else float("inf")  # Kaggle's published MSE score
    
    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.validation_step_outputs[dataloader_idx].append(outputs)
    
    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        outputs = AlgorithmBase.convert_to_numpy(outputs)
        self.test_step_outputs[dataloader_idx].append(outputs)
    
    def get_metrics(self, preds, targets, masks, weights):
        
        batch_se = se(preds, targets, masks, weights)
        batch_mse = batch_se / masks.sum()
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
            loss = batch_mse

        # L2-SP regularization (optional)
        if bool(self._finetune_conf.get("enable", False)) and bool(self._finetune_conf.get("l2sp", {}).get("enable", False)):
            alpha = float(self._finetune_conf.get("l2sp", {}).get("alpha", 1e-4))
            l2sp = self._l2sp_penalty()
            loss = loss + alpha * l2sp
        
        return {
            "loss": loss,
            "mse": batch_mse,
        }

    # Override to support discriminative LR and warmup
    def configure_optimizers(self):
        if not bool(self._finetune_conf.get("enable", False)):
            return super().configure_optimizers()

        # Base optimizer settings
        import torch.optim as optim
        base_conf = self._optimizer_conf
        try:
            base_lr = float(base_conf.get("lr", 3e-4))  # OmegaConf-like access
        except Exception:
            base_lr = 3e-4

        discr_conf = self._finetune_conf.get("discriminative_lr", {})
        use_discr = bool(discr_conf.get("enable", False))
        enc_factor = float(discr_conf.get("encoder_lr_factor", 0.1))

        # Build param groups
        params_encoder = []
        params_other = []
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            if ".encoder." in name or name.startswith("unet.encoder") or name.startswith("encoder."):
                params_encoder.append(p)
            else:
                params_other.append(p)

        try:
            num_enc = sum(p.numel() for p in params_encoder)
            num_oth = sum(p.numel() for p in params_other)
            log.info(f"[finetune] param groups: encoder_params={len(params_encoder)} ({num_enc} weights), "
                     f"other_params={len(params_other)} ({num_oth} weights), "
                     f"discriminative_lr={'on' if use_discr else 'off'} (encoder_lr_factor={enc_factor})")
        except Exception:
            pass

        if use_discr:
            param_groups = [
                {"params": params_encoder, "lr": base_lr * enc_factor},
                {"params": params_other, "lr": base_lr},
            ]
        else:
            param_groups = [{"params": params_encoder + params_other, "lr": base_lr}]

        optimizer = optim.Adam(param_groups)

        # Warmup scheduler (epoch-based)
        warm_conf = self._finetune_conf.get("warmup", {})
        use_warm = bool(warm_conf.get("enable", False))
        if use_warm:
            warm_epochs = int(warm_conf.get("warmup_epochs", 5))
            warm_factor = float(warm_conf.get("warmup_factor", 0.1))
            log.info(f"[finetune] warmup enabled: epochs={warm_epochs}, factor={warm_factor}")

            def lr_lambda(epoch):
                if epoch >= warm_epochs:
                    return 1.0
                return warm_factor + (1.0 - warm_factor) * (epoch / max(1, warm_epochs))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": None,
                },
            }
        else:
            return {"optimizer": optimizer}
    
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

        # derive RMSE from MSE (tensors throughout in this codebase)
        if "mse" in combined_general_metrics:
            import torch as _torch
            combined_general_metrics["rmse"] = _torch.sqrt(combined_general_metrics["mse"])    
        
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
