import logging
import os
import shutil
import time
from collections import defaultdict
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
from src.utils.mlsp.augmentations import resize_db, resize_nearest
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
        
        self.training_step_outputs = []
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.loss = nn.MSELoss()
    
    def pred(self, batch):
        inputs, targets, masks, sample = batch

        # Use original dimensions for computing scale and resizing back
        old_h, old_w = sample["orig_H"], sample["orig_W"]
        # Match icassp2025 logic: derive normalized dims from IMG_TARGET_SIZE and original H/W
        scale_factor = min(IMG_TARGET_SIZE / old_h, IMG_TARGET_SIZE / old_w)
        norm_h, norm_w = int(old_h * scale_factor), int(old_w * scale_factor)

        pred = self.network(inputs.cuda(self._gpu).unsqueeze(0)).squeeze(0)
        pred = pred[torch.where(masks == 1)].reshape((norm_h, norm_w))
        pred = resize_db(pred.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
        pred = pred.detach().cpu().numpy()

        return {
            "pred": pred
        }
    
    # noinspection PyMethodOverriding
    def _step(self, batch, split_name, *args, **kwargs):
        inputs, targets, masks, sample = batch
        
        if split_name == "val" and sample["task_idx"][0].item() != -1:
            # No evaluation for sanity check
            if self.trainer.sanity_checking:
                return {
                    "loss": torch.Tensor([float("inf")]),
                    "mse": torch.Tensor([float("inf")]),
                }
            
            # Getting predictions
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
                mask_i = masks[i]
                targets_i = targets[i]
                pred_i = preds[i]
                sample_i = {k: sample[k][i] for k in sample.keys()}
                # Use original dimensions for computing scale and resizing back
                old_h, old_w = sample_i["orig_H"], sample_i["orig_W"]
                # Match icassp2025 logic for evaluation reshape dims
                scale_factor = min(IMG_TARGET_SIZE / old_h, IMG_TARGET_SIZE / old_w)
                norm_h, norm_w = int(old_h * scale_factor), int(old_w * scale_factor)
                try:
                    pred_i = pred_i.squeeze(0)[torch.where(mask_i == 1)].reshape((norm_h, norm_w))
                    pred_i = resize_db(pred_i.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
                    targets_i = targets_i.squeeze(0)[torch.where(mask_i == 1)].reshape((norm_h, norm_w))
                    targets_i = resize_db(targets_i.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
                    sparse_mask = (input_i[-1] == 0).unsqueeze(0)
                    sparse_mask = resize_nearest(sparse_mask, new_size=(old_h, old_w)).squeeze(0)
                    sample_se = se(pred_i, targets_i, sparse_mask)
                    sample_mse = sample_se / sparse_mask.sum()
                    mses.append(sample_mse)
                except Exception as ex:
                    log.error(ex)
                    continue
            
            return {
                "loss": torch.mean(torch.Tensor(mses)),
                "mse": torch.mean(torch.Tensor(mses)),
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
        
        return {
            "loss": loss,
            "mse": batch_mse,
        }
    
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
