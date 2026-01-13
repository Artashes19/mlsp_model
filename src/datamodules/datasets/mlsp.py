import json
import os
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.utils import normalize_size, RadarSample
from src.utils.mlsp.augmentations import AugmentationPipeline
from src.utils.mlsp.featurizer import featurizer
from src.utils.mlsp.types import RadarSampleInputs

INITIAL_PIXEL_SIZE = 0.25
IMG_TARGET_SIZE = 256


class PathlossDataset(Dataset):
    
    def __init__(
        self,
        inputs_list,
        training: bool,
        mlsp_task1: bool,
        mlsp_task_idx: int,
        task_idx: Optional[int],
        pl_clip: Optional[int],
        use_approximator_feature: bool,
        use_transmittance_loss: bool,
        inference: bool,
        reps_per_epoch: int,
        augment_val: bool,
        augmentations: Optional[AugmentationPipeline],
        *args, **kwargs
    ):
        self.inputs_list = inputs_list
        self.training = training
        self.augmentations = augmentations
        self.mlsp_task1 = mlsp_task1
        self.task_idx = task_idx
        self.mlsp_task_idx = mlsp_task_idx
        self.pl_clip = pl_clip
        self.use_approximator_feature = use_approximator_feature
        self.use_transmittance_loss = use_transmittance_loss
        self.inference = inference
        self.reps_per_epoch = reps_per_epoch
        self.augment_val = augment_val
        
        # Ensure sparse_range is a list/tuple of floats
        sparse_range_val = kwargs["sparse_range"]
        if isinstance(sparse_range_val, str):
            # Parse string representation if it comes as a string (e.g. "[0.0, 0.01]")
            try:
                cleaned = sparse_range_val.strip("[]()")
                self.sparse_range = [float(x.strip()) for x in cleaned.split(",")]
            except ValueError:
                raise ValueError(f"Could not parse sparse_range string: {sparse_range_val}")
        else:
            # Convert from omegaconf.ListConfig or other iterables to a plain list
            self.sparse_range = list(sparse_range_val)
            
        if not isinstance(self.sparse_range, (list, tuple)) or len(self.sparse_range) != 2:
            raise ValueError(f"sparse_range must be a list/tuple of 2 floats, got {self.sparse_range}")
        
        # Modality dropout parameters
        self.modality_dropout_prob = float(kwargs.get("modality_dropout_prob", 0.6666))
        self.sparse_dropout_given_dropout = float(kwargs.get("sparse_dropout_given_dropout", 0.5))
            
        self.target_size = IMG_TARGET_SIZE
    
    def __len__(self):
        if self.inference:
            return len(self.inputs_list)
        return len(self.inputs_list) * self.reps_per_epoch
    
    @staticmethod
    def pad_sample(sample: RadarSample) -> RadarSample:
        C, H, W = sample.input_img.shape
        x_ant, y_ant = sample.x_ant, sample.y_ant
        
        pad_left = int(max(0, -x_ant))
        pad_right = int(max(0, x_ant - (W - 1)))
        pad_top = int(max(0, -y_ant))
        pad_bottom = int(max(0, y_ant - (H - 1)))
        
        if not any([pad_left, pad_right, pad_top, pad_bottom]):
            return sample
        
        sample.input_img = F.pad(
            sample.input_img.unsqueeze(0),  # (C, H, W) -> (1, C, H, W)
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0
        ).squeeze(0)  # -> (C, new_H, new_W)
        
        if sample.output_img is not None:
            sample.output_img = F.pad(
                sample.output_img.unsqueeze(0),  # (H, W) or (C, H, W)
                (pad_left, pad_right, pad_top, pad_bottom),
                value=0
            ).squeeze(0)
        
        sample.mask = F.pad(
            sample.mask.unsqueeze(0),  # (H, W) -> (1, H, W)
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0
        ).squeeze(0)  # (new_H, new_W)
        
        if sample.floor_plan is not None:
            sample.floor_plan = F.pad(
                sample.floor_plan.unsqueeze(0),
                (pad_left, pad_right, pad_top, pad_bottom),
                value=0
            ).squeeze(0)

        sample.x_ant += pad_left
        sample.y_ant += pad_top
        _, new_H, new_W = sample.input_img.shape
        sample.H, sample.W = new_H, new_W
        return sample
    
    def read_sample_synthetic(self, inputs: dict) -> RadarSample:
        npz_path = inputs["npz_file"]
        json_path = inputs.get("json_file")
        file_name = inputs.get("file_name", os.path.basename(npz_path))
        data = np.load(npz_path)
        with open(json_path, "r") as f:
            meta = json.load(f)
        
        # Enforce presence of required fields; mask optional, pathloss required for supervision
        required_keys = ["reflectance", "transmittance", "pathloss"]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise RuntimeError(
                f"Synthetic NPZ missing required arrays {missing} in {npz_path}. Training cannot proceed."
            )
        reflectance = data["reflectance"].astype(np.float32)
        transmittance = data["transmittance"].astype(np.float32)
        mask_np = (data["mask"].astype(np.float32) if "mask" in data else np.ones_like(reflectance, dtype=np.float32))
        pathloss = data["pathloss"].astype(np.float32)
        # Validate shapes match
        H, W = reflectance.shape
        for arr_name, arr in {
            "transmittance": transmittance,
            "mask": mask_np,
            "pathloss": pathloss,
        }.items():
            if arr.shape != (H, W):
                raise RuntimeError(
                    f"Array '{arr_name}' shape {arr.shape} != reflectance shape {(H, W)} in {npz_path}"
                )
        
        H, W = reflectance.shape
        
        # Enforce required JSON fields
        if "antenna" not in meta or not isinstance(meta["antenna"], dict):
            raise RuntimeError(f"Missing 'antenna' object in JSON {json_path}")
        ant = meta["antenna"]
        if "x_px" not in ant or "y_px" not in ant:
            raise RuntimeError(f"Missing antenna 'x_px'/'y_px' in JSON {json_path}")
        if "frequency_MHz" not in meta:
            raise RuntimeError(f"Missing 'frequency_MHz' in JSON {json_path}")
        if "pixel_size_m" not in meta:
            raise RuntimeError(f"Missing 'pixel_size_m' in JSON {json_path}")
        
        x_ant = float(ant["x_px"])  # will raise if not numeric
        y_ant = float(ant["y_px"])  # will raise if not numeric
        pixel_size = float(meta["pixel_size_m"])  # will raise if not numeric
        
        yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
        dist_px = np.hypot(xx - x_ant, yy - y_ant)
        dist_m = (dist_px * pixel_size).astype(np.float32)
        
        input_img = torch.zeros((3, H, W), dtype=torch.float32)
        input_img[0] = torch.from_numpy(reflectance)
        input_img[1] = torch.from_numpy(transmittance)
        input_img[2] = torch.from_numpy(dist_m)
        
        output_img = torch.from_numpy(pathloss)
        # Normalize targets to [0, 1] by dividing by max dB (160)
        output_img = output_img / 160.0
        
        freq_MHz = float(meta["frequency_MHz"])  # required above
        radiation_pattern = torch.ones(360, dtype=torch.float32)
        
        if self.pl_clip is not None and not self.inference:
            pl_clip = torch.tensor(self.pl_clip, dtype=torch.float32)
        else:
            pl_clip = float("inf")
        
        sample = RadarSample(
            file_name=file_name,
            task_idx=self.task_idx,
            pl_clip=pl_clip,
            use_approximator_feature=self.use_approximator_feature,
            use_transmittance_loss=self.use_transmittance_loss,
            H=H,
            W=W,
            x_ant=x_ant,
            y_ant=y_ant,
            azimuth=0.0,
            freq_MHz=freq_MHz,
            input_img=input_img,
            output_img=output_img,
            radiation_pattern=radiation_pattern,
            pixel_size=pixel_size,
            mask=torch.from_numpy(mask_np),
        )
        sample = self.pad_sample(sample)
        return sample
    
    def read_sample_icassp(self, inputs: Union[RadarSampleInputs, dict]) -> RadarSample:
        if isinstance(inputs, RadarSampleInputs):
            inputs = inputs.asdict()
        file_name = inputs["file_name"]
        freq_MHz = inputs["freq_MHz"]
        input_file = inputs["input_file"]
        output_file = inputs["output_file"]
        position_file = inputs["position_file"]
        sampling_position = inputs["sampling_position"]
        radiation_pattern_file = inputs["radiation_pattern_file"]
        
        input_img = read_image(input_file).float()
        C, H, W = input_img.shape
        
        if not os.path.exists(output_file):
            output_img = ""
        else:
            output_img = read_image(output_file).float()
            if output_img.size(0) == 1:
                output_img = output_img.squeeze(0)
            # Normalize targets to [0, 1] by dividing by max dB (160)
            output_img = output_img / 160.0
        sampling_positions = pd.read_csv(position_file)
        x_ant, y_ant, azimuth = sampling_positions.loc[int(sampling_position), ["Y", "X", "Azimuth"]]
        radiation_pattern_np = np.genfromtxt(radiation_pattern_file, delimiter=',')
        radiation_pattern = torch.from_numpy(radiation_pattern_np).float()
        
        if self.pl_clip is not None and not self.inference:
            pl_clip = torch.tensor(self.pl_clip, dtype=torch.float32)
        else:
            pl_clip = float("inf")
        
        sample = RadarSample(
            file_name=file_name,
            task_idx=self.task_idx,
            pl_clip=pl_clip,
            use_approximator_feature=self.use_approximator_feature,
            use_transmittance_loss=self.use_transmittance_loss,
            H=H,
            W=W,
            x_ant=x_ant,
            y_ant=y_ant,
            azimuth=azimuth,
            freq_MHz=freq_MHz,
            input_img=input_img,
            output_img=output_img,
            radiation_pattern=radiation_pattern,
            pixel_size=INITIAL_PIXEL_SIZE,
            mask=torch.ones_like(input_img[0]),
        )
        sample = self.pad_sample(sample)
        return sample
    
    def read_sample(self, inputs: Union[RadarSampleInputs, dict]) -> RadarSample:
        if isinstance(inputs, dict) and "npz_file" in inputs:
            return self.read_sample_synthetic(inputs)
        return self.read_sample_icassp(inputs)
    
    def __getitem__(self, idx):
        idx = idx % len(self.inputs_list)
        inp = self.inputs_list[idx]
        sample = self.read_sample(inp)
        
        orig_h, orig_w = sample.H, sample.W
        
        sample = normalize_size(sample=sample, target_size=self.target_size)
        
        if (
            self.training or
            (self.augment_val and sample.output_img != "")
        ) and self.augmentations is not None:
            sample = self.augmentations(sample)
        
        output_tensor = sample.output_img if sample.output_img is not None else None
        
        input_tensor = featurizer(
            sample=sample,
            sparse_range=self.sparse_range,
            modality_dropout_prob=self.modality_dropout_prob,
            sparse_dropout_given_dropout=self.sparse_dropout_given_dropout
        )
        mask = sample.mask
        # Store original dimensions for algorithm to use
        # Return only lightweight metadata to minimize batch transfer overhead
        meta = {
            "file_name": sample.file_name,
            # keep as a plain float; default_collate will tensorize to (B,)
            "pixel_size": float(sample.pixel_size),
            # keep task_idx as tensor so downstream checks like sample["task_idx"][0].item() work
            "task_idx": torch.tensor(sample.task_idx, dtype=torch.int64),
        }
        # Reset dimensions back to original for consistency with inference.py logic
        sample.H = orig_h
        sample.W = orig_w
        return input_tensor, output_tensor, mask, meta
