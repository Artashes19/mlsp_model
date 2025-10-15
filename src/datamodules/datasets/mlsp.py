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
from src.utils.mlsp.featurizer import featurizer, sparse_sampling
from src.utils.mlsp.types import RadarSampleInputs

INITIAL_PIXEL_SIZE = 0.25
IMG_TARGET_SIZE = 640


class PathlossDataset(Dataset):
    
    def __init__(
        self,
        inputs_list,
        training: bool,
        mlsp_task1: bool,
        mlsp_task_idx: int,
        task_idx: Optional[int],
        pl_clip: Optional[int],
        use_fspl: bool,
        use_transmittance_loss: bool,
        inference: bool,
        sparsity_range: tuple[float, float],
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
        self.use_fspl = use_fspl
        self.use_transmittance_loss = use_transmittance_loss
        self.inference = inference
        self.sparsity_range = sparsity_range
        self.reps_per_epoch = reps_per_epoch
        self.augment_val = augment_val
        
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
        
        sample.x_ant += pad_left
        sample.y_ant += pad_top
        _, new_H, new_W = sample.input_img.shape
        sample.H, sample.W = new_H, new_W
        return sample
    
    def read_sample(self, inputs: Union[RadarSampleInputs, dict]) -> RadarSample:
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
            if output_img.size(0) == 1:  # If single channel, remove channel dimension
                output_img = output_img.squeeze(0)
        
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
            use_fspl=self.use_fspl,
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
        
        # Ensure the antenna is within bounds
        sample = self.pad_sample(sample)
        
        return sample
    
    def __getitem__(self, idx):
        idx = idx % len(self.inputs_list)
        inp = self.inputs_list[idx]
        sample = self.read_sample(inp)
        
        orig_h, orig_w = sample.H, sample.W
        if self.mlsp_task1:
            sample = sparse_sampling(
                sample,
                task_idx=self.mlsp_task_idx,
                inference=self.inference,
                sparsity_range=self.sparsity_range
            )

        sample = normalize_size(sample=sample, target_size=self.target_size)

        if (
            self.training or
            (self.augment_val and sample.output_img != "")
        ) and self.augmentations is not None:
            sample = self.augmentations(sample)

        output_tensor = sample.output_img if sample.output_img is not None else None

        input_tensor = featurizer(sample=sample)
        mask = sample.mask
        # Store original dimensions for algorithm to use
        sample_dict = sample.asdict()
        sample_dict['orig_H'] = orig_h
        sample_dict['orig_W'] = orig_w
        # Reset dimensions back to original for consistency with inference.py logic
        sample.H = orig_h
        sample.W = orig_w
        return input_tensor, output_tensor, mask, sample_dict
