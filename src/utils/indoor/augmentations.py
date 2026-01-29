import random
from typing import List

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.utils.indoor.types import RadarSample


def resize_bilinear(img, new_size):
    return TF.resize(img, new_size, interpolation=InterpolationMode.BILINEAR)


def normalize_size(sample: RadarSample, target_size) -> RadarSample:
    if 0 > sample.x_ant >= sample.W or 0 > sample.y_ant >= sample.H:
        print(
            f"Warning: antenna coords out of range. (x_ant={sample.x_ant}, y_ant={sample.y_ant}), (W={sample.W}, H={sample.H}) -> clamping to valid range."
        )
        sample.x_ant = max(0, min(sample.x_ant, sample.W - 1))
        sample.y_ant = max(0, min(sample.y_ant, sample.H - 1))
    
    C, H, W = sample.input_img.shape
    new_size = (target_size, target_size)
    
    # Scale factors for x and y (no aspect ratio preservation)
    scale_x = target_size / W
    scale_y = target_size / H
    
    reflectance = sample.input_img[0:1]  # First channel with dimension [1, H, W]
    transmittance = sample.input_img[1:2]  # Second channel with dimension [1, H, W]
    distance = sample.input_img[2:3]  # Third channel with dimension [1, H, W]
    
    reflectance_resized = resize_bilinear(reflectance, new_size)
    transmittance_resized = resize_bilinear(transmittance, new_size)
    distance_resized = resize_bilinear(distance, new_size)
    mask_resized = resize_bilinear(sample.mask.unsqueeze(0), new_size).squeeze(0)
    
    sample.x_ant = int(sample.x_ant * scale_x)
    sample.y_ant = int(sample.y_ant * scale_y)
    
    sample.input_img = torch.zeros(
        (max(3, C), target_size, target_size), dtype=torch.float32, device=torch.device("cpu")
        )
    sample.input_img[0:1] = reflectance_resized
    sample.input_img[1:2] = transmittance_resized
    sample.input_img[2:3] = distance_resized
    
    if sample.floor_plan is not None:
        sample.floor_plan = resize_bilinear(sample.floor_plan.unsqueeze(0), new_size).squeeze(0)
    
    if sample.output_img != "":
        sample.output_img = resize_bilinear(sample.output_img.unsqueeze(0), new_size).squeeze(0)
    
    sample.H = sample.W = target_size
    sample.mask = mask_resized
    
    return sample


class BaseAugmentation:
    """Base class for all augmentations"""
    
    def __call__(self, sample: RadarSample) -> RadarSample:
        raise NotImplementedError


class CardinalRotationAugmentation(BaseAugmentation):
    """Augmentation that applies lossless 90/180/270 degree rotations."""
    
    def __init__(self, p: float = 1.0):
        """
        Args:
            p: Probability of applying cardinal rotation (default 1.0 = always apply)
        """
        self.p = p
    
    def __call__(self, sample: RadarSample) -> RadarSample:
        if random.random() > self.p:
            return sample
        
        return self._apply_cardinal_rotation(sample)
    
    def _apply_cardinal_rotation(self, sample: RadarSample) -> RadarSample:
        """
        Rotate by one of {90, 180, 270} degrees *losslessly* using torch.rot90.
        We also must update x_ant, y_ant, azimuth accordingly.
        """
        # Randomly choose 90°, 180°, or 270° (k=1,2,3). If you want to allow 0°, add k=0.
        k = random.choice([1, 2, 3])
        
        old_H, old_W = sample.H, sample.W
        sample.input_img = torch.rot90(sample.input_img, k, (1, 2))
        new_H, new_W = sample.input_img.shape[1], sample.input_img.shape[2]
        
        if k == 1:  # 90 deg counter-clockwise
            new_x = sample.y_ant
            new_y = old_W - sample.x_ant - 1
            sample.azimuth = (sample.azimuth + 90) % 360
        elif k == 2:  # 180 deg
            new_x = old_W - sample.x_ant - 1
            new_y = old_H - sample.y_ant - 1
            sample.azimuth = (sample.azimuth + 180) % 360
        elif k == 3:  # 270 deg
            new_x = old_H - sample.y_ant - 1
            new_y = sample.x_ant
            sample.azimuth = (sample.azimuth + 270) % 360
        
        sample.x_ant, sample.y_ant = new_x, new_y
        if sample.output_img is not None:
            sample.output_img = torch.rot90(sample.output_img, k, (0, 1))
        if sample.floor_plan is not None:
            sample.floor_plan = torch.rot90(sample.floor_plan, k, (0, 1))
        if sample.mask is not None:
            sample.mask = torch.rot90(sample.mask, k, (0, 1))
        
        sample.H, sample.W = new_H, new_W
        return sample


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations in sequence"""
    
    def __init__(self, augmentations: List[BaseAugmentation], training: bool = True):
        """
        Args:
            augmentations: List of augmentation instances
            training: Whether to apply augmentations (only in training mode)
        """
        self.training = training
        self.augmentations = augmentations
    
    def __call__(self, sample: RadarSample) -> RadarSample:
        """Apply all augmentations in sequence to the sample.
        
        Args:
            sample: RadarSample instance
            
        Returns:
            Augmented RadarSample instance
        """
        if not self.training:
            return sample
        
        # Apply each augmentation in sequence
        for aug in self.augmentations:
            sample = aug(sample)
        
        return sample
