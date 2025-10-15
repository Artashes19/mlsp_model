import random
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.utils.mlsp.featurizer import calculate_transmittance_loss
from src.utils.mlsp.types import RadarSample


def resize_nearest(img, new_size):
    return TF.resize(img, new_size, interpolation=InterpolationMode.NEAREST_EXACT)


def resize_linear(img, new_size):
    return TF.resize(img, new_size, interpolation=InterpolationMode.BILINEAR)


def resize_db(img, new_size):
    lin_energy = 10.0 ** (img / 10.0)
    lin_rs = TF.resize(lin_energy, new_size, interpolation=InterpolationMode.BILINEAR)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = 10.0 * torch.log10(lin_rs[valid_mask])
    
    return img_rs


def rotate_nearest(img, angle):
    return TF.rotate(img, angle, interpolation=InterpolationMode.NEAREST, fill=0, expand=True)


def rotate_linear(img, angle):
    return TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True)


def rotate_db(img, angle):
    lin_energy = 10.0 ** (img / 10.0)
    lin_rs = TF.rotate(lin_energy, angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = 10.0 * torch.log10(lin_rs[valid_mask])
    return img_rs


def normalize_size(sample: RadarSample, target_size) -> RadarSample:
    if 0 > sample.x_ant >= sample.W or 0 > sample.y_ant >= sample.H:
        print(
            f"Warning: antenna coords out of range. (x_ant={sample.x_ant}, y_ant={sample.y_ant}), (W={sample.W}, H={sample.H}) -> clamping to valid range."
        )
        sample.x_ant = max(0, min(sample.x_ant, sample.W - 1))
        sample.y_ant = max(0, min(sample.y_ant, sample.H - 1))
    
    C, H, W = sample.input_img.shape
    scale_factor = min(target_size / H, target_size / W)
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    new_size = (new_h, new_w)
    
    reflectance = sample.input_img[0:1]  # First channel with dimension [1, H, W]
    transmittance = sample.input_img[1:2]  # Second channel with dimension [1, H, W]
    sparse_sample = sample.input_img[3:4]  # Fourth channel with dimension [1, H, W]
    
    reflectance_resized = resize_nearest(reflectance, new_size)
    transmittance_resized = resize_nearest(transmittance, new_size)
    sparse_sample_resized = resize_nearest(sparse_sample, new_size)
    mask_resized = resize_nearest(sample.mask.unsqueeze(0), new_size).squeeze(0)
    
    sample.x_ant = int(sample.x_ant * scale_factor)
    sample.y_ant = int(sample.y_ant * scale_factor)
    
    sample.pixel_size /= scale_factor  # Update pixel size (divide by scale factor)
    
    sample.input_img = torch.zeros((C, target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.input_img[0:1, :new_h, :new_w] = reflectance_resized
    sample.input_img[1:2, :new_h, :new_w] = transmittance_resized
    sample.input_img[3:4, :new_h, :new_w] = sparse_sample_resized
    
    if sample.output_img != "":
        resized_output = resize_db(sample.output_img.unsqueeze(0), new_size).squeeze(0)
        padded_output = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
        padded_output[:new_h, :new_w] = resized_output
        sample.output_img = padded_output
    
    sample.H = sample.W = target_size
    
    sample.mask = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.mask[:new_h, :new_w] = mask_resized
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(target_size, dtype=torch.float32, device=torch.device('cpu')),
        torch.arange(target_size, dtype=torch.float32, device=torch.device('cpu')),
        indexing='ij'
    )
    
    sample.input_img[2, :, :] = torch.sqrt(
        (x_grid - sample.x_ant) ** 2 + (y_grid - sample.y_ant) ** 2
    ) * sample.pixel_size
    
    return sample


class BaseAugmentation:
    """Base class for all augmentations"""
    
    def __call__(self, sample: RadarSample) -> RadarSample:
        raise NotImplementedError


class GeometricAugmentation(BaseAugmentation):
    
    def __init__(
        self,
        p: float,
        walls_p: Optional[int],
        transmittance_range: Optional[tuple[int, int]],
        flip_vertical: bool,
        flip_horizontal: bool,
        angle_range: Optional[tuple[float, float]],
        cardinal_rotation: bool,
        scale_range: Optional[tuple[float, float]],
    ):
        self.p = p
        self.walls_p = walls_p
        self.transmittance_range = transmittance_range
        self.scale_range = scale_range
        self.angle_range = angle_range
        self.cardinal_rotation = cardinal_rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
    
    def _apply_distance_scaling(self, sample: RadarSample, scale_factor: float) -> RadarSample:
        distances = sample.input_img[2]
        
        scaled_distances = distances * scale_factor
        fspl_adjustment = 20.0 * np.log10(scale_factor)
        
        sample.input_img[2] = scaled_distances
        sample.input_img[-1][sample.input_img[-1] != 0] = fspl_adjustment
        sample.output_img += fspl_adjustment
        sample.pixel_size *= scale_factor
        
        if sample.pl_clip != float("inf"):
            sample.pl_clip += fspl_adjustment
        
        return sample
    
    def _apply_rotation(self, sample: RadarSample, angle: float) -> RadarSample:
        old_H, old_W = sample.H, sample.W
        antenna_img = torch.zeros((old_H, old_W), dtype=torch.float32)
        antenna_img[int(round(sample.y_ant)), int(round(sample.x_ant))] = 100.0
        antenna_rot = rotate_linear(antenna_img.unsqueeze(0), angle).squeeze(0)
        coords = (antenna_rot > 0).nonzero(as_tuple=False)
        
        new_ay, new_ax = coords[antenna_rot[coords[:, 0], coords[:, 1]] == antenna_rot.max()][0].tolist()
        
        sample.x_ant, sample.y_ant = float(new_ax), float(new_ay)
        sample.azimuth = (sample.azimuth + angle) % 360
        
        reflectance = sample.input_img[0:1]  # (1, H, W)
        transmittance = sample.input_img[1:2]  # (1, H, W)
        distance = sample.input_img[2:3]  # (1, H, W)
        sparse_sample = sample.input_img[3:4]  # (1, H, W)
        
        rot_reflectance = rotate_nearest(reflectance, angle)
        rot_transmittance = rotate_nearest(transmittance, angle)
        rot_distance = rotate_nearest(distance, angle)
        rot_sparse_sample = rotate_nearest(sparse_sample, angle)
        
        if sample.output_img is not None:
            out_expanded = sample.output_img.unsqueeze(0)  # (1,H,W)
            rot_output = rotate_db(out_expanded, angle).squeeze(0)
            sample.output_img = rot_output
        
        if sample.mask is not None:
            mask_expanded = sample.mask.unsqueeze(0)
            rot_mask = rotate_nearest(mask_expanded, angle).squeeze(0)
            sample.mask = rot_mask
        
        _, new_H, new_W = rot_reflectance.shape
        sample.H, sample.W = new_H, new_W
        sample.input_img = torch.cat([rot_reflectance, rot_transmittance, rot_distance, rot_sparse_sample], dim=0)
        
        sample = normalize_size(sample=sample, target_size=old_H)
        
        return sample
    
    def _apply_flipping(self, sample: RadarSample, flip_h: bool, flip_v: bool) -> RadarSample:
        if not (flip_h or flip_v):
            return sample
        
        if flip_h:
            sample.input_img = TF.hflip(sample.input_img)
        if flip_v:
            sample.input_img = TF.vflip(sample.input_img)
        
        if sample.output_img is not None:
            output_expanded = sample.output_img.unsqueeze(0)
            if flip_h:
                output_expanded = TF.hflip(output_expanded)
            if flip_v:
                output_expanded = TF.vflip(output_expanded)
            sample.output_img = output_expanded.squeeze(0)
        
        if flip_h:
            sample.x_ant = sample.W - sample.x_ant
            sample.azimuth = (180 - sample.azimuth) % 360
        if flip_v:
            sample.y_ant = sample.H - sample.y_ant
            sample.azimuth = (360 - sample.azimuth) % 360
        
        return sample
    
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
        if sample.mask is not None:
            sample.mask = torch.rot90(sample.mask, k, (0, 1))
        
        sample.H, sample.W = new_H, new_W
        return sample
    
    def _apply_walls(self, sample: RadarSample) -> RadarSample:
        new_walls = torch.zeros((sample.H, sample.W), dtype=torch.float32)
        transmittance: torch.Tensor = sample.input_img[1][sample.mask == 1]
        
        # Randomly choosing the number of vertical and horizontal walls
        max_wall_count = transmittance.size()[0] / (transmittance != 0).sum().item()
        num_vertical_walls = np.random.randint(1, max_wall_count / 3)
        num_horizontal_walls = np.random.randint(1, max_wall_count / 3)
        
        # Choosing random positions for the walls
        vertical_walls = np.random.choice(sample.W, num_vertical_walls, replace=False)
        horizontal_walls = np.random.choice(sample.H, num_horizontal_walls, replace=False)
        
        # Getting random values to the walls
        max_transmittance = np.random.randint(self.transmittance_range[0], self.transmittance_range[1], )
        vertical_wall_values = torch.randint(
            1, max_transmittance,
            (num_vertical_walls,), dtype=torch.float32
        )
        horizontal_wall_values = torch.randint(
            1, max_transmittance,
            (num_horizontal_walls,), dtype=torch.float32
        )
        
        # Setting values for new walls
        new_walls[:, vertical_walls] = vertical_wall_values
        new_walls[horizontal_walls, :] = horizontal_wall_values.unsqueeze(1)
        
        # Adjusting pathloss values accordingly
        new_walls_transmittance_loss = calculate_transmittance_loss(new_walls, sample.x_ant, sample.y_ant)
        
        # Updating the sample
        sample.input_img[1][sample.mask == 1] += new_walls[sample.mask == 1]
        sample.output_img[sample.mask == 1] += new_walls_transmittance_loss[sample.mask == 1]
        sample.input_img[-1][sample.input_img[-1] != 0] += new_walls_transmittance_loss[sample.input_img[-1] != 0]
        if sample.pl_clip != float("inf"):
            sample.pl_clip += new_walls_transmittance_loss.max()
        
        return sample
    
    def __call__(self, sample: RadarSample) -> RadarSample:
        if random.random() < self.walls_p:
            sample = self._apply_walls(sample)
        
        if random.random() > self.p:
            return sample
        
        if self.scale_range is not None:
            scale_factor = random.uniform(*self.scale_range)
            sample = self._apply_distance_scaling(sample, scale_factor)
        
        if self.cardinal_rotation:
            sample = self._apply_cardinal_rotation(sample)
        
        if self.angle_range is not None:
            angle = random.uniform(*self.angle_range)
            sample = self._apply_rotation(sample, angle)
        
        flip_h = self.flip_horizontal and (random.random() < 0.5)
        flip_v = self.flip_vertical and (random.random() < 0.5)
        if flip_h or flip_v:
            sample = self._apply_flipping(sample, flip_h, flip_v)
        
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
