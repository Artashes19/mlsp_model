import random
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from numba import njit

from src.utils.indoor.types import RadarSample
from src.utils.indoor.channel_config import SPARSE_CHANNEL


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


@njit
def _calculate_transmittance_loss_numpy(
    transmittance_matrix,
    x_ant,
    y_ant,
    n_angles=360 * 128,
    radial_step=1.0,
    max_walls=10,
):
    """
    Calculate pathloss from transmittance matrix by ray-tracing from antenna.

    Numba-optimized implementation that traces rays from the antenna position
    and accumulates transmittance loss when crossing walls (transitions from
    transmittance>0 to transmittance=0).

    Args:
        transmittance_matrix: 2D numpy array of transmittance values
        x_ant: Antenna x-coordinate (column)
        y_ant: Antenna y-coordinate (row)
        n_angles: Number of angular rays to trace (default: 360*128)
        radial_step: Step size for radial sampling (default: 1.0 pixel)
        max_walls: Maximum number of walls to traverse per ray (default: 10)

    Returns:
        2D numpy array of accumulated transmittance loss values
    """
    h, w = transmittance_matrix.shape
    dtheta = 2.0 * np.pi / n_angles
    output = np.zeros((h, w), dtype=transmittance_matrix.dtype)

    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.sqrt(w * w + h * h)

    for i in range(n_angles):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]

        sum_loss = 0.0
        last_val = None
        wall_count = 0
        r = 0.0

        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t

            px = int(round(x))
            py = int(round(y))

            if px < 0 or px >= w or py < 0 or py >= h:
                if last_val is not None and last_val > 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        pass  # Already out of bounds
                break

            val = transmittance_matrix[py, px]

            if last_val is None:
                last_val = val

            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        r_temp = r
                        while r_temp <= max_dist:
                            x_temp = x_ant + r_temp * cos_t
                            y_temp = y_ant + r_temp * sin_t
                            px_temp = int(round(x_temp))
                            py_temp = int(round(y_temp))

                            if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                                break

                            if output[py_temp, px_temp] == 0 or sum_loss < output[py_temp, px_temp]:
                                output[py_temp, px_temp] = sum_loss
                            r_temp += radial_step
                        break
                last_val = val

            if output[py, px] == 0 or (sum_loss < output[py, px]):
                output[py, px] = sum_loss

            r += radial_step

    return output


def calculate_transmittance_loss(
    transmittance_matrix: torch.Tensor,
    x_ant: float,
    y_ant: float,
    n_angles: int = 360 * 128,
    radial_step: float = 1.0,
    max_walls: int = 10,
) -> torch.Tensor:
    """
    Calculate pathloss from transmittance matrix (torch wrapper).

    Args:
        transmittance_matrix: 2D torch tensor of transmittance values
        x_ant: Antenna x-coordinate (column)
        y_ant: Antenna y-coordinate (row)
        n_angles: Number of angular rays to trace
        radial_step: Step size for radial sampling
        max_walls: Maximum number of walls to traverse per ray

    Returns:
        2D torch tensor of accumulated transmittance loss values
    """
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(
        transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls
    )
    return torch.from_numpy(output_np).to(device=torch.device("cpu"))


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


class WallInsertionAugmentation(BaseAugmentation):
    """
    Augmentation that inserts synthetic walls into the scene.

    This augmentation adds random vertical and horizontal walls to the transmittance
    channel, then recalculates the output pathloss using ray-tracing to account for
    the additional transmittance loss from the new walls.

    Only pixels where transmittance is positive but reflectance is zero are considered
    for wall insertion (air regions that can have walls added).
    """

    def __init__(
        self,
        p: float = 0.5,
        transmittance_range: tuple = (2, 18),
        max_wall_density: float = 0.3,
        n_angles: int = 360 * 128,
        radial_step: float = 1.0,
        max_walls: int = 10,
    ):
        """
        Args:
            p: Probability of applying wall insertion (0.0 to 1.0)
            transmittance_range: (min, max) transmittance values for inserted walls
            max_wall_density: Maximum fraction of pixels that can become walls
            n_angles: Number of angular rays for pathloss recalculation
            radial_step: Radial step size for ray tracing
            max_walls: Maximum number of walls per ray in loss calculation
        """
        self.p = p
        self.transmittance_range = transmittance_range
        self.max_wall_density = max_wall_density
        self.n_angles = n_angles
        self.radial_step = radial_step
        self.max_walls = max_walls

    def __call__(self, sample: RadarSample) -> RadarSample:
        """Apply wall insertion augmentation to the sample."""
        if random.random() > self.p:
            return sample

        return self._apply_walls(sample)

    def _apply_walls(self, sample: RadarSample) -> RadarSample:
        """
        Insert random walls into the scene and recalculate pathloss.

        Walls are added as vertical and horizontal lines with random transmittance
        values. The output pathloss is then recalculated to account for the new walls.
        """
        H, W = sample.H, sample.W
        new_walls = torch.zeros((H, W), dtype=torch.float32)

        # Get transmittance from masked valid pixels
        transmittance = sample.input_img[1]  # Channel 1 is transmittance
        valid_mask = sample.mask == 1

        # Calculate maximum number of walls based on density
        n_valid_pixels = valid_mask.sum().item()
        if n_valid_pixels == 0:
            return sample  # No valid pixels, skip augmentation

        # Estimate wall pixels: vertical_walls * H + horizontal_walls * W - intersections
        # Approximate: for uniform distribution, max_walls lines should cover ~max_density pixels
        # A vertical wall covers H pixels, horizontal covers W pixels
        # To limit total coverage, we use: n_vert * H + n_horiz * W <= max_density * H * W
        max_wall_pixels = int(n_valid_pixels * self.max_wall_density)

        # Conservative estimate: limit total lines rather than total pixels
        max_vertical_lines = min(W, max(1, int(max_wall_pixels / (2 * H))))
        max_horizontal_lines = min(H, max(1, int(max_wall_pixels / (2 * W))))

        if max_vertical_lines < 1 and max_horizontal_lines < 1:
            return sample  # Too few pixels for meaningful augmentation

        # Randomly choose number of vertical and horizontal walls
        num_vertical_walls = np.random.randint(0, max_vertical_lines + 1)
        num_horizontal_walls = np.random.randint(0, max_horizontal_lines + 1)

        if num_vertical_walls == 0 and num_horizontal_walls == 0:
            return sample  # No walls to add

        # Choose random positions for walls
        if num_vertical_walls > 0:
            vertical_walls = np.random.choice(W, num_vertical_walls, replace=False)
        else:
            vertical_walls = np.array([], dtype=np.int64)

        if num_horizontal_walls > 0:
            horizontal_walls = np.random.choice(H, num_horizontal_walls, replace=False)
        else:
            horizontal_walls = np.array([], dtype=np.int64)

        if num_vertical_walls > 0:
            vertical_wall_values = torch.randint(
                self.transmittance_range[0],
                self.transmittance_range[1],
                (num_vertical_walls,),
                dtype=torch.float32,
            )
            new_walls[:, vertical_walls] = vertical_wall_values

        if num_horizontal_walls > 0:
            horizontal_wall_values = torch.randint(
                self.transmittance_range[0],
                self.transmittance_range[1],
                (num_horizontal_walls,),
                dtype=torch.float32,
            )
            new_walls[horizontal_walls, :] = horizontal_wall_values.unsqueeze(1)

        # Apply mask to walls before loss calculation to keep outputs consistent
        new_walls = new_walls * valid_mask

        # Calculate the pathloss contribution from the new walls
        new_walls_transmittance_loss = calculate_transmittance_loss(
            new_walls,
            sample.x_ant,
            sample.y_ant,
            n_angles=self.n_angles,
            radial_step=self.radial_step,
            max_walls=self.max_walls,
        )

        # Update the sample: add walls to transmittance channel only where mask is valid
        sample.input_img[1][valid_mask] += new_walls[valid_mask]

        # Update output pathloss only where mask is valid
        if sample.output_img is not None and not isinstance(sample.output_img, str):
            sample.output_img[valid_mask] += new_walls_transmittance_loss[valid_mask]

        # Update sparse measurements to maintain consistency with new walls
        if SPARSE_CHANNEL is not None and SPARSE_CHANNEL < sample.input_img.shape[0]:
            sparse_channel = sample.input_img[SPARSE_CHANNEL]
            sparse_exists = sparse_channel != 0
            if sparse_exists.any():
                sample.input_img[SPARSE_CHANNEL][sparse_exists] += new_walls_transmittance_loss[sparse_exists]

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
