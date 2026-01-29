import random
from typing import Optional

import math
import numpy as np
import torch
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from src.utils.indoor.types import RadarSample
from src.utils.indoor.config_overrides import get_config
from src.utils.indoor.channel_config import (
    CHANNEL_ORDER,
    NUM_CHANNELS,
    INPUT_REFLECTANCE_CHANNEL,
    INPUT_TRANSMITTANCE_CHANNEL,
    INPUT_DISTANCE_CHANNEL,
)


def calculate_fspl(
    dist_m,  # distance in meters (torch tensor)
    freq_MHz,  # frequency in MHz
    antenna_gain,  # shape=(360,) antenna gain in dBi [0..359]
    min_dist_m=0.125,  # clamp distance below this
):
    dist_clamped = torch.clamp(dist_m, min=min_dist_m)
    freq_tensor = torch.tensor(freq_MHz, device=torch.device('cpu'))
    fspl_linear = 20.0 * torch.log10(dist_clamped) + 20.0 * torch.log10(freq_tensor) - 27.55
    pathloss_linear = fspl_linear - antenna_gain
    
    return pathloss_linear


def calculate_antenna_gain(radiation_pattern, W, H, azimuth, x_ant, y_ant):
    """
    Calculate antenna gain across a grid based on radiation pattern and antenna orientation.
    Works with torch tensors.
    """
    x_grid = torch.arange(W, device=torch.device('cpu')).expand(H, W)
    y_grid = torch.arange(H, device=torch.device('cpu')).view(-1, 1).expand(H, W)
    angles = -(180 / torch.pi) * torch.atan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + azimuth
    angles = torch.where(angles > 359, angles - 360, angles).to(torch.long)
    antenna_gain = radiation_pattern[angles]
    
    return antenna_gain


# Fourier frequency encoding parameters
FREQ_MIN = 100.0    # MHz
FREQ_MAX = 7000.0   # MHz
FREQ_N_LEVELS = 2   # Number of sin/cos pairs (4 channels total)
FREQ_N_CHANNELS = 2 * FREQ_N_LEVELS  # 4 channels


def encode_frequency_fourier(freq_mhz: float) -> tuple:
    """
    Fourier positional encoding for frequency in log-space.
    
    Args:
        freq_mhz: Frequency in MHz (clamped to [FREQ_MIN, FREQ_MAX])
    
    Returns:
        Tuple of FREQ_N_CHANNELS values in [-1, 1]
        - Even indices: sin at increasing scales (1, 2, 4, ...)
        - Odd indices: cos at increasing scales (1, 2, 4, ...)
    """
    freq_clamped = max(FREQ_MIN, min(FREQ_MAX, freq_mhz))
    
    # Normalize log-frequency to [0, 1]
    log_min, log_max = math.log(FREQ_MIN), math.log(FREQ_MAX)
    t = (math.log(freq_clamped) - log_min) / (log_max - log_min)
    
    activations = []
    for level in range(FREQ_N_LEVELS):
        scale = 2 ** level  # 1, 2, 4, ...
        activations.append(math.sin(2 * math.pi * scale * t))
        activations.append(math.cos(2 * math.pi * scale * t))
    
    return tuple(activations)


def get_num_channels(channels: str = None) -> int:
    """
    Return the number of channels from channel_config.
    The channels parameter is deprecated and ignored - use channel_config.CHANNEL_ORDER instead.
    """
    return NUM_CHANNELS


def normalize_input(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize input tensor based on channel types (standardize only).
    Uses CHANNEL_ORDER from channel_config to determine normalization per channel.
    
    - reflectance/transmittance/sparse: z-score on non-zero values, zeros remain 0
    - distance: log(d + eps) then z-score
    - freq_*: unchanged (Fourier encoding already in [-1, 1])
    - mask/floor_plan: unchanged (binary)
    - antenna_gain: unchanged
    """
    config = get_config()
    normalized = input_tensor.clone()
    stats = config.normalization_stats
    eps = 1e-6
    
    def _get_stat(key: str, field: str) -> float:
        if key not in stats or field not in stats[key]:
            raise ValueError(f"Missing normalization stat: {key}.{field}")
        return float(stats[key][field])
    
    r_mean = _get_stat("r", "mean_nz")
    r_std = _get_stat("r", "std_nz")
    t_mean = _get_stat("t", "mean_nz")
    t_std = _get_stat("t", "std_nz")
    d_log_mean = _get_stat("d", "log_mean")
    d_log_std = _get_stat("d", "log_std")
    s_mean = _get_stat("s", "mean_nz")
    s_std = _get_stat("s", "std_nz")
    
    if r_std == 0 or t_std == 0 or d_log_std == 0 or s_std == 0:
        raise ValueError("Normalization std must be non-zero.")
    
    for idx, ch_name in enumerate(CHANNEL_ORDER):
        if ch_name == "reflectance":
            channel = normalized[idx]
            mask = channel != 0
            if mask.any():
                channel = channel.clone()
                channel[mask] = (channel[mask] - r_mean) / r_std
            normalized[idx] = channel
        elif ch_name == "transmittance":
            channel = normalized[idx]
            mask = channel != 0
            if mask.any():
                channel = channel.clone()
                channel[mask] = (channel[mask] - t_mean) / t_std
            normalized[idx] = channel
        elif ch_name == "distance":
            channel = torch.log(normalized[idx] + eps)
            normalized[idx] = (channel - d_log_mean) / d_log_std
        elif ch_name == "sparse":
            channel = normalized[idx]
            mask = channel != 0
            if mask.any():
                channel = channel.clone()
                channel[mask] = (channel[mask] - s_mean) / s_std
            normalized[idx] = channel
        # antenna_gain, freq_*, mask, floor_plan: no normalization needed
    
    return normalized


# Sampling-based masking removed; model operates on full data


def kriging(
    pl_init, aux_sample,
    length_scale=20.0, length_scale_bounds=(1.0, 100.0),
    constant_value=1.0, constant_value_bounds=(1e-3, 1e3)
):
    h, w = pl_init.shape
    mask = aux_sample != 0
    sample_indices = np.argwhere(mask)
    x_train = sample_indices.T
    y_train = aux_sample[mask] - pl_init[mask]
    
    # Define the Gaussian Process model
    kernel = C(constant_value, constant_value_bounds) * RBF(length_scale, length_scale_bounds)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1.0)
    
    # Fit the GP model on residuals
    gp.fit(x_train, y_train)
    
    # Predict residuals for all pixels
    xx, yy = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    x_pred = np.column_stack([xx.ravel(), yy.ravel()])
    y_pred, sigma = gp.predict(x_pred, return_std=True)
    
    # Create the adjusted pathloss map
    residual_map = y_pred.reshape((h, w))
    adjusted_map = pl_init + residual_map
    return adjusted_map


def interpolate_difference(pl_init: np.ndarray, aux_sample: np.ndarray, method="linear") -> np.ndarray:
    """
    Interpolates the difference between pl_init and an auxiliary sample using linear interpolation.

    Parameters:
        pl_init (np.ndarray): Initial pathloss estimate (H x W).
        aux_sample (np.ndarray): Ground truth measurements (H x W), with 0 indicating missing data.
        method (str): Interpolation method, one of 'linear', 'nearest', or 'cubic'.

    Returns:
        np.ndarray: Updated pathloss map after interpolation-based correction.
    """
    # Identify valid measurement positions
    mask = aux_sample > 0
    
    coords = np.column_stack(np.nonzero(mask)).T  # (N, 2) -> (row, col)
    diff_values = (aux_sample - pl_init)[mask]
    
    # Generate full grid
    h, w = pl_init.shape
    grid_x, grid_y = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Interpolate
    interpolated_diff = griddata(coords, diff_values, grid_points, method=method, fill_value=0)
    interpolated_diff = interpolated_diff.reshape(pl_init.shape)
    
    # Apply correction
    corrected_map = pl_init + interpolated_diff
    return corrected_map


def extrapolate_difference(pl_init: np.ndarray, aux_sample: np.ndarray, neighbors=10) -> np.ndarray:
    mask = (aux_sample > 0) & np.isfinite(aux_sample)
    coords = np.array(np.nonzero(mask))
    diff_values = (aux_sample - pl_init)[mask]
    
    H, W = pl_init.shape
    grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    
    rbf = RBFInterpolator(coords, diff_values, neighbors=neighbors, smoothing=0.1)
    extrapolated_diff = rbf(grid_points).reshape(H, W)
    
    return pl_init + extrapolated_diff


def get_fspl(sample: RadarSample) -> torch.Tensor:
    radiation_pattern = sample.radiation_pattern
    antenna_gain = calculate_antenna_gain(
        radiation_pattern,
        sample.W,
        sample.H,
        sample.azimuth,
        sample.x_ant,
        sample.y_ant
    )
    fspl = calculate_fspl(
        dist_m=sample.input_img[INPUT_DISTANCE_CHANNEL],
        freq_MHz=sample.freq_MHz,
        antenna_gain=antenna_gain
    )
    return fspl


def featurizer(
    sample: RadarSample,
    sparse_range: tuple[float, float],
    modality_dropout_prob: float,
    sparse_dropout_given_dropout: float,
    force_drop_sparse: Optional[bool],
    force_drop_trans_ref: Optional[bool],
    precomputed_sparse: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Build input tensor with channels defined in channel_config.CHANNEL_ORDER.
    
    Channel letters:
        r - reflectance
        t - transmittance
        d - distance
        g - antenna gain
        f - frequency channels encoded in 2 sin/cos pairs using Fourier frequency encoding, 4 channels total
        m - mask
        p - floor plan
        s - sparse measurements
    
    Args:
        force_drop_sparse: If True, force sparse channel to be all zeros.
            If False, force sparse channel to be enabled. If None, use random dropout.
        force_drop_trans_ref: If True, force trans+ref channels to be all zeros.
            If False, force trans+ref channels to be enabled. If None, use random dropout.
        precomputed_sparse: Optional pre-loaded sparse measurements tensor (H, W).
            If provided, use this instead of generating random sparse samples.
    """
    # Modality dropout logic:
    # With prob modality_dropout_prob, turn off one modality (trans+ref OR sparse)
    # If dropout happens, sparse_dropout_given_dropout controls which one is turned off
    drop_trans_ref = False
    drop_sparse = False
    
    # Apply force flags if set, otherwise use random dropout
    if force_drop_sparse is not None:
        drop_sparse = force_drop_sparse
    if force_drop_trans_ref is not None:
        drop_trans_ref = force_drop_trans_ref
    
    # Only apply random dropout if neither force flag is set
    if force_drop_sparse is None and force_drop_trans_ref is None:
        if random.random() < modality_dropout_prob:
            # Dropout one modality
            if random.random() < sparse_dropout_given_dropout:
                drop_sparse = True
            else:
                drop_trans_ref = True
    
    # Precompute base data from sample
    reflectance = sample.input_img[INPUT_REFLECTANCE_CHANNEL]
    transmittance = sample.input_img[INPUT_TRANSMITTANCE_CHANNEL]
    distance = sample.input_img[INPUT_DISTANCE_CHANNEL]
    
    # Compute antenna gain only if needed
    antenna_gain = None
    if "antenna_gain" in CHANNEL_ORDER:
        radiation_pattern = sample.radiation_pattern
        antenna_gain = calculate_antenna_gain(
            radiation_pattern,
            sample.W,
            sample.H,
            sample.azimuth,
            sample.x_ant,
            sample.y_ant
        )
    
    # Compute sparse measurements only if needed and not dropped
    sparse_data = None
    if "sparse" in CHANNEL_ORDER and not drop_sparse:
        if precomputed_sparse is not None:
            # Use pre-loaded sparse measurements (e.g., from test data)
            sparse_data = precomputed_sparse
            # Update mask: set mask to 0 where sparse measurements exist
            sparse_mask = sparse_data != 0
            sample.mask[sparse_mask] = 0
        elif sample.output_img is not None:
            # Generate random sparse samples from ground truth
            sparsity = random.uniform(sparse_range[0], sparse_range[1])
            # Only sample from valid mask region
            valid_indices = torch.nonzero(sample.mask)
            if valid_indices.numel() > 0:
                num_samples = int(valid_indices.size(0) * sparsity)
                if num_samples > 0:
                    perm = torch.randperm(valid_indices.size(0))
                    selected_indices = valid_indices[perm[:num_samples]]
                    
                    # Get ground truth values (handling potential shape mismatch if output_img is (C,H,W))
                    output_img = sample.output_img
                    if output_img.ndim == 3:
                        output_img = output_img.squeeze(0)
                    
                    sparse_data = torch.zeros((sample.H, sample.W), dtype=torch.float32)
                    rows = selected_indices[:, 0]
                    cols = selected_indices[:, 1]
                    sparse_data[rows, cols] = output_img[rows, cols]
                    sample.mask[rows, cols] = 0
    
    # Precompute Fourier frequency encoding if any freq channels are present
    fourier_activations = None
    freq_channels = ["freq_sin_1", "freq_cos_1", "freq_sin_2", "freq_cos_2"]
    if any(ch in CHANNEL_ORDER for ch in freq_channels):
        fourier_activations = encode_frequency_fourier(sample.freq_MHz)
    
    # Build input tensor based on CHANNEL_ORDER
    input_tensor = torch.zeros((NUM_CHANNELS, sample.H, sample.W), dtype=torch.float32, device=torch.device("cpu"))
    
    for idx, ch_name in enumerate(CHANNEL_ORDER):
        if ch_name == "reflectance":
            if drop_trans_ref:
                input_tensor[idx] = torch.zeros_like(reflectance)
            else:
                input_tensor[idx] = reflectance
        elif ch_name == "transmittance":
            if drop_trans_ref:
                input_tensor[idx] = torch.zeros_like(transmittance)
            else:
                input_tensor[idx] = transmittance
        elif ch_name == "distance":
            input_tensor[idx] = distance
        elif ch_name == "antenna_gain":
            input_tensor[idx] = antenna_gain
        elif ch_name == "freq_sin_1":
            input_tensor[idx] = torch.full((sample.H, sample.W), fourier_activations[0], dtype=torch.float32)
        elif ch_name == "freq_cos_1":
            input_tensor[idx] = torch.full((sample.H, sample.W), fourier_activations[1], dtype=torch.float32)
        elif ch_name == "freq_sin_2":
            input_tensor[idx] = torch.full((sample.H, sample.W), fourier_activations[2], dtype=torch.float32)
        elif ch_name == "freq_cos_2":
            input_tensor[idx] = torch.full((sample.H, sample.W), fourier_activations[3], dtype=torch.float32)
        elif ch_name == "mask":
            input_tensor[idx] = sample.mask
        elif ch_name == "floor_plan":
            if sample.floor_plan is not None:
                input_tensor[idx] = sample.floor_plan
            else:
                input_tensor[idx] = ((reflectance > 0) | (transmittance > 0)).float()
        elif ch_name == "sparse":
            if sparse_data is not None:
                input_tensor[idx] = sparse_data
    
    input_tensor = normalize_input(input_tensor)
    
    return input_tensor
