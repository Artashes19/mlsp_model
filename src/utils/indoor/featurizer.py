import random

import math
import numpy as np
import torch
from numba import njit, prange
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from src.utils.indoor.types import RadarSample
from src.utils.indoor.config_overrides import get_config
from src.utils.indoor.channel_config import CHANNEL_ORDER, NUM_CHANNELS


@njit
def _calculate_transmittance_loss_numpy(
    transmittance_matrix, x_ant, y_ant, n_angles=360 * 128 / 1, radial_step=1.0, max_walls=10
):
    """
    Numpy implementation for numba optimization.
    This function must stay as numpy for numba to work.
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
                        pass  # Already out of bounds, so we do nothing more
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
    transmittance_matrix, x_ant, y_ant, n_angles=360 * 128 / 1, radial_step=1.0, max_walls=10
):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


@njit
def _calculate_hybrid_loss_numpy(
    material_matrix, x_ant, y_ant, n_angles=360 * 128, radial_step=0.5, max_walls=5, reflection_prob=0.5
):
    h, w = material_matrix.shape
    output = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    dtheta = 2.0 * np.pi / n_angles
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.sqrt(w * w + h * h)
    
    random_sequence = np.zeros(n_angles * int(max_dist // radial_step) * max_walls, dtype=np.float32)
    for i in range(len(random_sequence)):
        random_sequence[i] = (i * 1103515245 + 12345) % 2 ** 31 / 2 ** 31
    
    for i in range(n_angles):
        dir_x = cos_vals[i]
        dir_y = sin_vals[i]
        
        x = float(x_ant)
        y = float(y_ant)
        sum_loss = 0.0
        wall_count = 0
        r = 0.0
        
        last_val = None
        random_idx = i * int(max_dist // radial_step) * max_walls
        
        while r <= max_dist and wall_count < max_walls:
            x += dir_x * radial_step
            y += dir_y * radial_step
            r += radial_step
            
            px = int(round(x))
            py = int(round(y))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            val = material_matrix[py, px]
            if last_val is None:
                last_val = val
            
            n = counts[py, px]
            output[py, px] = (output[py, px] * n + sum_loss) / (n + 1)
            counts[py, px] = n + 1
            
            if val != last_val:
                if val > 0 or last_val > 0:
                    material_val = max(val, last_val)
                    sum_loss += material_val
                    wall_count += 1
                    
                    random_val = random_sequence[random_idx]
                    random_idx += 1
                    
                    if random_val < reflection_prob:
                        normal_x, normal_y = 0.0, 0.0
                        
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                if material_matrix[ny, nx] != material_matrix[py, px]:
                                    normal_x += float(dx)
                                    normal_y += float(dy)
                        
                        norm = np.sqrt(normal_x ** 2 + normal_y ** 2)
                        if norm > 0:
                            normal_x /= norm
                            normal_y /= norm
                        else:
                            normal_x, normal_y = -dir_x, -dir_y
                        
                        dot = dir_x * normal_x + dir_y * normal_y
                        dir_x = dir_x - 2.0 * dot * normal_x
                        dir_y = dir_y - 2.0 * dot * normal_y
                        
                        norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
                        if norm > 0:
                            dir_x /= norm
                            dir_y /= norm
                        
                        x += dir_x * radial_step * 0.1
                        y += dir_y * radial_step * 0.1
                        
                        new_px, new_py = int(round(x)), int(round(y))
                        if new_px < 0 or new_px >= w or new_py < 0 or new_py >= h:
                            break
            
            last_val = val
    
    return output


@njit(parallel=True, fastmath=True)
def _calculate_hybrid_loss_mc_numpy_fast(
    reflectance_matrix: np.ndarray,
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    n_angles: int = 360 * 64,
    radial_step: float = 1.0,
    max_reflect: int = 5,
    max_transmit: int = 10,
    reflection_prob: float = 0.5,
    samples_per_angle: int = 8,
    max_loss: float = 160.0
) -> np.ndarray:
    h, w = reflectance_matrix.shape
    # initialize to max_loss
    output = np.full((h, w), max_loss, dtype=np.float32)
    
    # precompute dirs
    two_pi = 2.0 * np.pi
    dtheta = two_pi / n_angles
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.hypot(w, h)
    
    # precompute RNG array once
    rng_count = n_angles * samples_per_angle * (max_reflect + max_transmit)
    rng = (np.arange(rng_count, dtype=np.int64) * 1103515245 + 12345) & 0x7FFFFFFF
    rng = (rng / np.float32(2 ** 31))
    
    # Parallel over angles
    for i in prange(n_angles):
        dx0 = cos_vals[i]
        dy0 = sin_vals[i]
        base_rng_i = i * samples_per_angle * (max_reflect + max_transmit)
        
        for s in range(samples_per_angle):
            # ray state
            x = x_ant
            y = y_ant
            dx = dx0
            dy = dy0
            sum_loss = 0.0
            last_val = -1.0  # sentinel
            refl_ct = 0
            trans_ct = 0
            rng_idx = base_rng_i + s * (max_reflect + max_transmit)
            
            # march until both caps or loss cap hit
            while True:
                # trace along this direction to next boundary
                traveled = 0.0
                hit_px = hit_py = -1
                
                while traveled <= max_dist:
                    x += dx * radial_step
                    y += dy * radial_step
                    traveled += radial_step
                    
                    px = int(round(x))
                    py = int(round(y))
                    if px < 0 or px >= w or py < 0 or py >= h:
                        # out of image
                        traveled = max_dist + 1.0
                        break
                    
                    # update best (min) loss so far
                    if sum_loss < output[py, px]:
                        output[py, px] = sum_loss
                    
                    # check for boundary
                    val = reflectance_matrix[py, px]
                    if last_val < 0.0:
                        last_val = val
                    if val != last_val:
                        hit_px, hit_py = px, py
                        break
                    last_val = val
                
                if hit_px < 0:
                    break  # left image
                
                # decide branch
                if refl_ct >= max_reflect and trans_ct >= max_transmit:
                    break
                elif refl_ct >= max_reflect:
                    branch = 0
                elif trans_ct >= max_transmit:
                    branch = 1
                else:
                    branch = 1 if rng[rng_idx] < reflection_prob else 0
                    rng_idx += 1
                
                # apply loss + update
                if branch == 1:
                    # reflect
                    sum_loss += reflectance_matrix[hit_py, hit_px]
                    refl_ct += 1
                    # estimate normal (4-nbr) and reflect dir
                    nx = 0.0;
                    ny = 0.0
                    # inline neighbor checks
                    if hit_px > 0 and reflectance_matrix[hit_py, hit_px - 1] != val:
                        nx -= 1.0
                    if hit_px < w - 1 and reflectance_matrix[hit_py, hit_px + 1] != val:
                        nx += 1.0
                    if hit_py > 0 and reflectance_matrix[hit_py - 1, hit_px] != val:
                        ny -= 1.0
                    if hit_py < h - 1 and reflectance_matrix[hit_py + 1, hit_px] != val:
                        ny += 1.0
                    norm = np.hypot(nx, ny)
                    if norm > 0.0:
                        nx /= norm;
                        ny /= norm
                    else:
                        nx, ny = -dx, -dy
                    # reflect vector
                    dot = dx * nx + dy * ny
                    dx -= 2.0 * dot * nx
                    dy -= 2.0 * dot * ny
                    mag = np.hypot(dx, dy)
                    if mag > 0.0:
                        dx /= mag;
                        dy /= mag
                else:
                    # transmit
                    sum_loss += transmittance_matrix[hit_py, hit_px]
                    trans_ct += 1
                    # direction unchanged
                
                # cap-check
                if sum_loss >= max_loss:
                    break
                
                # continue from this boundary
                x = hit_px
                y = hit_py
    
    return output


def calculate_hybrid_loss(
    reflectance_matrix, transmittance_matrix, x_ant, y_ant, n_angles=360 * 128, radial_step=1.0, max_walls=10
):
    reflectance_np = reflectance_matrix.cpu().numpy()
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_hybrid_loss_mc_numpy_fast(reflectance_np, transmittance_np, x_ant, y_ant)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def _calculate_reflectance_eff_numpy(
    reflectance_matrix: np.ndarray,
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    n_angles: int = 360 * 128,
    radial_step: float = 1.0,
    max_walls: int = 5,
    reflection_prob: float = 0.5
) -> np.ndarray:
    """
    Single‐ray effective attenuation: each wall interface
    contributes an expected loss a_eff = p_refl*R + (1-p_refl)*T.
    March one ray per angle, accumulate these a_eff losses,
    and record the per-pixel minimum accumulated loss.
    """
    h, w = reflectance_matrix.shape
    # initialize to a large value
    output = np.full((h, w), np.inf, dtype=np.float32)
    
    two_pi = 2.0 * np.pi
    dtheta = two_pi / n_angles
    cosv = np.cos(np.arange(n_angles) * dtheta)
    sinv = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.hypot(w, h)
    
    for i in prange(n_angles):
        dx = cosv[i]
        dy = sinv[i]
        sum_loss = 0.0
        last_val = -1.0  # sentinel: not yet on material
        r = 0.0
        # march until edge
        while r <= max_dist:
            x = x_ant + r * dx
            y = y_ant + r * dy
            px = int(x + 0.5)
            py = int(y + 0.5)
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            val = reflectance_matrix[py, px]
            # on interface if crossing from material->air or air->material
            if last_val >= 0.0 and val != last_val:
                # expected attenuation at this interface
                R = reflectance_matrix[py, px]
                T = transmittance_matrix[py, px]
                a_eff = reflection_prob * R + (1.0 - reflection_prob) * T
                sum_loss += a_eff
                # count walls and stop after max_walls
                last_val = val
                # record this loss at the interface pixel
                if sum_loss < output[py, px]:
                    output[py, px] = sum_loss
                # optionally stop is reached walls cap
                # but effective doesn't branch so continue
            last_val = val
            # record cumulative loss at every pixel
            if sum_loss < output[py, px]:
                output[py, px] = sum_loss
            r += radial_step
    
    return output


def calculate_reflectance_eff(
    reflectance_matrix: torch.Tensor,
    transmittance_matrix: torch.Tensor,
    x_ant: float,
    y_ant: float,
) -> np.ndarray:
    reflectance_np = reflectance_matrix.cpu().numpy()
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_reflectance_eff_numpy(
        reflectance_np,
        transmittance_np,
        x_ant,
        y_ant
    )
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


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


def update_rectangles(transmittance_loss, aux_sample):
    # Get unique rectangle values
    unique_values = torch.unique(transmittance_loss.flatten())
    updated_loss = transmittance_loss.clone()
    
    # Update each rectangle separately
    for val in unique_values:
        # Create a mask for the current rectangle
        mask = (transmittance_loss == val)
        
        # Get auxiliary values within this rectangle
        aux_values = aux_sample[mask]
        aux_values = aux_values[aux_values != 0]
        
        # Update rectangle if there are any auxiliary values
        if aux_values.numel() > 0:
            # Use the mean of auxiliary values as the new value
            updated_loss[mask] = aux_values.mean()
    
    return updated_loss


def calculate_pl_init(
    sample: RadarSample,
    distance,
    antenna_gain,
    transmittance,
    aux_sample=None,
):
    # Calculate free space path loss on CPU
    free_space_pathloss = calculate_fspl(
        dist_m=distance,
        freq_MHz=sample.freq_MHz,
        antenna_gain=antenna_gain,
    )
    
    # Calculate transmittance loss on CPU
    transmittance_loss = calculate_transmittance_loss(
        transmittance,
        sample.x_ant,
        sample.y_ant
    )
    
    # auxiliary based update disabled
    
    pl_init = free_space_pathloss + transmittance_loss
    
    return pl_init


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


@njit
def select_indices(cand_idx, W, num_points, min_sep):
    sel = np.empty((num_points, 2), np.int64)
    count = 0
    min_sep_sq = min_sep * min_sep
    for idx in cand_idx:
        if count >= num_points:
            break
        r = idx // W
        c = idx - r * W
        ok = True
        for j in range(count):
            dr = r - sel[j, 0]
            dc = c - sel[j, 1]
            if dr * dr + dc * dc < min_sep_sq:
                ok = False
                break
        if ok:
            sel[count, 0] = r
            sel[count, 1] = c
            count += 1
    return sel, count


def add_points(matrix, x_ant, y_ant, num_points, alpha=2.0, min_sep=None, oversample=10):
    H, W = matrix.shape
    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]
    dist = np.hypot(ys - y_ant, xs - x_ant).flatten()
    probs = dist ** alpha
    total = probs.sum()
    if total == 0:
        probs[:] = 1
        total = H * W
    probs /= total
    cand = np.random.choice(H * W, num_points * oversample, True, p=probs)
    cand = np.unique(cand)
    np.random.shuffle(cand)
    if min_sep is None:
        min_sep = 0.5 * math.sqrt(H * W / num_points)
    sel_arr, cnt = select_indices(cand.astype(np.int64), W, num_points, min_sep)
    selected = [(int(sel_arr[i, 0]), int(sel_arr[i, 1])) for i in range(cnt)]
    if cnt < num_points:
        rem = np.setdiff1d(np.arange(H * W), [r * W + c for r, c in selected], assume_unique=False)
        fill = np.random.choice(rem, num_points - cnt, False, p=probs[rem] / probs[rem].sum())
        for idx in fill:
            selected.append(divmod(int(idx), W))
    for r, c in selected:
        matrix[r, c] = 1
    return selected


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
        dist_m=sample.input_img[2],
        freq_MHz=sample.freq_MHz,
        antenna_gain=antenna_gain
    )
    return fspl


def featurizer(
    sample: RadarSample,
    sparse_range: tuple[float, float] = (0.0, 0.01),
    modality_dropout_prob: float = 0.6666,
    sparse_dropout_given_dropout: float = 0.5,
) -> torch.Tensor:
    """
    Build input tensor with channels defined in channel_config.CHANNEL_ORDER.
    
    Channel order and count are configured in src/utils/indoor/channel_config.py.
    """
    # Modality dropout logic:
    # With prob modality_dropout_prob, turn off one modality (trans+ref OR sparse)
    # If dropout happens, sparse_dropout_given_dropout controls which one is turned off
    drop_trans_ref = False
    drop_sparse = False
    
    if random.random() < modality_dropout_prob:
        # Dropout one modality
        if random.random() < sparse_dropout_given_dropout:
            drop_sparse = True
        else:
            drop_trans_ref = True
    
    # Precompute base data from sample
    reflectance = sample.input_img[0]  # First channel
    transmittance = sample.input_img[1]  # Second channel
    distance = sample.input_img[2]  # Third channel
    
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
    if "sparse" in CHANNEL_ORDER and not drop_sparse and sample.output_img is not None:
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


_calculate_transmittance_loss_numpy(np.array([[1]]), 0, 0)
_calculate_hybrid_loss_mc_numpy_fast(np.array([[1]]), np.array([[1]]), 0, 0)
_calculate_reflectance_eff_numpy(np.array([[1]]), np.array([[1]]), 0, 0)
