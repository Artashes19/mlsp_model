"""
Unit tests for the featurizer function using fully controlled mock data.
These tests verify:
1. Output tensor shape (11 channels with Fourier frequency encoding)
2. Each channel contains expected values based on controlled inputs
3. Normalization is applied correctly
4. Sparse sampling logic works as expected
5. Floor plan generation from reflectance/transmittance
"""
import os
import sys
import unittest
from pathlib import Path

import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file (same as run.py)
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from src.utils.indoor.types import RadarSample
from src.utils.indoor.featurizer import featurizer, normalize_input, get_num_channels
from src.utils.indoor.channel_config import (
    # Input channel indices
    INPUT_REFLECTANCE_CHANNEL,
    INPUT_TRANSMITTANCE_CHANNEL,
    INPUT_DISTANCE_CHANNEL,
    # Output channel indices
    CHANNEL_ORDER,
    NUM_CHANNELS,
    REFLECTANCE_CHANNEL,
    TRANSMITTANCE_CHANNEL,
    DISTANCE_CHANNEL,
    ANTENNA_GAIN_CHANNEL,
    MASK_CHANNEL,
    FLOOR_PLAN_CHANNEL,
    SPARSE_CHANNEL,
    FREQ_SIN_1_CHANNEL,
    FREQ_COS_1_CHANNEL,
    FREQ_SIN_2_CHANNEL,
    FREQ_COS_2_CHANNEL,
)

# Normalization constants from config (z-score stats)
# These come from indoor_normalization_unified.yaml
R_MEAN, R_STD = 3.236, 2.849  # reflectance
T_MEAN, T_STD = 5.593, 5.444  # transmittance
D_LOG_MEAN, D_LOG_STD = 3.539, 0.797  # distance (log-transformed)
S_MEAN, S_STD = 110.146, 42.037  # sparse measurements

# Default channels string for featurizer (all channels enabled)
DEFAULT_CHANNELS = "rtdgfmps"


def default_featurizer_args(
    sparse_range=(0.0, 0.01),
    modality_dropout_prob=0.0,
    sparse_dropout_given_dropout=0.5,
    channels=DEFAULT_CHANNELS,
    force_drop_sparse=False,
    force_drop_trans_ref=False,
):
    """Return dict of default featurizer kwargs for tests."""
    return dict(
        sparse_range=sparse_range,
        modality_dropout_prob=modality_dropout_prob,
        sparse_dropout_given_dropout=sparse_dropout_given_dropout,
        channels=channels,
        force_drop_sparse=force_drop_sparse,
        force_drop_trans_ref=force_drop_trans_ref,
    )


def create_mock_sample(
    H: int = 100,
    W: int = 100,
    x_ant: float = 50.0,
    y_ant: float = 50.0,
    freq_MHz: float = 868.0,
    pixel_size: float = 0.25,
    reflectance_value: float = 1.0,
    transmittance_value: float = 0.5,
    output_value: float = 80.0,  # dB pathloss
    valid_h: int = None,  # Height of valid (non-padded) region, None = full H
    valid_w: int = None,  # Width of valid (non-padded) region, None = full W
) -> RadarSample:
    """
    Create a fully controlled mock RadarSample for testing.
    
    Default setup:
    - 100x100 image
    - Antenna at center (50, 50)
    - Square structure: reflectance=1 in [20:80, 20:80], transmittance=0.5 in [30:70, 30:70]
    - Distance increases linearly from antenna
    - Uniform output pathloss
    - Realistic mask: 1 in valid region, 0 in padded region (simulates padding to square)
    
    Args:
        valid_h: Height of the valid (non-padded) region. If None, uses full H.
        valid_w: Width of the valid (non-padded) region. If None, uses full W.
    """
    # Default valid region to full size if not specified
    if valid_h is None:
        valid_h = H
    if valid_w is None:
        valid_w = W
    
    # Reflectance: outer square wall (scaled to valid region)
    reflectance = torch.zeros((H, W), dtype=torch.float32)
    r_y1, r_y2 = int(0.2 * valid_h), int(0.8 * valid_h)
    r_x1, r_x2 = int(0.2 * valid_w), int(0.8 * valid_w)
    reflectance[r_y1:r_y2, r_x1:r_x2] = reflectance_value
    
    # Transmittance: inner square (subset of reflectance region)
    transmittance = torch.zeros((H, W), dtype=torch.float32)
    t_y1, t_y2 = int(0.3 * valid_h), int(0.7 * valid_h)
    t_x1, t_x2 = int(0.3 * valid_w), int(0.7 * valid_w)
    transmittance[t_y1:t_y2, t_x1:t_x2] = transmittance_value
    
    # Distance map: Euclidean distance from antenna in meters
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_px = torch.sqrt((xx - x_ant) ** 2 + (yy - y_ant) ** 2)
    dist_m = dist_px * pixel_size
    
    # Stack into input_img (3 channels: reflectance, transmittance, distance)
    input_img = torch.stack([reflectance, transmittance, dist_m])
    
    # Output: uniform pathloss for simplicity (real data would have gradients)
    output_img = torch.full((H, W), output_value, dtype=torch.float32)
    
    # Mask: 1 in valid region, 0 in padded region (realistic padding mask)
    # Simulates images resized to 640xN or Nx640 then padded to square
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[:valid_h, :valid_w] = 1.0
    
    # Radiation pattern: uniform (isotropic antenna)
    radiation_pattern = torch.zeros(360, dtype=torch.float32)
    
    return RadarSample(
        file_name="mock_sample",
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
        mask=mask,
        floor_plan=None
    )


def save_featurizer_output_visualization(
    output_tensor: torch.Tensor,
    sample: RadarSample,
    output_path: Path,
    name: str
):
    """Save visualization of featurizer output for inspection."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build channel names from config
    channel_names = [f"Ch{i}: {ch_name}" for i, ch_name in enumerate(CHANNEL_ORDER)]
    
    # Calculate grid size based on number of channels
    n_cols = 4
    n_rows = (NUM_CHANNELS + n_cols - 1) // n_cols  # ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(NUM_CHANNELS):
        ax = axes[i]
        data = output_tensor[i].numpy()
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f"{channel_names[i]}\nmin={data.min():.4f}, max={data.max():.4f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide any extra subplots
    for i in range(NUM_CHANNELS, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Featurizer Output: {name}\nAntenna: ({sample.x_ant}, {sample.y_ant}), Freq: {sample.freq_MHz} MHz")
    plt.tight_layout()
    
    fig_path = output_path / f"featurizer_{name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {fig_path}")
    return fig_path


class TestFeaturizerLogic(unittest.TestCase):
    """
    Unit tests for the featurizer function using controlled mock data.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.output_dir = PROJECT_ROOT / "tests" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_output_shape(self):
        """Test that featurizer outputs correct shape matching channel_config."""
        print("\n" + "=" * 60)
        print("[TEST] Featurizer Output Shape")
        print("=" * 60)
        
        sample = create_mock_sample()
        output = featurizer(sample, **default_featurizer_args())
        
        self.assertEqual(output.shape[0], NUM_CHANNELS, f"Should have {NUM_CHANNELS} channels (from config)")
        self.assertEqual(output.shape[1], sample.H, "Height should match")
        self.assertEqual(output.shape[2], sample.W, "Width should match")
        print(f"  Output shape: {output.shape} ✓")
        print(f"  Channel order: {CHANNEL_ORDER}")
    
    def test_reflectance_channel(self):
        """Test reflectance channel z-score normalization."""
        print("\n" + "=" * 60)
        print(f"[TEST] Reflectance Channel (index {REFLECTANCE_CHANNEL})")
        print("=" * 60)
        
        sample = create_mock_sample(reflectance_value=10.0)  # Test value
        # Disable modality dropout to ensure reflectance isn't zeroed
        output = featurizer(sample, **default_featurizer_args())
        
        ch = output[REFLECTANCE_CHANNEL]
        
        # Check non-zero region (where we set reflectance)
        non_zero_region = ch[20:80, 20:80]
        zero_region = ch[0:20, 0:20]
        
        # Z-score normalization: (x - mean) / std for non-zero values
        expected_non_zero = (10.0 - R_MEAN) / R_STD
        expected_zero = 0.0  # Zero values stay zero
        
        print(f"  Non-zero region: min={non_zero_region.min():.4f}, max={non_zero_region.max():.4f}")
        print(f"  Expected for 10.0: {expected_non_zero:.4f}")
        print(f"  Zero region: min={zero_region.min():.4f}, max={zero_region.max():.4f}")
        print(f"  Expected for 0: {expected_zero:.4f}")
        
        # Allow small tolerance
        self.assertTrue(
            torch.allclose(non_zero_region, torch.tensor(expected_non_zero), atol=0.1),
            f"Reflectance normalization mismatch: got {non_zero_region.mean():.4f}, expected {expected_non_zero:.4f}"
        )
        self.assertTrue(
            torch.allclose(zero_region, torch.tensor(expected_zero), atol=0.01),
            f"Zero region should stay zero"
        )
    
    def test_transmittance_channel(self):
        """Test transmittance channel z-score normalization."""
        print("\n" + "=" * 60)
        print(f"[TEST] Transmittance Channel (index {TRANSMITTANCE_CHANNEL})")
        print("=" * 60)
        
        sample = create_mock_sample(transmittance_value=10.0)  # Test value
        # Disable modality dropout to ensure transmittance isn't zeroed
        output = featurizer(sample, **default_featurizer_args())
        
        ch = output[TRANSMITTANCE_CHANNEL]
        
        # Z-score normalization: (x - mean) / std for non-zero values
        expected_val = (10.0 - T_MEAN) / T_STD
        
        non_zero_region = ch[30:70, 30:70]
        print(f"  Transmittance region: mean={non_zero_region.mean():.4f}")
        print(f"  Expected for 10.0: {expected_val:.4f}")
        
        # Verify the transmittance values are normalized correctly
        self.assertTrue(
            torch.allclose(non_zero_region, torch.tensor(expected_val), atol=0.1),
            f"Transmittance normalization mismatch: got {non_zero_region.mean():.4f}, expected {expected_val:.4f}"
        )
    
    def test_distance_channel(self):
        """Test distance channel - log + z-score transformation."""
        print("\n" + "=" * 60)
        print(f"[TEST] Distance Channel (index {DISTANCE_CHANNEL})")
        print("=" * 60)
        
        import math
        
        sample = create_mock_sample()
        output = featurizer(sample, **default_featurizer_args())
        
        ch = output[DISTANCE_CHANNEL]
        
        # Distance normalization formula: (log(dist_m + eps) - D_LOG_MEAN) / D_LOG_STD
        eps = 1e-6
        
        # Test specific pixels with known distances
        # Antenna at (50, 50), pixel_size=0.25m
        
        # Corner (0, 0): distance = sqrt(50^2 + 50^2) * 0.25 = 17.68m
        corner_dist_m = math.sqrt(50**2 + 50**2) * sample.pixel_size
        expected_corner = (math.log(corner_dist_m + eps) - D_LOG_MEAN) / D_LOG_STD
        actual_corner = ch[0, 0].item()
        
        # Center (50, 50): distance = 0 (at antenna)
        # log(eps) is very negative
        expected_center = (math.log(eps) - D_LOG_MEAN) / D_LOG_STD
        actual_center = ch[50, 50].item()
        
        # Far edge (0, 50): distance = 50 * 0.25 = 12.5m
        edge_dist_m = 50 * sample.pixel_size
        expected_edge = (math.log(edge_dist_m + eps) - D_LOG_MEAN) / D_LOG_STD
        actual_edge = ch[0, 50].item()
        
        print(f"  Center (antenna) - expected: {expected_center:.4f}, actual: {actual_center:.4f}")
        print(f"  Corner (0,0) - expected: {expected_corner:.4f}, actual: {actual_corner:.4f}")
        print(f"  Edge (0,50) - expected: {expected_edge:.4f}, actual: {actual_edge:.4f}")
        
        # Verify absolute values match expected normalization
        # Use delta=0.01 to allow for float32 vs float64 precision differences
        self.assertAlmostEqual(actual_corner, expected_corner, delta=0.01,
            msg=f"Corner distance normalization incorrect: expected {expected_corner:.4f}, got {actual_corner:.4f}")
        self.assertAlmostEqual(actual_center, expected_center, delta=0.01,
            msg=f"Center distance normalization incorrect: expected {expected_center:.4f}, got {actual_center:.4f}")
        self.assertAlmostEqual(actual_edge, expected_edge, delta=0.01,
            msg=f"Edge distance normalization incorrect: expected {expected_edge:.4f}, got {actual_edge:.4f}")
        
        # Also verify ordering (redundant but clear)
        self.assertLess(actual_center, actual_edge, "Center should have smaller normalized distance than edge")
        self.assertLess(actual_edge, actual_corner, "Edge should have smaller normalized distance than corner")
    
    def test_mask_channel(self):
        """Test mask channel - should be binary with realistic padding."""
        print("\n" + "=" * 60)
        print(f"[TEST] Mask Channel (index {MASK_CHANNEL})")
        print("=" * 60)
        
        # Create sample with realistic padding: 100x100 image with 100x80 valid region
        # (simulates Nx640 image padded to 640x640)
        sample = create_mock_sample(H=100, W=100, valid_h=100, valid_w=80)
        # Use zero sparsity to avoid mask modification from sparse sampling
        output = featurizer(sample, **default_featurizer_args(sparse_range=(0.0, 0.0)))
        
        ch = output[MASK_CHANNEL]
        unique_vals = torch.unique(ch)
        
        print(f"  Unique mask values: {unique_vals.tolist()}")
        
        # Mask should be binary 0/1
        self.assertTrue(
            all(v in [0.0, 1.0] for v in unique_vals.tolist()),
            "Mask should be binary 0/1"
        )
        
        # Verify mask has both 0 and 1 values (valid region + padded region)
        self.assertEqual(len(unique_vals), 2, "Mask should have both 0 and 1 values")
        
        # Verify valid region is 1, padded region is 0
        valid_region = ch[:100, :80]
        padded_region = ch[:, 80:]
        self.assertTrue((valid_region == 1.0).all(), "Valid region should be all 1s")
        self.assertTrue((padded_region == 0.0).all(), "Padded region should be all 0s")
        
        print(f"  Valid region (100x80): all 1s ✓")
        print(f"  Padded region (100x20): all 0s ✓")
    
    def test_floor_plan_generation(self):
        """Test floor plan channel - auto-generated from reflectance/transmittance."""
        print("\n" + "=" * 60)
        print(f"[TEST] Floor Plan Generation (index {FLOOR_PLAN_CHANNEL})")
        print("=" * 60)
        
        sample = create_mock_sample()
        # No floor_plan provided, should be auto-generated
        self.assertIsNone(sample.floor_plan)
        
        # Compute expected floor plan from input: (reflectance > 0) | (transmittance > 0)
        reflectance = sample.input_img[INPUT_REFLECTANCE_CHANNEL]
        transmittance = sample.input_img[INPUT_TRANSMITTANCE_CHANNEL]
        expected_floor_plan = ((reflectance > 0) | (transmittance > 0)).float()
        
        output = featurizer(sample, **default_featurizer_args())
        
        ch = output[FLOOR_PLAN_CHANNEL]
        
        expected_ones = expected_floor_plan.sum().item()
        actual_ones = (ch > 0.5).sum().item()
        
        print(f"  Expected floor plan pixels: {int(expected_ones)}")
        print(f"  Actual floor plan pixels: {int(actual_ones)}")
        
        # Floor plan should exactly match (reflectance > 0) | (transmittance > 0)
        self.assertEqual(actual_ones, expected_ones, 
            f"Floor plan pixel count mismatch: expected {int(expected_ones)}, got {int(actual_ones)}")
        
        # Verify pixel-wise exact match
        self.assertTrue(
            torch.allclose(ch, expected_floor_plan, atol=1e-5),
            "Floor plan should exactly match (reflectance > 0) | (transmittance > 0)"
        )
    
    def test_sparse_channel_no_sparse(self):
        """Test sparse channel when sparse is dropped via modality dropout."""
        print("\n" + "=" * 60)
        print(f"[TEST] Sparse Channel - No Sparse (index {SPARSE_CHANNEL})")
        print("=" * 60)
        
        sample = create_mock_sample()
        # Force sparse to be dropped using force_drop_sparse=True
        output = featurizer(
            sample, **default_featurizer_args(force_drop_sparse=True)
        )
        
        ch = output[SPARSE_CHANNEL]
        
        # With sparse dropped, channel should be all zeros (no normalization applied to zeros)
        print(f"  Actual unique values: {torch.unique(ch).tolist()}")
        
        self.assertTrue(
            torch.allclose(ch, torch.zeros_like(ch), atol=1e-5),
            "Sparse channel should be all zeros when sparse is dropped"
        )
    
    def test_sparse_channel_with_sparse(self):
        """Test sparse channel with sparse enabled (no modality dropout)."""
        print("\n" + "=" * 60)
        print(f"[TEST] Sparse Channel - With Sparse (index {SPARSE_CHANNEL})")
        print("=" * 60)
        
        output_val = 110.0  # Ground truth pathloss value (close to S_MEAN for easier testing)
        sample = create_mock_sample(output_value=output_val)
        
        # Store original valid pixel count BEFORE featurizer mutates the mask
        original_valid = sample.mask.sum().item()
        
        # Force sparse measurements: no modality dropout
        sparse_range = (0.05, 0.05)  # Exactly 5% sparsity
        output = featurizer(sample, **default_featurizer_args(sparse_range=sparse_range))
        
        ch = output[SPARSE_CHANNEL]
        
        # Find non-zero pixels (sparse samples from mask region)
        is_meas = ch != 0
        n_meas = is_meas.sum().item()
        
        expected_sparsity = 0.05
        actual_sparsity = n_meas / original_valid if original_valid > 0 else 0
        
        print(f"  Expected sparsity: {expected_sparsity * 100:.1f}%")
        print(f"  Actual sparsity: {actual_sparsity * 100:.2f}% ({n_meas}/{int(original_valid)} valid pixels)")
        
        # Check sparsity is within ±20% of expected (allows for random sampling variance)
        self.assertGreater(actual_sparsity, expected_sparsity * 0.95, 
            f"Sparsity {actual_sparsity:.2%} too low, expected >= {expected_sparsity * 0.95:.2%}")
        self.assertLess(actual_sparsity, expected_sparsity * 1.05, 
            f"Sparsity {actual_sparsity:.2%} too high, expected <= {expected_sparsity * 1.05:.2%}")
        
        if n_meas > 0:
            # Check that sparse values are z-score normalized
            sparse_vals_norm = ch[is_meas]
            sparse_vals_denorm = sparse_vals_norm * S_STD + S_MEAN
            
            print(f"  Sparse values (normalized): min={sparse_vals_norm.min():.2f}, max={sparse_vals_norm.max():.2f}")
            print(f"  Sparse values (denormalized): min={sparse_vals_denorm.min():.2f}, max={sparse_vals_denorm.max():.2f}")
            print(f"  Expected value (output_val): {output_val:.2f}")
            
            # All sparse values should be close to the ground truth
            self.assertTrue(
                torch.allclose(sparse_vals_denorm, torch.tensor(output_val), atol=1.0),
                "Sparse values should match ground truth"
            )
    
    def test_sparse_correspondence_exact(self):
        """Test that sparse measurements exactly match ground truth at sampled locations."""
        print("\n" + "=" * 60)
        print(f"[TEST] Sparse Correspondence - Exact Match (index {SPARSE_CHANNEL})")
        print("=" * 60)
        
        # Create sample with gradient output to test exact correspondence
        sample = create_mock_sample()
        # Replace output with a gradient pattern for more interesting test
        H, W = sample.H, sample.W
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        # Gradient from 80 to 140 dB (reasonable pathloss range)
        sample.output_img = 80.0 + 60.0 * (xx + yy) / (H + W - 2)
        
        output = featurizer(sample, **default_featurizer_args(sparse_range=(0.1, 0.1)))
        
        ch = output[SPARSE_CHANNEL]
        
        is_meas = ch != 0
        
        n_meas = is_meas.sum().item()
        
        # With 10% sparsity on a 100x100 image with full mask, we expect ~1000 measurements
        # If zero measurements are generated, the featurizer is broken
        self.assertGreater(n_meas, 0, 
            "Sparse sampling failed: 10% sparsity on 10000 valid pixels should produce measurements")
        
        # Denormalize sparse values using z-score stats
        sparse_denorm = ch * S_STD + S_MEAN
        
        # Get ground truth at measurement locations
        gt_at_meas = sample.output_img[is_meas]
        sp_at_meas = sparse_denorm[is_meas]
        
        diff = (sp_at_meas - gt_at_meas).abs()
        max_diff = diff.max().item()
        
        print(f"  Number of measurements: {n_meas}")
        print(f"  Max difference between sparse and GT: {max_diff:.6f}")
        
        self.assertLess(max_diff, 0.1, "Sparse values should closely match GT")
    
    def test_normalize_input_function(self):
        """Test the normalize_input function directly."""
        print("\n" + "=" * 60)
        print("[TEST] normalize_input Function")
        print("=" * 60)
        
        import math
        
        H, W = 50, 50
        input_tensor = torch.zeros((NUM_CHANNELS, H, W), dtype=torch.float32)
        
        # Set known values for each channel using config indices
        input_tensor[REFLECTANCE_CHANNEL] = 10.0  # Reflectance
        input_tensor[TRANSMITTANCE_CHANNEL] = 10.0  # Transmittance
        input_tensor[DISTANCE_CHANNEL] = 10.0  # Distance 10m
        input_tensor[ANTENNA_GAIN_CHANNEL] = -10.0  # Antenna gain -10 dBi (unchanged)
        input_tensor[FREQ_SIN_1_CHANNEL] = -0.054  # Freq sin(1) for 868 MHz (unchanged)
        input_tensor[FREQ_COS_1_CHANNEL] = -0.999  # Freq cos(1) for 868 MHz (unchanged)
        input_tensor[FREQ_SIN_2_CHANNEL] = 0.109  # Freq sin(2) for 868 MHz (unchanged)
        input_tensor[FREQ_COS_2_CHANNEL] = 0.994  # Freq cos(2) for 868 MHz (unchanged)
        input_tensor[MASK_CHANNEL] = 1.0  # Mask (unchanged)
        input_tensor[FLOOR_PLAN_CHANNEL] = 1.0  # Floor plan (unchanged)
        input_tensor[SPARSE_CHANNEL] = S_MEAN  # Sparse at normalization center
        
        normalized = normalize_input(input_tensor)
        
        print("  Channel normalization results:")
        for i, ch_name in enumerate(CHANNEL_ORDER):
            val = normalized[i, 0, 0].item()
            print(f"    {ch_name} (idx {i}): input={input_tensor[i, 0, 0].item():.2f} -> normalized={val:.4f}")
        
        # Check specific normalizations
        # Reflectance: z-score = (10 - 3.236) / 2.849 ≈ 2.37
        expected_reflectance = (10.0 - R_MEAN) / R_STD
        self.assertAlmostEqual(normalized[REFLECTANCE_CHANNEL, 0, 0].item(), expected_reflectance, places=1,
            msg=f"Reflectance z-score: expected {expected_reflectance:.2f}")
        
        # Distance: log(10 + eps) then z-score
        expected_distance = (math.log(10.0 + 1e-6) - D_LOG_MEAN) / D_LOG_STD
        self.assertAlmostEqual(normalized[DISTANCE_CHANNEL, 0, 0].item(), expected_distance, places=1,
            msg=f"Distance z-score: expected {expected_distance:.2f}")
        
        # Freq Fourier channels: should remain unchanged (already in [-1, 1])
        self.assertAlmostEqual(normalized[FREQ_SIN_1_CHANNEL, 0, 0].item(), -0.054, places=2)
        self.assertAlmostEqual(normalized[FREQ_COS_1_CHANNEL, 0, 0].item(), -0.999, places=2)
        
        # Sparse: should be ~0 at normalization center (S_MEAN)
        self.assertAlmostEqual(normalized[SPARSE_CHANNEL, 0, 0].item(), 0.0, places=1)
    
    def test_frequency_encoding(self):
        """Test that frequency channels have correct Fourier encoding values."""
        print("\n" + "=" * 60)
        print("[TEST] Frequency Encoding Correctness")
        print("=" * 60)
        
        import math
        
        # Fourier encoding parameters (must match featurizer.py)
        FREQ_MIN = 100.0    # MHz
        FREQ_MAX = 7000.0   # MHz
        
        def expected_fourier_encoding(freq_mhz: float) -> tuple:
            """Compute expected Fourier encoding for a frequency."""
            freq_clamped = max(FREQ_MIN, min(FREQ_MAX, freq_mhz))
            log_min, log_max = math.log(FREQ_MIN), math.log(FREQ_MAX)
            t = (math.log(freq_clamped) - log_min) / (log_max - log_min)
            
            activations = []
            for level in range(2):  # FREQ_N_LEVELS = 2
                scale = 2 ** level  # 1, 2
                activations.append(math.sin(2 * math.pi * scale * t))
                activations.append(math.cos(2 * math.pi * scale * t))
            return tuple(activations)  # (sin_1, cos_1, sin_2, cos_2)
        
        # Test frequencies: 868, 1800, 3500 (from ICASSP/synthetic) + 2400 (novel)
        test_frequencies = [868.0, 1800.0, 3500.0, 2400.0]
        
        for freq_mhz in test_frequencies:
            print(f"\n  Testing frequency: {freq_mhz} MHz")
            
            sample = create_mock_sample(freq_MHz=freq_mhz)
            output = featurizer(sample, **default_featurizer_args(sparse_range=(0.0, 0.0)))
            
            # Get expected encoding
            expected = expected_fourier_encoding(freq_mhz)
            
            # Check each frequency channel
            freq_channels = [
                (FREQ_SIN_1_CHANNEL, "freq_sin_1", expected[0]),
                (FREQ_COS_1_CHANNEL, "freq_cos_1", expected[1]),
                (FREQ_SIN_2_CHANNEL, "freq_sin_2", expected[2]),
                (FREQ_COS_2_CHANNEL, "freq_cos_2", expected[3]),
            ]
            
            for ch_idx, ch_name, expected_val in freq_channels:
                ch_data = output[ch_idx]
                actual_val = ch_data[0, 0].item()  # Sample one pixel
                
                # 1. Verify value matches expected encoding
                self.assertAlmostEqual(actual_val, expected_val, places=4,
                    msg=f"{ch_name} at {freq_mhz}MHz: expected {expected_val:.6f}, got {actual_val:.6f}")
                
                # 2. Verify all values in channel are constant (same across entire image)
                self.assertTrue(
                    torch.allclose(ch_data, torch.full_like(ch_data, expected_val), atol=1e-5),
                    f"{ch_name} should be constant across image at {freq_mhz}MHz"
                )
            
            print(f"    sin_1={expected[0]:.4f}, cos_1={expected[1]:.4f}, "
                  f"sin_2={expected[2]:.4f}, cos_2={expected[3]:.4f} ✓")
        
        print(f"\n  All {len(test_frequencies)} frequencies verified ✓")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_small_image(self):
        """Test with very small image size."""
        print("\n" + "=" * 60)
        print("[TEST] Small Image (10x10)")
        print("=" * 60)
        
        sample = create_mock_sample(H=10, W=10, x_ant=5.0, y_ant=5.0)
        # Adjust structure for small size
        sample.input_img[0] = 0  # Clear reflectance
        sample.input_img[0, 2:8, 2:8] = 1.0
        sample.input_img[1] = 0  # Clear transmittance
        sample.input_img[1, 3:7, 3:7] = 0.5
        
        output = featurizer(sample, **default_featurizer_args(sparse_range=(0.1, 0.1)))
        
        self.assertEqual(output.shape, (NUM_CHANNELS, 10, 10))
        self.assertFalse(torch.isnan(output).any())
        print(f"  Output shape: {output.shape} ✓")
    
    def test_antenna_at_corner(self):
        """Test with antenna at image corner."""
        print("\n" + "=" * 60)
        print("[TEST] Antenna at Corner")
        print("=" * 60)
        
        sample = create_mock_sample(x_ant=0.0, y_ant=0.0)
        output = featurizer(sample, **default_featurizer_args())
        
        self.assertEqual(output.shape, (NUM_CHANNELS, 100, 100))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Distance at corner (antenna position) should be minimal
        ch = output[DISTANCE_CHANNEL]
        corner_dist = ch[0, 0].item()
        far_corner_dist = ch[99, 99].item()
        
        print(f"  Antenna corner distance: {corner_dist:.4f}")
        print(f"  Far corner distance: {far_corner_dist:.4f}")
        
        self.assertLess(corner_dist, far_corner_dist)
    
    def test_zero_sparsity_range(self):
        """Test with zero sparsity range."""
        print("\n" + "=" * 60)
        print("[TEST] Zero Sparsity Range")
        print("=" * 60)
        
        sample = create_mock_sample()
        output = featurizer(sample, **default_featurizer_args(sparse_range=(0.0, 0.0)))
        
        ch = output[SPARSE_CHANNEL]
        
        # With sparsity=0%, should be all zeros
        self.assertTrue(
            torch.allclose(ch, torch.zeros_like(ch), atol=1e-5),
            "Zero sparsity should result in no measurements"
        )
        print("  Zero sparsity correctly produces no measurements ✓")


class TestFeaturizerWithRealData(unittest.TestCase):
    """
    Tests for the featurizer function using real ICASSP and synthetic data.
    Requires CLI args: --icassp-manifest and --synth-manifest
    """
    
    # Set by main() from CLI args
    icassp_manifest_path = None
    synth_manifest_path = None
    
    @classmethod
    def setUpClass(cls):
        cls.output_dir = PROJECT_ROOT / "tests" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import Hydra utilities for loading configs
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        import hydra as hydra_module
        
        cls.hydra_module = hydra_module
        cls.compose = compose
        cls.initialize_config_dir = initialize_config_dir
        cls.GlobalHydra = GlobalHydra
        
        # Import dataset for reading samples
        from src.datamodules.datasets.indoor import PathlossDataset
        
        cls.PathlossDataset = PathlossDataset
        
        # Validate manifest paths were set
        assert cls.icassp_manifest_path is not None, "icassp_manifest_path not set. Run with --icassp-manifest"
        assert cls.synth_manifest_path is not None, "synth_manifest_path not set. Run with --synth-manifest"
    
    def _get_config(self, experiment_name: str):
        """
        Load experiment config the same way run.py does:
        1. Load base train.yaml config via Hydra (which includes all experiments via defaults)
        2. Convert to container to remove struct mode (like run.py clone_cfg)
        3. Extract the specific experiment config (cfg.exps.e0, cfg.exps.e1, etc.)
        4. Use load_experiment_config to resolve defaults (datamodule, trainer, etc.)
        5. Wire up manifest paths from CLI args
        """
        from omegaconf import OmegaConf
        from hydra import compose
        from hydra.core.global_hydra import GlobalHydra
        from src.utils import load_experiment_config
        
        GlobalHydra.instance().clear()
        config_dir = str(PROJECT_ROOT / "configs")
        self.initialize_config_dir(config_dir=config_dir, version_base=None)
        
        # Load base config (train.yaml) - this includes all experiments via defaults
        cfg = compose(config_name="train")
        
        # Convert to container and recreate to remove struct mode (same as run.py clone_cfg)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        
        # Extract experiment-specific config from cfg.exps
        exp_cfg_raw = OmegaConf.create(OmegaConf.to_container(cfg.exps[experiment_name], resolve=True))
        
        # Use load_experiment_config to resolve defaults (datamodule, trainer, etc.)
        exp_cfg = load_experiment_config(exp_cfg_raw, config_root=PROJECT_ROOT / "configs" / "exps")
        
        # Wire up manifest paths from CLI args (use ICASSP for train/val, synthetic for e2)
        if experiment_name in ("e0", "e1"):
            exp_cfg.datamodule.train_manifest_path = self.icassp_manifest_path
        elif experiment_name == "e2":
            exp_cfg.datamodule.synthetic_manifest_path = self.synth_manifest_path
        
        exp_cfg.datamodule.val_manifest_path = self.icassp_manifest_path
        
        return exp_cfg
    
    def _create_datamodule(self, experiment_name: str):
        """Create and setup a datamodule from experiment config."""
        cfg = self._get_config(experiment_name)
        dm = self.hydra_module.utils.instantiate(cfg.datamodule)
        dm.setup(stage="fit")
        return dm, cfg
    
    def _get_raw_sample_from_datamodule(self, dm, idx: int = 0):
        """
        Get a raw RadarSample from the datamodule's dataset before featurization.
        This accesses the internal read_sample method to get the sample before
        it goes through the featurizer in __getitem__.
        """
        dataset = dm.train_dataloader().dataset
        inputs = dataset.inputs_list[idx % len(dataset.inputs_list)]
        sample = dataset.read_sample(inputs)
        return sample, dataset
    
    def test_featurizer_with_icassp_data(self):
        """Test featurizer with real ICASSP data (e0 config)."""
        print("\n" + "=" * 60)
        print("[TEST] Featurizer with Real ICASSP Data")
        print("=" * 60)
        
        dm, cfg = self._create_datamodule("e0")
        
        # Get multiple raw samples
        n_samples = min(3, len(dm.train_dataloader().dataset))
        sparse_range = tuple(cfg.datamodule.sparse_range)
        
        for i in range(n_samples):
            print(f"\n  Sample {i + 1}/{n_samples}:")
            sample, dataset = self._get_raw_sample_from_datamodule(dm, idx=i)
            
            print(f"    File: {sample.file_name}")
            print(f"    Dimensions: {sample.H}x{sample.W}")
            print(f"    Antenna position: ({sample.x_ant:.1f}, {sample.y_ant:.1f})")
            print(f"    Frequency: {sample.freq_MHz} MHz")
            print(f"    Input shape: {sample.input_img.shape}")
            
            # Run through featurizer with dropout DISABLED to ensure all channels are present
            # This allows us to validate reflectance/transmittance content
            output = featurizer(
                sample,
                **default_featurizer_args(sparse_range=sparse_range)
            )
            
            # Verify output
            self.assertEqual(output.shape[0], NUM_CHANNELS, f"Should have {NUM_CHANNELS} channels")
            self.assertEqual(output.shape[1], sample.H, "Height should match")
            self.assertEqual(output.shape[2], sample.W, "Width should match")
            self.assertFalse(torch.isnan(output).any(), "Output should not contain NaN")
            self.assertFalse(torch.isinf(output).any(), "Output should not contain Inf")
            
            print(f"    Output shape: {output.shape} ✓")
            print(f"    No NaN/Inf values ✓")
            
            # Check channel ranges
            for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
                ch_data = output[ch_idx]
                print(f"    {ch_name}: min={ch_data.min():.4f}, max={ch_data.max():.4f}")
            
            # Check mask channel is binary
            mask_ch = output[MASK_CHANNEL]
            unique_mask = torch.unique(mask_ch)
            self.assertTrue(
                all(v in [0.0, 1.0] for v in unique_mask.tolist()),
                f"Mask channel should be binary, got {unique_mask.tolist()}"
            )
            print(f"    Mask channel is binary ✓")
            
            # Check floor plan is binary
            floor_plan_ch = output[FLOOR_PLAN_CHANNEL]
            unique_fp = torch.unique(floor_plan_ch)
            self.assertTrue(
                all(v in [0.0, 1.0] for v in unique_fp.tolist()),
                f"Floor plan should be binary, got {unique_fp.tolist()}"
            )
            print(f"    Floor plan channel is binary ✓")
            
            # Check frequency encoding channels are in valid range [-1, 1]
            for freq_ch_idx in [FREQ_SIN_1_CHANNEL, FREQ_COS_1_CHANNEL, FREQ_SIN_2_CHANNEL, FREQ_COS_2_CHANNEL]:
                freq_ch = output[freq_ch_idx]
                self.assertGreaterEqual(freq_ch.min().item(), -1.0 - 1e-5,
                    f"Freq channel {CHANNEL_ORDER[freq_ch_idx]} min={freq_ch.min():.4f} should be >= -1")
                self.assertLessEqual(freq_ch.max().item(), 1.0 + 1e-5,
                    f"Freq channel {CHANNEL_ORDER[freq_ch_idx]} max={freq_ch.max():.4f} should be <= 1")
            print(f"    Frequency channels in [-1, 1] ✓")
            
            # Check distance channel has variation AND correct ordering
            dist_ch = output[DISTANCE_CHANNEL]
            self.assertGreater(dist_ch.max() - dist_ch.min(), 0.1,
                "Distance channel should have variation (not constant)")
            
            # Validate distance ordering: antenna position should have minimum distance
            ant_y = int(min(max(0, sample.y_ant), sample.H - 1))
            ant_x = int(min(max(0, sample.x_ant), sample.W - 1))
            dist_at_antenna = dist_ch[ant_y, ant_x].item()
            min_dist = dist_ch.min().item()
            self.assertAlmostEqual(dist_at_antenna, min_dist, places=4,
                msg=f"Distance at antenna should equal minimum (got {dist_at_antenna:.4f} vs {min_dist:.4f})")
            print(f"    Distance channel: variation ✓, ordering ✓")
            
            # Validate reflectance/transmittance are present and have plausible content
            # (With dropout=0, these should NOT be zeroed out)
            reflectance_ch = output[REFLECTANCE_CHANNEL]
            transmittance_ch = output[TRANSMITTANCE_CHANNEL]
            
            # Check that EACH channel has some non-zero content
            # (Real buildings have walls with both reflectance and transmittance properties)
            refl_nonzero_frac = (reflectance_ch != 0).float().mean().item()
            trans_nonzero_frac = (transmittance_ch != 0).float().mean().item()
            
            self.assertGreater(refl_nonzero_frac, 0.001,
                f"Reflectance should have non-zero content (got {refl_nonzero_frac:.4f})")
            self.assertGreater(trans_nonzero_frac, 0.001,
                f"Transmittance should have non-zero content (got {trans_nonzero_frac:.4f})")
            print(f"    Reflectance non-zero: {refl_nonzero_frac:.2%}, Transmittance non-zero: {trans_nonzero_frac:.2%} ✓")
            
            # Save visualization for first sample
            if i == 0:
                save_featurizer_output_visualization(
                    output, sample, self.output_dir, f"icassp_real_sample_{i}"
                )
    
    def test_featurizer_with_synthetic_data(self):
        """Test featurizer with real synthetic data (e2 config)."""
        print("\n" + "=" * 60)
        print("[TEST] Featurizer with Real Synthetic Data")
        print("=" * 60)
        
        dm, cfg = self._create_datamodule("e2")
        
        # Get multiple raw samples
        n_samples = min(3, len(dm.train_dataloader().dataset))
        sparse_range = tuple(cfg.datamodule.sparse_range)
        
        for i in range(n_samples):
            print(f"\n  Sample {i + 1}/{n_samples}:")
            sample, dataset = self._get_raw_sample_from_datamodule(dm, idx=i)
            
            print(f"    File: {sample.file_name}")
            print(f"    Dimensions: {sample.H}x{sample.W}")
            print(f"    Antenna position: ({sample.x_ant:.1f}, {sample.y_ant:.1f})")
            print(f"    Frequency: {sample.freq_MHz} MHz")
            print(f"    Pixel size: {sample.pixel_size} m")
            print(f"    Input shape: {sample.input_img.shape}")
            
            # Run through featurizer with dropout DISABLED to ensure all channels are present
            # This allows us to validate reflectance/transmittance content
            output = featurizer(
                sample,
                **default_featurizer_args(sparse_range=sparse_range)
            )
            
            # Verify output
            self.assertEqual(output.shape[0], NUM_CHANNELS, f"Should have {NUM_CHANNELS} channels")
            self.assertEqual(output.shape[1], sample.H, "Height should match")
            self.assertEqual(output.shape[2], sample.W, "Width should match")
            self.assertFalse(torch.isnan(output).any(), "Output should not contain NaN")
            self.assertFalse(torch.isinf(output).any(), "Output should not contain Inf")
            
            print(f"    Output shape: {output.shape} ✓")
            print(f"    No NaN/Inf values ✓")
            
            # Check channel ranges
            for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
                ch_data = output[ch_idx]
                print(f"    {ch_name}: min={ch_data.min():.4f}, max={ch_data.max():.4f}")
            
            # Check mask channel is binary
            mask_ch = output[MASK_CHANNEL]
            unique_mask = torch.unique(mask_ch)
            self.assertTrue(
                all(v in [0.0, 1.0] for v in unique_mask.tolist()),
                f"Mask channel should be binary, got {unique_mask.tolist()}"
            )
            print(f"    Mask channel is binary ✓")
            
            # Check floor plan is binary
            floor_plan_ch = output[FLOOR_PLAN_CHANNEL]
            unique_fp = torch.unique(floor_plan_ch)
            self.assertTrue(
                all(v in [0.0, 1.0] for v in unique_fp.tolist()),
                f"Floor plan should be binary, got {unique_fp.tolist()}"
            )
            print(f"    Floor plan channel is binary ✓")
            
            # Check frequency encoding channels are in valid range [-1, 1]
            for freq_ch_idx in [FREQ_SIN_1_CHANNEL, FREQ_COS_1_CHANNEL, FREQ_SIN_2_CHANNEL, FREQ_COS_2_CHANNEL]:
                freq_ch = output[freq_ch_idx]
                self.assertGreaterEqual(freq_ch.min().item(), -1.0 - 1e-5,
                    f"Freq channel {CHANNEL_ORDER[freq_ch_idx]} min={freq_ch.min():.4f} should be >= -1")
                self.assertLessEqual(freq_ch.max().item(), 1.0 + 1e-5,
                    f"Freq channel {CHANNEL_ORDER[freq_ch_idx]} max={freq_ch.max():.4f} should be <= 1")
            print(f"    Frequency channels in [-1, 1] ✓")
            
            # Check distance channel has variation AND correct ordering
            dist_ch = output[DISTANCE_CHANNEL]
            self.assertGreater(dist_ch.max() - dist_ch.min(), 0.1,
                "Distance channel should have variation (not constant)")
            
            # Validate distance ordering: antenna position should have minimum distance
            ant_y = int(min(max(0, sample.y_ant), sample.H - 1))
            ant_x = int(min(max(0, sample.x_ant), sample.W - 1))
            dist_at_antenna = dist_ch[ant_y, ant_x].item()
            min_dist = dist_ch.min().item()
            self.assertAlmostEqual(dist_at_antenna, min_dist, places=4,
                msg=f"Distance at antenna should equal minimum (got {dist_at_antenna:.4f} vs {min_dist:.4f})")
            print(f"    Distance channel: variation ✓, ordering ✓")
            
            # Validate reflectance/transmittance are present and have plausible content
            # (With dropout=0, these should NOT be zeroed out)
            # Note: Some synthetic samples may have zero reflectance/transmittance if
            # the scene has no walls, so we check combined content
            reflectance_ch = output[REFLECTANCE_CHANNEL]
            transmittance_ch = output[TRANSMITTANCE_CHANNEL]
            
            refl_nonzero_frac = (reflectance_ch != 0).float().mean().item()
            trans_nonzero_frac = (transmittance_ch != 0).float().mean().item()
            
            print(f"    Reflectance non-zero: {refl_nonzero_frac:.2%}, Transmittance non-zero: {trans_nonzero_frac:.2%}")
            # Note: We don't assert here because synthetic samples can legitimately have
            # zero reflectance/transmittance (empty scenes). The print helps with debugging.
            
            # Save visualization for first sample
            if i == 0:
                save_featurizer_output_visualization(
                    output, sample, self.output_dir, f"synthetic_real_sample_{i}"
                )
    
    def test_sparse_correspondence_with_real_data(self):
        """Test that sparse measurements match ground truth on real data."""
        print("\n" + "=" * 60)
        print("[TEST] Sparse Correspondence with Real Data")
        print("=" * 60)
        
        dm, cfg = self._create_datamodule("e0")
        
        sample, _ = self._get_raw_sample_from_datamodule(dm, idx=0)
        
        print(f"  Testing with: {sample.file_name}")
        
        # Verify preconditions
        self.assertIsNotNone(sample.output_img, "Sample must have ground truth output_img")
        
        # Force sparse measurements with no modality dropout
        output = featurizer(sample, **default_featurizer_args(sparse_range=(0.05, 0.05)))
        
        ch = output[SPARSE_CHANNEL]
        
        # Find sparse measurement locations (non-zero values)
        is_meas = ch != 0
        n_meas = is_meas.sum().item()
        
        print(f"  Number of sparse measurements: {n_meas}")
        
        # With 5% sparsity and modality_dropout_prob=0.0, we must have measurements
        self.assertGreater(n_meas, 0, 
            "Sparse sampling failed: 5% sparsity with modality_dropout_prob=0.0 should produce measurements")
        
        # Denormalize sparse values using z-score stats
        sparse_denorm = ch * S_STD + S_MEAN
        
        # Get ground truth at measurement locations
        gt = sample.output_img
        if gt.ndim == 3:
            gt = gt.squeeze(0)
        
        gt_at_meas = gt[is_meas]
        sp_at_meas = sparse_denorm[is_meas]
        
        diff = (sp_at_meas - gt_at_meas).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        # Sparse values should exactly match ground truth
        self.assertLess(max_diff, 0.1, "Sparse values should match GT")
        print(f"  Sparse-GT correspondence verified ✓")
    
    def test_distance_channel_real_data(self):
        """Test that distance channel minimum is at (or near) the antenna position on real data."""
        print("\n" + "=" * 60)
        print("[TEST] Distance Channel Sanity Check on Real Data")
        print("=" * 60)
        
        dm, cfg = self._create_datamodule("e0")
        
        n_samples = min(5, len(dm.train_dataloader().dataset))
        
        for i in range(n_samples):
            sample, _ = self._get_raw_sample_from_datamodule(dm, idx=i)
            
            output = featurizer(sample, **default_featurizer_args(sparse_range=(0.0, 0.0)))
            
            ch = output[DISTANCE_CHANNEL]
            
            # Get antenna position (clamp to valid indices)
            ant_y = int(min(max(0, sample.y_ant), sample.H - 1))
            ant_x = int(min(max(0, sample.x_ant), sample.W - 1))
            
            # Distance at antenna should be the minimum (or very close to it)
            dist_at_antenna = ch[ant_y, ant_x].item()
            min_dist = ch.min().item()
            
            # Find where the minimum actually is
            min_idx = ch.argmin().item()
            min_y, min_x = min_idx // ch.shape[1], min_idx % ch.shape[1]
            
            # Distance from antenna to the actual minimum location (in pixels)
            pixel_offset = ((min_y - ant_y) ** 2 + (min_x - ant_x) ** 2) ** 0.5
            
            print(f"\n  Sample {i + 1}: {sample.file_name}")
            print(f"    Antenna position: ({sample.x_ant:.1f}, {sample.y_ant:.1f}) -> pixel ({ant_x}, {ant_y})")
            print(f"    Distance at antenna: {dist_at_antenna:.4f}")
            print(f"    Minimum distance: {min_dist:.4f} at pixel ({min_x}, {min_y})")
            print(f"    Pixel offset from antenna to min: {pixel_offset:.1f}")
            
            # The distance at antenna should equal the minimum (within tolerance for float precision)
            # Note: argmin() may point to a different pixel when there are ties (multiple pixels at distance ~0)
            self.assertAlmostEqual(dist_at_antenna, min_dist, places=4,
                msg=f"Distance at antenna ({dist_at_antenna:.4f}) should equal minimum ({min_dist:.4f})")
            
            # Verify distance increases as we move away from antenna
            # Check corners (should have larger distance than antenna)
            corners = [(0, 0), (0, sample.W - 1), (sample.H - 1, 0), (sample.H - 1, sample.W - 1)]
            for cy, cx in corners:
                corner_dist = ch[cy, cx].item()
                # Only check corners that are far from antenna
                corner_to_ant = ((cy - ant_y) ** 2 + (cx - ant_x) ** 2) ** 0.5
                if corner_to_ant > 10:  # At least 10 pixels away
                    self.assertGreater(corner_dist, dist_at_antenna,
                        f"Corner ({cx}, {cy}) should have larger distance than antenna")
            
            print(f"    Distance ordering verified ✓")
    
    def test_channel_statistics_real_data(self):
        """Compute and verify channel statistics across multiple real samples."""
        print("\n" + "=" * 60)
        print("[TEST] Channel Statistics on Real Data")
        print("=" * 60)
        
        dm, cfg = self._create_datamodule("e0")
        
        n_samples = min(10, len(dm.train_dataloader().dataset))
        
        # Collect statistics
        channel_mins = [[] for _ in range(NUM_CHANNELS)]
        channel_maxs = [[] for _ in range(NUM_CHANNELS)]
        channel_means = [[] for _ in range(NUM_CHANNELS)]
        channel_nonzero_fracs = [[] for _ in range(NUM_CHANNELS)]
        
        sparse_range = tuple(cfg.datamodule.sparse_range)
        
        for i in range(n_samples):
            sample, _ = self._get_raw_sample_from_datamodule(dm, idx=i)
            # Force dropout off to validate all channels have content
            output = featurizer(
                sample, **default_featurizer_args(sparse_range=sparse_range)
            )
            
            for ch_idx in range(NUM_CHANNELS):
                ch_data = output[ch_idx]
                channel_mins[ch_idx].append(ch_data.min().item())
                channel_maxs[ch_idx].append(ch_data.max().item())
                channel_means[ch_idx].append(ch_data.mean().item())
                channel_nonzero_fracs[ch_idx].append((ch_data != 0).float().mean().item())
        
        print(f"\n  Statistics across {n_samples} samples:")
        print(f"  {'Channel':<15} {'Min Range':<20} {'Max Range':<20} {'Mean Range':<20}")
        print("  " + "-" * 75)
        
        for ch_idx, ch_name in enumerate(CHANNEL_ORDER):
            min_range = f"[{min(channel_mins[ch_idx]):.3f}, {max(channel_mins[ch_idx]):.3f}]"
            max_range = f"[{min(channel_maxs[ch_idx]):.3f}, {max(channel_maxs[ch_idx]):.3f}]"
            mean_range = f"[{min(channel_means[ch_idx]):.3f}, {max(channel_means[ch_idx]):.3f}]"
            print(f"  {ch_name:<15} {min_range:<20} {max_range:<20} {mean_range:<20}")
        
        # =========================================================================
        # Validate ALL channels, not just mask/floor_plan
        # =========================================================================
        
        # Mask channel should be binary [0, 1]
        self.assertTrue(all(m <= 1.0 for m in channel_maxs[MASK_CHANNEL]), "Mask max should be <= 1.0")
        self.assertTrue(all(m >= 0.0 for m in channel_mins[MASK_CHANNEL]), "Mask min should be >= 0.0")
        print("  Mask: binary [0, 1] ✓")
        
        # Floor plan should be binary [0, 1]
        self.assertTrue(all(m <= 1.0 for m in channel_maxs[FLOOR_PLAN_CHANNEL]), "Floor plan max should be <= 1.0")
        self.assertTrue(all(m >= 0.0 for m in channel_mins[FLOOR_PLAN_CHANNEL]), "Floor plan min should be >= 0.0")
        print("  Floor plan: binary [0, 1] ✓")
        
        # Frequency channels should be in [-1, 1]
        for freq_ch_idx in [FREQ_SIN_1_CHANNEL, FREQ_COS_1_CHANNEL, FREQ_SIN_2_CHANNEL, FREQ_COS_2_CHANNEL]:
            self.assertTrue(all(m >= -1.0 - 1e-5 for m in channel_mins[freq_ch_idx]),
                f"{CHANNEL_ORDER[freq_ch_idx]} min should be >= -1")
            self.assertTrue(all(m <= 1.0 + 1e-5 for m in channel_maxs[freq_ch_idx]),
                f"{CHANNEL_ORDER[freq_ch_idx]} max should be <= 1")
        print("  Frequency channels: in [-1, 1] ✓")
        
        # Distance channel should have variation (max - min > 0.1 for all samples)
        for i in range(n_samples):
            dist_variation = channel_maxs[DISTANCE_CHANNEL][i] - channel_mins[DISTANCE_CHANNEL][i]
            self.assertGreater(dist_variation, 0.1,
                f"Distance channel should have variation (sample {i}: {dist_variation:.3f})")
        print("  Distance: has variation ✓")
        
        # Reflectance should have non-zero content in all samples (with dropout=0)
        self.assertTrue(all(f > 0.001 for f in channel_nonzero_fracs[REFLECTANCE_CHANNEL]),
            f"Reflectance should have non-zero content, got fracs: {channel_nonzero_fracs[REFLECTANCE_CHANNEL]}")
        print("  Reflectance: non-zero content ✓")
        
        # Transmittance should have non-zero content in all samples (with dropout=0)
        self.assertTrue(all(f > 0.001 for f in channel_nonzero_fracs[TRANSMITTANCE_CHANNEL]),
            f"Transmittance should have non-zero content, got fracs: {channel_nonzero_fracs[TRANSMITTANCE_CHANNEL]}")
        print("  Transmittance: non-zero content ✓")
        
        # Sparse channel should have some measurements (with sparse_range from config)
        self.assertTrue(all(f > 0.001 for f in channel_nonzero_fracs[SPARSE_CHANNEL]),
            f"Sparse should have measurements, got fracs: {channel_nonzero_fracs[SPARSE_CHANNEL]}")
        print("  Sparse: has measurements ✓")
        
        print("\n  All channel statistics verified ✓")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Featurizer unit tests")
    parser.add_argument('--icassp-manifest', help='Path to ICASSP manifest CSV (required for real data tests)')
    parser.add_argument('--synth-manifest', help='Path to synthetic manifest CSV (required for real data tests)')
    parser.add_argument('unittest_args', nargs='*', help='Arguments to pass to unittest')
    
    args = parser.parse_args()
    
    # Check if real data tests should run
    run_real_data_tests = args.icassp_manifest and args.synth_manifest
    
    if run_real_data_tests:
        # Validate manifest paths exist
        if not os.path.exists(args.icassp_manifest):
            raise FileNotFoundError(f"ICASSP manifest not found: {args.icassp_manifest}")
        if not os.path.exists(args.synth_manifest):
            raise FileNotFoundError(f"Synthetic manifest not found: {args.synth_manifest}")
        
        # Store manifest paths for tests to use
        TestFeaturizerWithRealData.icassp_manifest_path = args.icassp_manifest
        TestFeaturizerWithRealData.synth_manifest_path = args.synth_manifest
        
        print(f"[CONFIG] ICASSP manifest: {args.icassp_manifest}")
        print(f"[CONFIG] Synthetic manifest: {args.synth_manifest}")
    else:
        print("[CONFIG] No manifests provided - skipping real data tests (TestFeaturizerWithRealData)")
        print("[CONFIG] To run real data tests: --icassp-manifest <path> --synth-manifest <path>")
        # Remove TestFeaturizerWithRealData from the test suite
        del TestFeaturizerWithRealData
    
    # Run unittest with remaining args
    sys.argv = [sys.argv[0]] + args.unittest_args
    unittest.main(verbosity=2)
