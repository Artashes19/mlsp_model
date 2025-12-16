"""
Unit tests for the featurizer function using fully controlled mock data.
These tests verify:
1. Output tensor shape (9 channels)
2. Each channel contains expected values based on controlled inputs
3. Normalization is applied correctly
4. Sparse sampling logic works as expected
5. Floor plan generation from reflectance/transmittance
"""
import os
import sys
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mlsp.types import RadarSample
from src.utils.mlsp.featurizer import featurizer, normalize_input, get_fspl, calculate_antenna_gain

# Normalization constants from featurizer
NORM_OFFSET = 87.0
NORM_SCALE = 160.0


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
    mask_value: float = 1.0,
) -> RadarSample:
    """
    Create a fully controlled mock RadarSample for testing.
    
    Default setup:
    - 100x100 image
    - Antenna at center (50, 50)
    - Square structure: reflectance=1 in [20:80, 20:80], transmittance=0.5 in [30:70, 30:70]
    - Distance increases linearly from antenna
    - Uniform output pathloss
    - Full valid mask
    """
    # Reflectance: outer square wall
    reflectance = torch.zeros((H, W), dtype=torch.float32)
    reflectance[20:80, 20:80] = reflectance_value
    
    # Transmittance: inner square (subset of reflectance region)
    transmittance = torch.zeros((H, W), dtype=torch.float32)
    transmittance[30:70, 30:70] = transmittance_value
    
    # Distance map: Euclidean distance from antenna in meters
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist_px = torch.sqrt((xx - x_ant)**2 + (yy - y_ant)**2)
    dist_m = dist_px * pixel_size
    
    # Stack into input_img (3 channels: reflectance, transmittance, distance)
    input_img = torch.stack([reflectance, transmittance, dist_m])
    
    # Output: uniform pathloss for simplicity (real data would have gradients)
    output_img = torch.full((H, W), output_value, dtype=torch.float32)
    
    # Mask: all valid
    mask = torch.full((H, W), mask_value, dtype=torch.float32)
    
    # Radiation pattern: uniform (isotropic antenna)
    radiation_pattern = torch.zeros(360, dtype=torch.float32)
    
    return RadarSample(
        file_name="mock_sample",
        task_idx=0,
        pl_clip=None,
        use_fspl=True,
        use_transmittance_loss=True,
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
    
    channel_names = [
        "Ch0: Reflectance (norm)",
        "Ch1: Transmittance (norm)",
        "Ch2: Distance (log)",
        "Ch3: Antenna Gain (norm)",
        "Ch4: Frequency (log)",
        "Ch5: Mask",
        "Ch6: Floor Plan",
        "Ch7: Approx FSPL (norm)",
        "Ch8: Sparse Meas (norm)",
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(9):
        ax = axes[i]
        data = output_tensor[i].numpy()
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f"{channel_names[i]}\nmin={data.min():.4f}, max={data.max():.4f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
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
        """Test that featurizer outputs correct shape (9 channels)."""
        print("\n" + "="*60)
        print("[TEST] Featurizer Output Shape")
        print("="*60)
        
        sample = create_mock_sample()
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        self.assertEqual(output.shape[0], 9, "Should have 9 channels")
        self.assertEqual(output.shape[1], sample.H, "Height should match")
        self.assertEqual(output.shape[2], sample.W, "Width should match")
        print(f"  Output shape: {output.shape} ✓")
    
    def test_reflectance_channel(self):
        """Test channel 0 (reflectance) normalization."""
        print("\n" + "="*60)
        print("[TEST] Reflectance Channel (Ch0)")
        print("="*60)
        
        sample = create_mock_sample(reflectance_value=255.0)  # Max value
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch0 = output[0]
        
        # ImageNet normalization: (val/255 - 0.485) / 0.229
        # For val=255: (1.0 - 0.485) / 0.229 ≈ 2.249
        # For val=0: (0.0 - 0.485) / 0.229 ≈ -2.118
        
        # Check non-zero region (where we set reflectance=255)
        non_zero_region = ch0[20:80, 20:80]
        zero_region = ch0[0:20, 0:20]
        
        expected_non_zero = (255/255 - 0.485) / 0.229
        expected_zero = (0/255 - 0.485) / 0.229
        
        print(f"  Non-zero region: min={non_zero_region.min():.4f}, max={non_zero_region.max():.4f}")
        print(f"  Expected for 255: {expected_non_zero:.4f}")
        print(f"  Zero region: min={zero_region.min():.4f}, max={zero_region.max():.4f}")
        print(f"  Expected for 0: {expected_zero:.4f}")
        
        # Allow small tolerance
        self.assertTrue(
            torch.allclose(non_zero_region, torch.tensor(expected_non_zero), atol=0.01),
            f"Reflectance normalization mismatch"
        )
    
    def test_transmittance_channel(self):
        """Test channel 1 (transmittance) normalization."""
        print("\n" + "="*60)
        print("[TEST] Transmittance Channel (Ch1)")
        print("="*60)
        
        sample = create_mock_sample(transmittance_value=127.5)  # Mid value
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch1 = output[1]
        
        # ImageNet normalization: (val/255 - 0.456) / 0.224
        expected_mid = (127.5/255 - 0.456) / 0.224
        
        non_zero_region = ch1[30:70, 30:70]
        print(f"  Transmittance region: mean={non_zero_region.mean():.4f}")
        print(f"  Expected for 127.5: {expected_mid:.4f}")
    
    def test_distance_channel(self):
        """Test channel 2 (distance) - log transformation."""
        print("\n" + "="*60)
        print("[TEST] Distance Channel (Ch2)")
        print("="*60)
        
        sample = create_mock_sample()
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch2 = output[2]
        
        # Distance is log10(1 + dist_m)
        # At antenna (50,50): dist=0, log10(1+0)=0
        # At corner: dist ≈ sqrt(50^2 + 50^2) * 0.25 ≈ 17.68m, log10(1+17.68) ≈ 1.27
        
        center_val = ch2[50, 50].item()
        corner_val = ch2[0, 0].item()
        
        print(f"  Center (antenna) value: {center_val:.4f}")
        print(f"  Corner value: {corner_val:.4f}")
        
        # Center should be log10(1 + small_dist) ≈ 0
        self.assertLess(abs(center_val), 0.5, "Center distance should be near 0")
        # Corner should be positive (log of distance)
        self.assertGreater(corner_val, 0.5, "Corner distance should be positive")
    
    def test_mask_channel(self):
        """Test channel 5 (mask) - should be binary."""
        print("\n" + "="*60)
        print("[TEST] Mask Channel (Ch5)")
        print("="*60)
        
        sample = create_mock_sample(mask_value=1.0)
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch5 = output[5]
        unique_vals = torch.unique(ch5)
        
        print(f"  Unique mask values: {unique_vals.tolist()}")
        
        # Mask should remain unchanged (not normalized)
        self.assertTrue(
            all(v in [0.0, 1.0] for v in unique_vals.tolist()),
            "Mask should be binary 0/1"
        )
    
    def test_floor_plan_generation(self):
        """Test channel 6 (floor plan) - auto-generated from reflectance/transmittance."""
        print("\n" + "="*60)
        print("[TEST] Floor Plan Generation (Ch6)")
        print("="*60)
        
        sample = create_mock_sample()
        # No floor_plan provided, should be auto-generated
        self.assertIsNone(sample.floor_plan)
        
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch6 = output[6]
        
        # Floor plan = (reflectance > 0) | (transmittance > 0)
        # reflectance is 1 in [20:80, 20:80]
        # transmittance is 0.5 in [30:70, 30:70] (subset)
        # So floor plan should be 1 in [20:80, 20:80]
        
        expected_ones = (80-20) * (80-20)  # 3600 pixels
        actual_ones = (ch6 > 0.5).sum().item()
        
        print(f"  Expected floor plan pixels: {expected_ones}")
        print(f"  Actual floor plan pixels: {actual_ones}")
        
        self.assertEqual(actual_ones, expected_ones, "Floor plan should match union of reflectance/transmittance")
    
    def test_approximation_feature(self):
        """Test channel 7 (approximation/FSPL) - should be computed from FSPL."""
        print("\n" + "="*60)
        print("[TEST] Approximation Feature (Ch7)")
        print("="*60)
        
        sample = create_mock_sample()
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch7 = output[7]
        
        # FSPL formula: 20*log10(d) + 20*log10(f) - 27.55 - antenna_gain
        # At center (d≈0.125m clamped), f=868MHz: FSPL ≈ 20*log10(0.125) + 20*log10(868) - 27.55 ≈ 12.3 dB
        # Normalization: (val - 87) / 160
        
        # The approximation should not be all zeros
        self.assertGreater(ch7.abs().sum().item(), 0, "Approx feature should not be empty")
        
        # Check reasonable range after normalization
        # FSPL values typically range from ~10 to ~100 dB
        # Normalized: (10-87)/160 ≈ -0.48 to (100-87)/160 ≈ 0.08
        print(f"  Ch7 stats: min={ch7.min():.4f}, max={ch7.max():.4f}, mean={ch7.mean():.4f}")
        
        # Should increase with distance from antenna (more path loss)
        center_val = ch7[50, 50].item()
        corner_val = ch7[0, 0].item()
        print(f"  Center value: {center_val:.4f}")
        print(f"  Corner value: {corner_val:.4f}")
        
        # Path loss increases with distance, so corner should have higher (more positive) value
        self.assertGreater(corner_val, center_val, "FSPL should increase with distance")
    
    def test_sparse_channel_no_sparse(self):
        """Test channel 8 when sparse_prob=0 (no sparse measurements)."""
        print("\n" + "="*60)
        print("[TEST] Sparse Channel - No Sparse (prob=0)")
        print("="*60)
        
        sample = create_mock_sample()
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        ch8 = output[8]
        
        # With sparse_prob=0, no sparse measurements should be added
        # Channel should be all zeros before normalization
        # After normalization: (0 - 87) / 160 = -0.54375
        expected_bg = (0.0 - NORM_OFFSET) / NORM_SCALE
        
        print(f"  Expected background value: {expected_bg:.5f}")
        print(f"  Actual unique values: {torch.unique(ch8).tolist()}")
        
        self.assertTrue(
            torch.allclose(ch8, torch.full_like(ch8, expected_bg), atol=1e-5),
            "Sparse channel should be all background when sparse_prob=0"
        )
    
    def test_sparse_channel_with_sparse(self):
        """Test channel 8 with sparse_prob=1.0 (always add sparse measurements)."""
        print("\n" + "="*60)
        print("[TEST] Sparse Channel - With Sparse (prob=1.0)")
        print("="*60)
        
        output_val = 80.0  # Ground truth pathloss value
        sample = create_mock_sample(output_value=output_val)
        
        # Force sparse measurements with exact sparsity
        sparse_range = [0.05, 0.05]  # Exactly 5% sparsity
        output = featurizer(sample, sparse_prob=1.0, sparse_range=sparse_range)
        
        ch8 = output[8]
        
        # Background value after normalization
        bg_val = (0.0 - NORM_OFFSET) / NORM_SCALE
        
        # Find non-background pixels
        is_meas = torch.abs(ch8 - bg_val) > 1e-5
        n_meas = is_meas.sum().item()
        total_valid = sample.H * sample.W  # All pixels valid in our mock
        
        expected_sparsity = 0.05
        actual_sparsity = n_meas / total_valid
        
        print(f"  Expected sparsity: {expected_sparsity*100:.1f}%")
        print(f"  Actual sparsity: {actual_sparsity*100:.2f}% ({n_meas}/{total_valid} pixels)")
        
        # Check sparsity is close to expected (allow some variance due to mask filtering)
        self.assertGreater(actual_sparsity, expected_sparsity * 0.5)
        self.assertLess(actual_sparsity, expected_sparsity * 2.0)
        
        if n_meas > 0:
            # Check that sparse values match ground truth
            sparse_vals_norm = ch8[is_meas]
            sparse_vals_denorm = sparse_vals_norm * NORM_SCALE + NORM_OFFSET
            
            print(f"  Sparse values (denormalized): min={sparse_vals_denorm.min():.2f}, max={sparse_vals_denorm.max():.2f}")
            print(f"  Expected value (output_val): {output_val:.2f}")
            
            # All sparse values should equal the ground truth
            self.assertTrue(
                torch.allclose(sparse_vals_denorm, torch.tensor(output_val), atol=0.1),
                "Sparse values should match ground truth"
            )
    
    def test_sparse_correspondence_exact(self):
        """Test that sparse measurements exactly match ground truth at sampled locations."""
        print("\n" + "="*60)
        print("[TEST] Sparse Correspondence - Exact Match")
        print("="*60)
        
        # Create sample with gradient output to test exact correspondence
        sample = create_mock_sample()
        # Replace output with a gradient pattern for more interesting test
        H, W = sample.H, sample.W
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        # Gradient from 50 to 100 dB
        sample.output_img = 50.0 + 50.0 * (xx + yy) / (H + W - 2)
        
        output = featurizer(sample, sparse_prob=1.0, sparse_range=[0.1, 0.1])
        
        ch8 = output[8]
        bg_val = (0.0 - NORM_OFFSET) / NORM_SCALE
        
        is_meas = torch.abs(ch8 - bg_val) > 1e-5
        
        if is_meas.sum().item() > 0:
            # Denormalize sparse values
            sparse_denorm = ch8 * NORM_SCALE + NORM_OFFSET
            
            # Get ground truth at measurement locations
            gt_at_meas = sample.output_img[is_meas]
            sp_at_meas = sparse_denorm[is_meas]
            
            diff = (sp_at_meas - gt_at_meas).abs()
            max_diff = diff.max().item()
            
            print(f"  Number of measurements: {is_meas.sum().item()}")
            print(f"  Max difference between sparse and GT: {max_diff:.6f}")
            
            self.assertLess(max_diff, 0.01, "Sparse values should exactly match GT")
        else:
            self.fail("No sparse measurements generated despite sparse_prob=1.0")
    
    def test_custom_approximation_function(self):
        """Test using a custom approximation function."""
        print("\n" + "="*60)
        print("[TEST] Custom Approximation Function")
        print("="*60)
        
        sample = create_mock_sample()
        
        # Custom function that returns constant value
        constant_val = 42.0
        def custom_approx(s):
            return torch.full((s.H, s.W), constant_val)
        
        output = featurizer(
            sample,
            approximation_feature_func=custom_approx,
            sparse_prob=0.0,
            sparse_range=[0.0, 0.01]
        )
        
        ch7 = output[7]
        
        # After normalization: (42 - 87) / 160 = -0.28125
        expected_norm = (constant_val - NORM_OFFSET) / NORM_SCALE
        
        print(f"  Expected normalized value: {expected_norm:.5f}")
        print(f"  Actual ch7 unique values: {torch.unique(ch7).tolist()}")
        
        self.assertTrue(
            torch.allclose(ch7, torch.full_like(ch7, expected_norm), atol=1e-5),
            "Custom approx function should be applied and normalized"
        )
    
    def test_full_mock_sample_visualization(self):
        """Generate visualization of featurizer output for visual inspection."""
        print("\n" + "="*60)
        print("[TEST] Full Mock Sample Visualization")
        print("="*60)
        
        sample = create_mock_sample()
        
        # Test with sparse measurements
        output = featurizer(sample, sparse_prob=1.0, sparse_range=[0.05, 0.05])
        save_featurizer_output_visualization(output, sample, self.output_dir, "mock_with_sparse")
        
        # Test without sparse measurements
        output_no_sparse = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        save_featurizer_output_visualization(output_no_sparse, sample, self.output_dir, "mock_no_sparse")
    
    def test_normalize_input_function(self):
        """Test the normalize_input function directly."""
        print("\n" + "="*60)
        print("[TEST] normalize_input Function")
        print("="*60)
        
        H, W = 50, 50
        input_tensor = torch.zeros((9, H, W), dtype=torch.float32)
        
        # Set known values for each channel
        input_tensor[0] = 255.0  # Reflectance max
        input_tensor[1] = 0.0    # Transmittance zero
        input_tensor[2] = 10.0   # Distance 10m
        input_tensor[3] = -10.0  # Antenna gain -10 dBi
        input_tensor[4] = 868.0  # Frequency 868 MHz
        input_tensor[5] = 1.0    # Mask
        input_tensor[6] = 1.0    # Floor plan
        input_tensor[7] = 87.0   # FSPL at normalization center
        input_tensor[8] = 87.0   # Sparse at normalization center
        
        normalized = normalize_input(input_tensor)
        
        print("  Channel normalization results:")
        for i in range(9):
            val = normalized[i, 0, 0].item()
            print(f"    Ch{i}: input={input_tensor[i, 0, 0].item():.2f} -> normalized={val:.4f}")
        
        # Check specific normalizations
        # Ch0 (reflectance): (255/255 - 0.485) / 0.229 ≈ 2.249
        expected_ch0 = (1.0 - 0.485) / 0.229
        self.assertAlmostEqual(normalized[0, 0, 0].item(), expected_ch0, places=2)
        
        # Ch7 (approx): (87 - 87) / 160 = 0
        self.assertAlmostEqual(normalized[7, 0, 0].item(), 0.0, places=4)
        
        # Ch8 (sparse): (87 - 87) / 160 = 0
        self.assertAlmostEqual(normalized[8, 0, 0].item(), 0.0, places=4)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_small_image(self):
        """Test with very small image size."""
        print("\n" + "="*60)
        print("[TEST] Small Image (10x10)")
        print("="*60)
        
        sample = create_mock_sample(H=10, W=10, x_ant=5.0, y_ant=5.0)
        # Adjust structure for small size
        sample.input_img[0] = 0  # Clear reflectance
        sample.input_img[0, 2:8, 2:8] = 1.0
        sample.input_img[1] = 0  # Clear transmittance
        sample.input_img[1, 3:7, 3:7] = 0.5
        
        output = featurizer(sample, sparse_prob=1.0, sparse_range=[0.1, 0.1])
        
        self.assertEqual(output.shape, (9, 10, 10))
        self.assertFalse(torch.isnan(output).any())
        print(f"  Output shape: {output.shape} ✓")
    
    def test_antenna_at_corner(self):
        """Test with antenna at image corner."""
        print("\n" + "="*60)
        print("[TEST] Antenna at Corner")
        print("="*60)
        
        sample = create_mock_sample(x_ant=0.0, y_ant=0.0)
        output = featurizer(sample, sparse_prob=0.0, sparse_range=[0.0, 0.01])
        
        self.assertEqual(output.shape, (9, 100, 100))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Distance at corner (antenna position) should be minimal
        ch2 = output[2]
        corner_dist = ch2[0, 0].item()
        far_corner_dist = ch2[99, 99].item()
        
        print(f"  Antenna corner distance: {corner_dist:.4f}")
        print(f"  Far corner distance: {far_corner_dist:.4f}")
        
        self.assertLess(corner_dist, far_corner_dist)
    
    def test_zero_sparsity_range(self):
        """Test with zero sparsity range."""
        print("\n" + "="*60)
        print("[TEST] Zero Sparsity Range")
        print("="*60)
        
        sample = create_mock_sample()
        output = featurizer(sample, sparse_prob=1.0, sparse_range=[0.0, 0.0])
        
        ch8 = output[8]
        bg_val = (0.0 - NORM_OFFSET) / NORM_SCALE
        
        # With sparsity=0%, should be all background
        self.assertTrue(
            torch.allclose(ch8, torch.full_like(ch8, bg_val), atol=1e-5),
            "Zero sparsity should result in no measurements"
        )
        print("  Zero sparsity correctly produces no measurements ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
