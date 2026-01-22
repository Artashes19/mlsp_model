"""Test modality dropout distribution matches config."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf
from src.utils.indoor.featurizer import featurizer
from src.utils.indoor.types import RadarSample

# Normalized values when input is zero
# Channels 0,1: simple /255 normalization
CH01_ZERO = 0.0 / 255.0  # reflectance & transmittance
# Channel 7: (x - 87) / 160
CH7_ZERO = (0.0 - 87.0) / 160.0  # sparse measurements


def create_dummy_sample(H=64, W=64):
    """Create a dummy RadarSample for testing."""
    input_img = torch.zeros((3, H, W), dtype=torch.float32)
    input_img[0] = torch.rand(H, W) * 10  # reflectance
    input_img[1] = torch.rand(H, W) * 10  # transmittance
    input_img[2] = torch.rand(H, W) * 100  # distance
    
    return RadarSample(
        file_name="test",
        H=H,
        W=W,
        x_ant=H // 2,
        y_ant=W // 2,
        azimuth=0.0,
        freq_MHz=868.0,
        input_img=input_img,
        output_img=torch.rand(H, W) * 150,  # pathloss
        radiation_pattern=torch.ones(360),
        pixel_size=0.25,
        mask=torch.ones(H, W),
    )


def count_modality_states(n_samples: int, modality_dropout_prob: float, sparse_dropout_given_dropout: float):
    """Count samples by modality state: both_present, sparse_off, transref_off."""
    both_present = 0
    sparse_off = 0
    transref_off = 0
    
    for _ in range(n_samples):
        sample = create_dummy_sample()
        
        tensor = featurizer(
            sample=sample,
            sparse_range=(0.01, 0.02),
            modality_dropout_prob=modality_dropout_prob,
            sparse_dropout_given_dropout=sparse_dropout_given_dropout,
        )
        
        # Check trans+ref (channels 0,1) - if all values equal normalized zero
        ch0_all_zero = ((tensor[0] - CH01_ZERO).abs() < 1e-4).all()
        ch1_all_zero = ((tensor[1] - CH01_ZERO).abs() < 1e-4).all()
        transref_dropped = ch0_all_zero and ch1_all_zero
        
        # Check sparse (channel 7) - if all values equal normalized zero
        ch7_all_zero = ((tensor[7] - CH7_ZERO).abs() < 1e-4).all()
        sparse_dropped = ch7_all_zero
        
        if transref_dropped and not sparse_dropped:
            transref_off += 1
        elif sparse_dropped and not transref_dropped:
            sparse_off += 1
        else:
            both_present += 1
    
    return both_present, sparse_off, transref_off


def test_modality_dropout(modality_dropout_prob: float, sparse_dropout_given_dropout: float, n_samples: int = 500):
    """Test that modality dropout distribution matches expected probabilities."""
    
    both, sparse_off, transref_off = count_modality_states(n_samples, modality_dropout_prob, sparse_dropout_given_dropout)
    total = both + sparse_off + transref_off
    
    # Expected probabilities
    p_both_expected = 1.0 - modality_dropout_prob
    p_sparse_off_expected = modality_dropout_prob * sparse_dropout_given_dropout
    p_transref_off_expected = modality_dropout_prob * (1.0 - sparse_dropout_given_dropout)
    
    # Observed probabilities
    p_both_obs = both / total
    p_sparse_off_obs = sparse_off / total
    p_transref_off_obs = transref_off / total
    
    print(f"\n=== Test: modality_dropout_prob={modality_dropout_prob}, sparse_dropout_given_dropout={sparse_dropout_given_dropout} ===")
    print(f"Samples: {total}")
    print(f"  Both present:    observed={p_both_obs:.3f}, expected={p_both_expected:.3f}")
    print(f"  Sparse off:      observed={p_sparse_off_obs:.3f}, expected={p_sparse_off_expected:.3f}")
    print(f"  Trans+ref off:   observed={p_transref_off_obs:.3f}, expected={p_transref_off_expected:.3f}")
    
    # Allow 10% tolerance for statistical variance
    tol = 0.10
    assert abs(p_both_obs - p_both_expected) < tol, f"both_present mismatch: {p_both_obs:.3f} vs {p_both_expected:.3f}"
    assert abs(p_sparse_off_obs - p_sparse_off_expected) < tol, f"sparse_off mismatch: {p_sparse_off_obs:.3f} vs {p_sparse_off_expected:.3f}"
    assert abs(p_transref_off_obs - p_transref_off_expected) < tol, f"transref_off mismatch: {p_transref_off_obs:.3f} vs {p_transref_off_expected:.3f}"
    
    print("PASSED!")
    return True


def test_default_config_values():
    """Verify default config values match expected defaults."""
    # Load base datamodule config
    dm_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "exps" / "datamodule" / "indoor.yaml")
    
    # Load e0 experiment config and merge
    exp_cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "exps" / "e0.yaml")
    if "datamodule" in exp_cfg.e0:
        dm_cfg = OmegaConf.merge(dm_cfg, exp_cfg.e0.datamodule)
    
    print("\n=== Test: Default config values ===")
    print(f"  modality_dropout_prob: {dm_cfg.modality_dropout_prob}")
    print(f"  sparse_dropout_given_dropout: {dm_cfg.sparse_dropout_given_dropout}")
    
    assert abs(dm_cfg.modality_dropout_prob - 0.6666) < 0.001, \
        f"Default modality_dropout_prob should be 0.6666, got {dm_cfg.modality_dropout_prob}"
    assert abs(dm_cfg.sparse_dropout_given_dropout - 0.5) < 0.001, \
        f"Default sparse_dropout_given_dropout should be 0.5, got {dm_cfg.sparse_dropout_given_dropout}"
    
    print("PASSED!")


if __name__ == "__main__":
    # Test 1: Verify default config values
    test_default_config_values()
    
    # Test 2: Default config (2/3 dropout, 50/50 split) - ~1/3 each
    test_modality_dropout(0.6666, 0.5, n_samples=600)
    
    # Test 3: No dropout (both always present)
    test_modality_dropout(0.0, 0.5, n_samples=300)
    
    # Test 4: Always dropout, always sparse off
    test_modality_dropout(1.0, 1.0, n_samples=300)
    
    # Test 5: Always dropout, always trans+ref off
    test_modality_dropout(1.0, 0.0, n_samples=300)
    
    # Test 6: 50% dropout, 80/20 split
    test_modality_dropout(0.5, 0.8, n_samples=500)
    
    print("\n=== ALL TESTS PASSED ===")
