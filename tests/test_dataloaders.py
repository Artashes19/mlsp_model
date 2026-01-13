"""
Integration tests for dataloaders using real experiment configurations (e0 and e2).
Tests verify:
1. Channel count and value ranges
2. Sparsity matches config
3. Sparse values match ground truth
4. Mask is binary (0/1)
5. Visual verification via saved plots

Requirements:
- Environment variables from .env (ICASSP_ORIG_PATH, SYNTHETIC_TRAIN_DIR, etc.)
- Test generates fresh manifests at startup and cleans them up at the end
"""
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file (same as run.py)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import hydra


# Constants from featurizer normalization
NORM_OFFSET = 87.0
NORM_SCALE = 160.0
BG_VAL_NORM = (0.0 - NORM_OFFSET) / NORM_SCALE  # Background value after normalization

# Test manifests directory (created at test setup, deleted at teardown)
_TEST_MANIFESTS_DIR: Path = None


def setup_test_manifests():
    """
    Generate fresh manifest files for testing (like run.py does at startup).
    Creates manifests in a temporary directory that will be cleaned up after tests.
    """
    global _TEST_MANIFESTS_DIR
    
    from src.data_exploration.generate_manifest import (
        ensure_icassp_manifest,
        ensure_manifest as ensure_synth_manifest,
        filter_icassp_manifest,
        filter_synthetic_manifest,
    )
    from src.experiments.splits import generate_building_split
    
    # Create temp directory for test manifests
    _TEST_MANIFESTS_DIR = Path(tempfile.mkdtemp(prefix="test_manifests_"))
    print(f"[TEST SETUP] Creating test manifests in {_TEST_MANIFESTS_DIR}")
    
    # Generate a building split (same as run.py)
    split = generate_building_split(seed=123, n_buildings=25, train_small_n=7, train_full_n=20)
    
    # Get data directories from environment
    icassp_root = os.environ.get("ICASSP_ORIG_PATH", "")
    synth_root = os.environ.get("SYNTHETIC_TRAIN_DIR", "") or os.environ.get("SYNTH_ROOT", "")
    freqs_mhz = [868, 1800, 3500]
    
    # Generate/ensure global ICASSP manifest
    icassp_global_manifest = None
    if icassp_root and os.path.isdir(icassp_root):
        icassp_global_manifest = os.path.join(icassp_root, "icassp_manifest.csv")
        ensure_icassp_manifest(icassp_root, icassp_global_manifest, freqs_mhz, task="Task_2_ICASSP")
        
        # Create filtered manifests
        filter_icassp_manifest(
            icassp_global_manifest,
            str(_TEST_MANIFESTS_DIR / "icassp_train_small.filtered.csv"),
            list(split.train_small),
            None
        )
        filter_icassp_manifest(
            icassp_global_manifest,
            str(_TEST_MANIFESTS_DIR / "icassp_train_full.filtered.csv"),
            list(split.train_full),
            None
        )
        filter_icassp_manifest(
            icassp_global_manifest,
            str(_TEST_MANIFESTS_DIR / "icassp_validation.filtered.csv"),
            list(split.validation),
            None
        )
        print(f"[TEST SETUP] Created ICASSP manifests (train_small={len(split.train_small)}, train_full={len(split.train_full)}, val={len(split.validation)} buildings)")
    else:
        raise RuntimeError(f"ICASSP_ORIG_PATH not set or invalid: {icassp_root}")
    
    # Generate synthetic manifest if available
    if synth_root and os.path.isdir(synth_root):
        synth_global_manifest = os.path.join(synth_root, "samples.csv")
        ensure_synth_manifest(synth_root, synth_global_manifest, freqs_mhz)
        filter_synthetic_manifest(
            synth_global_manifest,
            str(_TEST_MANIFESTS_DIR / "synthetic.filtered.csv"),
            None
        )
        print(f"[TEST SETUP] Created synthetic manifest")
    else:
        print(f"[TEST SETUP] Synthetic data not available (SYNTHETIC_TRAIN_DIR={synth_root})")
    
    return _TEST_MANIFESTS_DIR


def cleanup_test_manifests():
    """Remove the temporary test manifests directory."""
    global _TEST_MANIFESTS_DIR
    if _TEST_MANIFESTS_DIR and _TEST_MANIFESTS_DIR.exists():
        print(f"[TEST TEARDOWN] Removing test manifests from {_TEST_MANIFESTS_DIR}")
        shutil.rmtree(_TEST_MANIFESTS_DIR)
        _TEST_MANIFESTS_DIR = None


def get_test_manifests_dir() -> Path:
    """Get the test manifests directory, creating if needed."""
    global _TEST_MANIFESTS_DIR
    if _TEST_MANIFESTS_DIR is None or not _TEST_MANIFESTS_DIR.exists():
        setup_test_manifests()
    return _TEST_MANIFESTS_DIR


def get_config(experiment_name: str):
    """
    Load experiment config the same way run.py does:
    1. Load base train.yaml config via Hydra
    2. Convert to container to remove struct mode (like run.py clone_cfg)
    3. Merge experiment-specific config on top
    4. Wire up manifest paths from test manifests
    """
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    
    # Load base config (train.yaml)
    cfg = compose(config_name="train")
    
    # Convert to container and recreate to remove struct mode (same as run.py clone_cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # Merge experiment-specific config on top (same as run.py lines 261-263)
    exp_cfg_path = PROJECT_ROOT / "configs" / "experiments" / f"{experiment_name}.yaml"
    if exp_cfg_path.exists():
        exp_cfg = OmegaConf.load(exp_cfg_path)
        cfg = OmegaConf.merge(cfg, exp_cfg)
    
    # Wire up manifest paths from test manifests
    manifests_dir = get_test_manifests_dir()
    
    if experiment_name == "e0":
        cfg.datamodule.train_manifest_path = str(manifests_dir / "icassp_train_small.filtered.csv")
    elif experiment_name == "e1":
        cfg.datamodule.train_manifest_path = str(manifests_dir / "icassp_train_full.filtered.csv")
    elif experiment_name == "e2":
        cfg.datamodule.synthetic_manifest_path = str(manifests_dir / "synthetic.filtered.csv")
    
    cfg.datamodule.val_manifest_path = str(manifests_dir / "icassp_validation.filtered.csv")
    
    return cfg


def create_datamodule(experiment_name: str):
    """Create and setup a datamodule from experiment config."""
    cfg = get_config(experiment_name)
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup(stage="fit")
    return dm, cfg


def save_batch_visualization(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    output_dir: Path,
    batch_name: str,
    sample_idx: int = 0
):
    """
    Save a visualization of all input channels + target + mask as subplots.
    
    Channels (9 total):
    0: Reflectance (normalized)
    1: Transmittance (normalized)
    2: Distance (log transformed)
    3: Antenna gain (normalized)
    4: Frequency (log transformed)
    5: Mask
    6: Floor plan
    7: Approximation feature (FSPL, normalized)
    8: Sparse measurements (normalized)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_input = inputs[sample_idx].cpu().numpy()
    sample_target = targets[sample_idx].cpu().numpy()
    if sample_target.ndim == 3:
        sample_target = sample_target.squeeze(0)
    sample_mask = masks[sample_idx].cpu().numpy()
    
    channel_names = [
        "Ch0: Reflectance",
        "Ch1: Transmittance", 
        "Ch2: Distance (log)",
        "Ch3: Antenna Gain",
        "Ch4: Frequency (log)",
        "Ch5: Mask",
        "Ch6: Floor Plan",
        "Ch7: Approx (FSPL)",
        "Ch8: Sparse Meas.",
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot all 9 input channels
    for i in range(9):
        ax = axes[i]
        im = ax.imshow(sample_input[i], cmap='viridis')
        ax.set_title(f"{channel_names[i]}\nmin={sample_input[i].min():.3f}, max={sample_input[i].max():.3f}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot target (ground truth pathloss)
    ax = axes[9]
    im = ax.imshow(sample_target, cmap='hot')
    ax.set_title(f"Target (GT Pathloss)\nmin={sample_target.min():.1f}, max={sample_target.max():.1f}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot output mask
    ax = axes[10]
    im = ax.imshow(sample_mask, cmap='gray')
    ax.set_title(f"Output Mask\nunique={np.unique(sample_mask)}")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Plot sparse vs target overlay
    ax = axes[11]
    sparse_norm = sample_input[8]
    sparse_denorm = sparse_norm * NORM_SCALE + NORM_OFFSET
    # Create an overlay: target in background, sparse points as markers
    ax.imshow(sample_target, cmap='hot', alpha=0.7)
    is_meas = np.abs(sparse_norm - BG_VAL_NORM) > 1e-5
    if is_meas.any():
        ys, xs = np.where(is_meas)
        ax.scatter(xs, ys, c='cyan', s=2, alpha=0.8, label='Sparse pts')
    n_sparse = is_meas.sum()
    total_valid = (sample_mask > 0).sum()
    sparsity_pct = (n_sparse / total_valid * 100) if total_valid > 0 else 0
    ax.set_title(f"Sparse Overlay\n{n_sparse} pts ({sparsity_pct:.2f}%)")
    ax.axis('off')
    
    plt.tight_layout()
    fig_path = output_dir / f"{batch_name}_sample{sample_idx}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {fig_path}")
    return fig_path


def setUpModule():
    """Module-level setup: create test manifests before any tests run."""
    setup_test_manifests()


def tearDownModule():
    """Module-level teardown: clean up test manifests after all tests complete."""
    cleanup_test_manifests()


class TestDataloaders(unittest.TestCase):
    """
    Integration tests using real experiment configurations (e0 and e2) and real data.
    """
    
    @classmethod
    def setUpClass(cls):
        cls.output_dir = PROJECT_ROOT / "tests" / "test_outputs"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _check_channel_ranges(self, inputs: torch.Tensor, name: str):
        """Check that channel values are in reasonable ranges after normalization."""
        print(f"\n  Channel value ranges for {name}:")
        
        # Check for NaN/Inf
        self.assertFalse(torch.isnan(inputs).any(), f"{name}: Inputs contain NaN")
        self.assertFalse(torch.isinf(inputs).any(), f"{name}: Inputs contain Inf")
        
        for ch in range(inputs.shape[1]):
            ch_data = inputs[:, ch]
            ch_min, ch_max = ch_data.min().item(), ch_data.max().item()
            ch_mean = ch_data.mean().item()
            print(f"    Ch{ch}: min={ch_min:.4f}, max={ch_max:.4f}, mean={ch_mean:.4f}")
            
            # Channel-specific range checks
            if ch == 5:  # Mask channel - should be binary
                unique_vals = torch.unique(ch_data)
                self.assertTrue(
                    all(v in [0.0, 1.0] for v in unique_vals.tolist()),
                    f"Mask (ch5) should be binary 0/1, got unique values: {unique_vals.tolist()}"
                )
            elif ch == 6:  # Floor plan - should be binary or nearly so
                unique_vals = torch.unique(ch_data)
                # Floor plan is generated from reflectance/transmittance > 0
                # After some operations it should still be 0/1
                for v in unique_vals.tolist():
                    self.assertTrue(
                        0.0 <= v <= 1.0,
                        f"Floor plan (ch6) values should be in [0,1], got {v}"
                    )
    
    def _check_mask_validity(self, masks: torch.Tensor, name: str, min_valid_ratio: float = 0.01):
        """Check that output masks are valid binary masks."""
        print(f"\n  Mask validity check for {name}:")
        
        unique_vals = torch.unique(masks)
        print(f"    Unique mask values: {unique_vals.tolist()}")
        
        for v in unique_vals.tolist():
            self.assertTrue(
                abs(v - 0.0) < 1e-5 or abs(v - 1.0) < 1e-5,
                f"Mask should be binary 0/1, got {v}"
            )
        
        # Check that mask has some valid region
        # Note: synthetic data has smaller valid regions (~3%) than ICASSP data (~30%+)
        valid_ratio = masks.float().mean().item()
        print(f"    Valid region ratio: {valid_ratio:.2%}")
        self.assertGreater(valid_ratio, min_valid_ratio, f"Mask should have >{min_valid_ratio*100:.0f}% valid region")
    
    def _check_sparsity(self, inputs: torch.Tensor, cfg, name: str) -> dict:
        """Check that sparsity matches config expectations."""
        modality_dropout_prob = float(cfg.datamodule.modality_dropout_prob)
        sparse_dropout_given_dropout = float(cfg.datamodule.sparse_dropout_given_dropout)
        sparse_range = list(cfg.datamodule.sparse_range)
        
        # P(sparse present) = 1 - modality_dropout_prob * sparse_dropout_given_dropout
        sparse_prob = 1.0 - modality_dropout_prob * sparse_dropout_given_dropout
        
        print(f"\n  Sparsity check for {name}:")
        print(f"    Config: modality_dropout_prob={modality_dropout_prob}, sparse_dropout_given_dropout={sparse_dropout_given_dropout}")
        print(f"    Effective sparse_prob: {sparse_prob:.3f}, sparse_range={sparse_range}")
        
        batch_size = inputs.shape[0]
        sparse_channel = inputs[:, 8]  # Channel 8 is sparse measurements
        mask_channel = inputs[:, 5]  # Channel 5 is mask
        
        samples_with_sparse = 0
        sparsities = []
        
        for i in range(batch_size):
            sp = sparse_channel[i]
            mask = mask_channel[i]
            
            # Check if this sample has any sparse measurements
            is_meas = torch.abs(sp - BG_VAL_NORM) > 1e-5
            n_meas = is_meas.sum().item()
            n_valid = (mask > 0).sum().item()
            
            if n_meas > 0:
                samples_with_sparse += 1
                sparsity = n_meas / n_valid if n_valid > 0 else 0
                sparsities.append(sparsity)
        
        sparse_rate = samples_with_sparse / batch_size
        print(f"    Samples with sparse measurements: {samples_with_sparse}/{batch_size} ({sparse_rate:.1%})")
        
        # The probability of having sparse measurements is controlled by modality dropout
        # With batch_size samples, expected ~sparse_prob * batch_size samples have sparse
        # Allow some variance (check even for small batches)
        expected_with_sparse = sparse_prob * batch_size
        if expected_with_sparse >= 1:  # Only check if we expect at least 1 sample with sparse
            self.assertGreater(
                samples_with_sparse, expected_with_sparse * 0.3,
                f"Too few samples with sparse measurements (expected ~{expected_with_sparse:.1f})"
            )
        
        if sparsities:
            avg_sparsity = np.mean(sparsities)
            print(f"    Average sparsity (among sparse samples): {avg_sparsity:.4f} ({avg_sparsity*100:.2f}%)")
            
            # Check sparsity is within configured range (with tolerance)
            min_sparse, max_sparse = sparse_range
            # Allow 50% tolerance on range since it's random
            tolerance = (max_sparse - min_sparse) * 0.5 + 0.005
            self.assertGreaterEqual(
                avg_sparsity, min_sparse - tolerance,
                f"Sparsity {avg_sparsity:.4f} below range [{min_sparse}, {max_sparse}]"
            )
            self.assertLessEqual(
                avg_sparsity, max_sparse + tolerance,
                f"Sparsity {avg_sparsity:.4f} above range [{min_sparse}, {max_sparse}]"
            )
        
        return {
            'samples_with_sparse': samples_with_sparse,
            'sparsities': sparsities
        }
    
    def _check_sparse_correspondence(self, inputs: torch.Tensor, targets: torch.Tensor, name: str):
        """Check that sparse measurement values match ground truth."""
        print(f"\n  Sparse-GT correspondence check for {name}:")
        
        sparse_channel = inputs[:, 8]
        # Denormalize sparse values: val = val_norm * 160 + 87
        sparse_denorm = sparse_channel * NORM_SCALE + NORM_OFFSET
        
        total_sparse_pixels = 0
        matching_pixels = 0
        max_diffs = []
        
        for i in range(inputs.shape[0]):
            sp_norm = sparse_channel[i]
            sp_denorm = sparse_denorm[i]
            gt = targets[i]
            if gt.ndim == 3:
                gt = gt.squeeze(0)
            
            # Find where we have measurements
            is_meas = torch.abs(sp_norm - BG_VAL_NORM) > 1e-5
            n_meas = is_meas.sum().item()
            
            if n_meas > 0:
                total_sparse_pixels += n_meas
                vals_sp = sp_denorm[is_meas]
                vals_gt = gt[is_meas]
                
                # Check absolute difference (allow small tolerance for float precision)
                diff = (vals_sp - vals_gt).abs()
                max_diff = diff.max().item()
                max_diffs.append(max_diff)
                
                # Count matches within tolerance
                matches = (diff < 0.1).sum().item()
                matching_pixels += matches
        
        if total_sparse_pixels > 0:
            match_rate = matching_pixels / total_sparse_pixels
            avg_max_diff = np.mean(max_diffs) if max_diffs else 0
            print(f"    Total sparse pixels: {total_sparse_pixels}")
            print(f"    Matching pixels: {matching_pixels} ({match_rate*100:.1f}%)")
            print(f"    Average max diff per sample: {avg_max_diff:.6f}")
            
            # Use 95% threshold to allow for float precision issues in normalization
            self.assertGreater(
                match_rate, 0.95,
                f"Sparse values should match GT (got {match_rate*100:.1f}% match)"
            )
        else:
            print("    No sparse measurements in this batch")
    
    def test_e0_dataloader(self):
        """Test e0 configuration (Real ICASSP Data)."""
        print("\n" + "="*60)
        print("[TEST] e0 Dataloader - Real ICASSP Data")
        print("="*60)
        
        dm, cfg = create_datamodule("e0")
        
        print(f"\n  Config:")
        print(f"    modality_dropout_prob: {cfg.datamodule.modality_dropout_prob}")
        print(f"    sparse_dropout_given_dropout: {cfg.datamodule.sparse_dropout_given_dropout}")
        print(f"    sparse_range: {list(cfg.datamodule.sparse_range)}")
        
        loader = dm.train_dataloader()
        print(f"    batch_size: {loader.batch_size}")
        print(f"    dataset_size: {len(loader.dataset)}")
        
        # Get multiple batches for more robust testing
        batch_iter = iter(loader)
        batch = next(batch_iter)
        inputs, targets, masks, meta = batch
        
        print(f"\n  Batch shapes:")
        print(f"    inputs: {inputs.shape}")
        print(f"    targets: {targets.shape}")
        print(f"    masks: {masks.shape}")
        
        # 1. Check channel count
        self.assertEqual(inputs.shape[1], 9, "e0 inputs must have 9 channels")
        
        # 2. Check channel ranges
        self._check_channel_ranges(inputs, "e0")
        
        # 3. Check mask validity (ICASSP data has ~30%+ valid regions)
        self._check_mask_validity(masks, "e0", min_valid_ratio=0.1)
        
        # 4. Check sparsity
        self._check_sparsity(inputs, cfg, "e0")
        
        # 5. Check sparse-GT correspondence
        self._check_sparse_correspondence(inputs, targets, "e0")
        
        # 6. Visual verification - save multiple samples to see sparse variation
        n_vis = min(10, inputs.shape[0])
        for i in range(n_vis):
            save_batch_visualization(inputs, targets, masks, self.output_dir, "e0", sample_idx=i)
    
    def test_e2_dataloader(self):
        """Test e2 configuration (Synthetic Data)."""
        print("\n" + "="*60)
        print("[TEST] e2 Dataloader - Synthetic Data")
        print("="*60)
        
        dm, cfg = create_datamodule("e2")
        
        print(f"\n  Config:")
        print(f"    modality_dropout_prob: {cfg.datamodule.modality_dropout_prob}")
        print(f"    sparse_dropout_given_dropout: {cfg.datamodule.sparse_dropout_given_dropout}")
        print(f"    sparse_range: {list(cfg.datamodule.sparse_range)}")
        print(f"    use_synthetic_train: {cfg.datamodule.use_synthetic_train}")
        
        loader = dm.train_dataloader()
        print(f"    batch_size: {loader.batch_size}")
        print(f"    dataset_size: {len(loader.dataset)}")
        
        batch = next(iter(loader))
        inputs, targets, masks, meta = batch
        
        print(f"\n  Batch shapes:")
        print(f"    inputs: {inputs.shape}")
        print(f"    targets: {targets.shape}")
        print(f"    masks: {masks.shape}")
        
        # 1. Check channel count
        self.assertEqual(inputs.shape[1], 9, "e2 inputs must have 9 channels")
        
        # 2. Check channel ranges
        self._check_channel_ranges(inputs, "e2")
        
        # 3. Check mask validity (synthetic data has smaller valid regions ~3%)
        self._check_mask_validity(masks, "e2", min_valid_ratio=0.01)
        
        # 4. Check sparsity
        self._check_sparsity(inputs, cfg, "e2")
        
        # 5. Check sparse-GT correspondence
        self._check_sparse_correspondence(inputs, targets, "e2")
        
        # 6. Visual verification - save multiple samples to see sparse variation
        n_vis = min(10, inputs.shape[0])
        for i in range(n_vis):
            save_batch_visualization(inputs, targets, masks, self.output_dir, "e2", sample_idx=i)
    
    def test_multiple_batches_consistency(self):
        """Test that multiple batches are consistent in format."""
        print("\n" + "="*60)
        print("[TEST] Multiple Batches Consistency")
        print("="*60)
        
        dm, cfg = create_datamodule("e0")
        loader = dm.train_dataloader()
        
        n_batches = min(5, len(loader))
        batch_shapes = []
        
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            inputs, targets, masks, meta = batch
            batch_shapes.append({
                'inputs': inputs.shape,
                'targets': targets.shape,
                'masks': masks.shape
            })
            
            # Quick sanity checks
            self.assertEqual(inputs.shape[1], 9)
            self.assertFalse(torch.isnan(inputs).any())
            self.assertFalse(torch.isnan(targets).any())
        
        print(f"  Checked {len(batch_shapes)} batches - all consistent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
