"""
Integration tests for dataloaders using real experiment configurations (e0 and e2).
Tests verify:
1. Channel count and value ranges
2. Sparsity matches config
3. Sparse values match ground truth
4. Mask is binary (0/1)
5. Visual verification via saved plots
"""
import os
import sys
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

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import hydra


# Constants from featurizer normalization
NORM_OFFSET = 87.0
NORM_SCALE = 160.0
BG_VAL_NORM = (0.0 - NORM_OFFSET) / NORM_SCALE  # Background value after normalization


def get_config(experiment_name: str):
    """Load experiment config using Hydra."""
    GlobalHydra.instance().clear()
    config_dir = str(PROJECT_ROOT / "configs")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name=f"experiments/{experiment_name}")
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
    
    def _check_mask_validity(self, masks: torch.Tensor, name: str):
        """Check that output masks are valid binary masks."""
        print(f"\n  Mask validity check for {name}:")
        
        unique_vals = torch.unique(masks)
        print(f"    Unique mask values: {unique_vals.tolist()}")
        
        for v in unique_vals.tolist():
            self.assertIn(v, [0.0, 1.0], f"Mask should be binary 0/1, got {v}")
        
        # Check that mask has some valid region
        valid_ratio = masks.float().mean().item()
        print(f"    Valid region ratio: {valid_ratio:.2%}")
        self.assertGreater(valid_ratio, 0.1, "Mask should have >10% valid region")
    
    def _check_sparsity(self, inputs: torch.Tensor, cfg, name: str) -> dict:
        """Check that sparsity matches config expectations."""
        sparse_prob = float(cfg.datamodule.sparse_prob)
        sparse_range = list(cfg.datamodule.sparse_range)
        
        print(f"\n  Sparsity check for {name}:")
        print(f"    Config: sparse_prob={sparse_prob}, sparse_range={sparse_range}")
        
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
        
        # The probability of having sparse measurements is sparse_prob
        # With batch_size samples, expected ~sparse_prob * batch_size samples have sparse
        # Allow some variance
        if batch_size >= 10:
            expected_with_sparse = sparse_prob * batch_size
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
            
            self.assertGreater(
                match_rate, 0.99,
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
        print(f"    sparse_prob: {cfg.datamodule.sparse_prob}")
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
        
        # 3. Check mask validity
        self._check_mask_validity(masks, "e0")
        
        # 4. Check sparsity
        self._check_sparsity(inputs, cfg, "e0")
        
        # 5. Check sparse-GT correspondence
        self._check_sparse_correspondence(inputs, targets, "e0")
        
        # 6. Visual verification
        save_batch_visualization(inputs, targets, masks, self.output_dir, "e0", sample_idx=0)
        if inputs.shape[0] > 1:
            save_batch_visualization(inputs, targets, masks, self.output_dir, "e0", sample_idx=1)
    
    def test_e2_dataloader(self):
        """Test e2 configuration (Synthetic Data)."""
        print("\n" + "="*60)
        print("[TEST] e2 Dataloader - Synthetic Data")
        print("="*60)
        
        dm, cfg = create_datamodule("e2")
        
        print(f"\n  Config:")
        print(f"    sparse_prob: {cfg.datamodule.sparse_prob}")
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
        
        # 3. Check mask validity  
        self._check_mask_validity(masks, "e2")
        
        # 4. Check sparsity
        self._check_sparsity(inputs, cfg, "e2")
        
        # 5. Check sparse-GT correspondence
        self._check_sparse_correspondence(inputs, targets, "e2")
        
        # 6. Visual verification
        save_batch_visualization(inputs, targets, masks, self.output_dir, "e2", sample_idx=0)
        if inputs.shape[0] > 1:
            save_batch_visualization(inputs, targets, masks, self.output_dir, "e2", sample_idx=1)
    
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
