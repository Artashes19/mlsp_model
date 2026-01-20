"""
Test script to verify that TxUNetModel (devbugfix_khoren) and TxUNet (korean-model)
produce identical outputs when initialized with the same random seed.

The key insight: If both models are architecturally identical and initialized with
the same random seed, they will have the exact same weights. We can then pass the
same input through both and verify outputs match.

Steps:
1. On devbugfix_khoren: set seed=42 → create model → generate input → forward → save output
2. Checkout korean-model branch
3. On korean-model: set seed=42 → create model (gets same weights!) → same input → forward
4. Compare outputs - they should be identical if models are the same function
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def get_current_branch() -> str:
    """Get current git branch name."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def checkout_branch(branch: str) -> None:
    """Checkout to a git branch."""
    subprocess.run(["git", "checkout", branch], check=True)


def stash_changes() -> bool:
    """Stash uncommitted changes. Returns True if there were changes to stash."""
    result = subprocess.run(
        ["git", "stash", "push", "-m", "test_model_equivalence_temp"],
        capture_output=True,
        text=True,
    )
    return "No local changes to save" not in result.stdout


def pop_stash() -> None:
    """Pop stashed changes."""
    subprocess.run(["git", "stash", "pop"], check=True)


def clear_module_cache(module_prefixes: list[str]) -> None:
    """Remove modules from sys.modules that start with given prefixes."""
    to_remove = [k for k in sys.modules.keys() if any(k.startswith(p) for p in module_prefixes)]
    for k in to_remove:
        del sys.modules[k]


def test_model_equivalence():
    """Test that devbugfix_khoren and korean-model produce identical outputs."""
    
    # Configuration matching korean-model defaults (cfgs/model_txunet.yaml)
    # Both branches should use these exact parameters
    config = {
        "in_ch": 4,
        "out_ch": 1,
        "base_ch": 48,
        "depths": [4, 6, 6, 8],
        "heads": [4, 4, 8, 8],
        "expand": 2.66,
        "use_checkpoint": False,
        "ln_eps": 1e-5,
        "window0": None,
        "window0_stride": None,
        "sra0_enabled": False,
        "sra0_stride": 4,
    }
    
    # Fixed seeds for reproducibility
    model_seed = 42      # Seed for model weight initialization
    input_seed = 1234    # Seed for generating input tensor
    
    batch_size = 2
    height = 64
    width = 64
    
    # Get current branch (should be devbugfix_khoren)
    original_branch = get_current_branch()
    print(f"Current branch: {original_branch}")
    
    # Create temp directory for saving artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.pt"
        output_devbugfix_path = Path(tmpdir) / "output_devbugfix.pt"
        
        # ============ PHASE 1: devbugfix_khoren model ============
        print("\n" + "=" * 60)
        print("PHASE 1: Creating model from devbugfix_khoren branch")
        print("=" * 60)
        
        # Set seed BEFORE model creation for reproducible weights
        torch.manual_seed(model_seed)
        
        # Import and build model from devbugfix_khoren
        from src.networks import TxUNetModel
        
        model_devbugfix = TxUNetModel(
            in_ch=config["in_ch"],
            out_ch=config["out_ch"],
            base_ch=config["base_ch"],
            depths=tuple(config["depths"]),
            heads=tuple(config["heads"]),
            expand=float(config["expand"]),
            use_checkpoint=config["use_checkpoint"],
            ln_eps=float(config["ln_eps"]),
            window0=config["window0"],
            window0_stride=config["window0_stride"],
            sra0_enabled=config["sra0_enabled"],
            sra0_stride=config["sra0_stride"],
        )
        model_devbugfix.eval()
        
        # Generate random input with separate seed
        torch.manual_seed(input_seed)
        x = torch.randn(batch_size, config["in_ch"], height, width)
        torch.save(x, input_path)  # Save for later
        
        # Forward pass
        with torch.no_grad():
            output_devbugfix = model_devbugfix(x)
        
        torch.save(output_devbugfix, output_devbugfix_path)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output_devbugfix.shape}")
        print(f"Output mean: {output_devbugfix.mean().item():.6f}")
        print(f"Output std: {output_devbugfix.std().item():.6f}")
        
        num_params = sum(p.numel() for p in model_devbugfix.parameters())
        print(f"Number of parameters: {num_params:,}")
        
        # Clean up to avoid module conflicts when switching branches
        del model_devbugfix
        del TxUNetModel
        clear_module_cache(["src.networks", "src", "radio_unet_tx"])
        
        # ============ PHASE 2: korean-model ============
        print("\n" + "=" * 60)
        print("PHASE 2: Checking out korean-model branch")
        print("=" * 60)
        
        # Now stash changes so we can switch branches
        had_changes = stash_changes()
        if had_changes:
            print("Stashed local changes to allow branch switch")
        
        checkout_branch("korean-model")
        print("Switched to korean-model branch")
        
        # Set SAME seed before model creation - this gives identical weights!
        torch.manual_seed(model_seed)
        
        # Add models directory to path and import
        models_path = REPO_ROOT / "models"
        if str(models_path) not in sys.path:
            sys.path.insert(0, str(models_path))
        
        from radio_unet_tx import TxUNet
        
        # Note: korean-model has extra sra_strides param, set to None
        model_korean = TxUNet(
            in_ch=config["in_ch"],
            out_ch=config["out_ch"],
            base_ch=config["base_ch"],
            depths=tuple(config["depths"]),
            heads=tuple(config["heads"]),
            expand=float(config["expand"]),
            use_checkpoint=config["use_checkpoint"],
            ln_eps=float(config["ln_eps"]),
            window0=config["window0"],
            window0_stride=config["window0_stride"],
            sra_strides=None,  # korean-model specific param
            sra0_enabled=config["sra0_enabled"],
            sra0_stride=config["sra0_stride"],
        )
        model_korean.eval()
        
        # Load same input
        x = torch.load(input_path, weights_only=True)
        
        # Forward pass
        with torch.no_grad():
            output_korean = model_korean(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output_korean.shape}")
        print(f"Output mean: {output_korean.mean().item():.6f}")
        print(f"Output std: {output_korean.std().item():.6f}")
        
        # ============ PHASE 3: Compare outputs ============
        print("\n" + "=" * 60)
        print("PHASE 3: Comparing outputs")
        print("=" * 60)
        
        # Load devbugfix output
        output_devbugfix = torch.load(output_devbugfix_path, weights_only=True)
        
        # Compute differences
        abs_diff = (output_devbugfix - output_korean).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        # Relative difference (avoid div by zero)
        denom = output_devbugfix.abs().clamp(min=1e-8)
        rel_diff = abs_diff / denom
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"Max absolute difference: {max_abs_diff:.2e}")
        print(f"Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"Max relative difference: {max_rel_diff:.2e}")
        print(f"Mean relative difference: {mean_rel_diff:.2e}")
        
        # Tolerance for floating point comparison
        tolerance = 1e-5
        outputs_match = max_abs_diff < tolerance
        
        print("\n" + "=" * 60)
        if outputs_match:
            print("✓ SUCCESS: Models produce identical outputs!")
            print(f"  (max difference {max_abs_diff:.2e} < tolerance {tolerance:.2e})")
        else:
            print("✗ FAILURE: Models produce different outputs!")
            print(f"  (max difference {max_abs_diff:.2e} >= tolerance {tolerance:.2e})")
        print("=" * 60)
        
        # ============ PHASE 4: Restore original branch ============
        print(f"\nRestoring original branch: {original_branch}")
        checkout_branch(original_branch)
        print(f"Switched back to {original_branch}")
        
        if had_changes:
            pop_stash()
            print("Restored stashed changes")
        
        assert outputs_match, f"Outputs differ by more than {tolerance}"


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    test_model_equivalence()
