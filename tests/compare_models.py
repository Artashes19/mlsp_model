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


def _run_cmd(cmd, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _ensure_korean_worktree(repo_root: Path, worktree_path: Path, branch: str = "korean-model") -> None:
    output = _run_cmd(["git", "worktree", "list", "--porcelain"], cwd=repo_root)
    worktree_paths = [
        line.split(" ", 1)[1] for line in output.splitlines() if line.startswith("worktree ")
    ]
    if str(worktree_path) in worktree_paths:
        return
    if worktree_path.exists():
        git_marker = worktree_path / ".git"
        if not git_marker.exists():
            raise RuntimeError(f"Worktree path exists and is not a git worktree: {worktree_path}")
    else:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _run_cmd(["git", "worktree", "add", str(worktree_path), branch], cwd=repo_root)


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
        print("PHASE 2: Running korean-model in worktree")
        print("=" * 60)
        worktree_path = Path("/tmp/branch_comparison/korean_model_worktree")
        _ensure_korean_worktree(REPO_ROOT, worktree_path)
        output_korean_path = Path(tmpdir) / "output_korean.pt"
        inline = f"""
import sys
import torch
from pathlib import Path

model_seed = {model_seed}
input_path = Path(r\"{str(input_path)}\")
output_path = Path(r\"{str(output_korean_path)}\")

repo_root = Path.cwd()
models_path = repo_root / "models"
if str(models_path) not in sys.path:
    sys.path.insert(0, str(models_path))

from radio_unet_tx import TxUNet

torch.manual_seed(model_seed)
model_korean = TxUNet(
    in_ch={config['in_ch']},
    out_ch={config['out_ch']},
    base_ch={config['base_ch']},
    depths={tuple(config['depths'])},
    heads={tuple(config['heads'])},
    expand={float(config['expand'])},
    use_checkpoint={config['use_checkpoint']},
    ln_eps={float(config['ln_eps'])},
    window0={config['window0']},
    window0_stride={config['window0_stride']},
    sra_strides=None,
    sra0_enabled={config['sra0_enabled']},
    sra0_stride={config['sra0_stride']},
)
model_korean.eval()

x = torch.load(input_path, weights_only=True)
with torch.no_grad():
    output_korean = model_korean(x)

torch.save(output_korean, output_path)
print("Input shape:", x.shape)
print("Output shape:", output_korean.shape)
print("Output mean:", output_korean.mean().item())
print("Output std:", output_korean.std().item())
"""
        result = subprocess.run([sys.executable, "-c", inline], cwd=str(worktree_path))
        if result.returncode != 0:
            raise RuntimeError("Failed to run korean-model inference in worktree.")
        output_korean = torch.load(output_korean_path, weights_only=True)
        
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
        
        assert outputs_match, f"Outputs differ by more than {tolerance}"


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    test_model_equivalence()