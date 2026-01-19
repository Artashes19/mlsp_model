"""
Centralized configuration overrides for korean-model equivalence testing.

Usage:
    Set env var MLSP_OVERRIDES_CONFIG to path of a YAML file, e.g.:
        export MLSP_OVERRIDES_CONFIG=/path/to/korean_mode.yaml
    
    If not set, defaults are used (torchvision resize, 9 channels).

Example YAML config (korean_mode.yaml):
    resize_backend: pil
    num_channels: 3
"""
import os
from typing import Literal, Optional
from dataclasses import dataclass, field
import yaml

# Environment variable name
OVERRIDES_ENV_VAR = "MLSP_OVERRIDES_CONFIG"


@dataclass
class OverridesConfig:
    """All configurable overrides for korean-model equivalence."""
    # Resize backend: "torchvision" (default) or "pil" (korean-model compatible)
    resize_backend: Literal["torchvision", "pil"] = "torchvision"
    
    # Number of output channels from featurizer: 9 (default) or 3 (korean-model compatible)
    num_channels: int = 9


# Global singleton instance
_config: Optional[OverridesConfig] = None


def _load_config() -> OverridesConfig:
    """Load config from env var or return defaults."""
    config_path = os.environ.get(OVERRIDES_ENV_VAR)
    
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        
        config = OverridesConfig(
            resize_backend=overrides.get("resize_backend", "torchvision"),
            num_channels=int(overrides.get("num_channels", 9)),
        )
        print(f"[config_overrides] Loaded from {config_path}: {config}")
        return config
    
    return OverridesConfig()  # defaults


def get_config() -> OverridesConfig:
    """Get the global config (loads once on first call)."""
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def reset_config():
    """Reset config to reload from env var (useful for testing)."""
    global _config
    _config = None
