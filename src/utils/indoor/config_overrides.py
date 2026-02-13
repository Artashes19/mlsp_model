"""
Centralized normalization configuration.

Usage:
    Set env var INDOOR_OVERRIDES_CONFIG to path of a YAML file, e.g.:
        export INDOOR_OVERRIDES_CONFIG=/path/to/normalization.yaml
    
    If not set, defaults are loaded from configs/indoor_normalization_unified.yaml.
"""
import os
from typing import Optional
from dataclasses import dataclass, field

import yaml

# Environment variable name
OVERRIDES_ENV_VAR = "INDOOR_OVERRIDES_CONFIG"

# Default normalization config path (unified)
DEFAULT_NORMALIZATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "configs",
    "indoor_normalization_unified.yaml",
)


@dataclass
class OverridesConfig:
    """Normalization configuration."""
    # Stats for standardize mode
    # Expected keys: r/t/s -> mean_nz,std_nz ; d -> log_mean,log_std
    normalization_stats: dict = field(default_factory=dict)


# Global singleton instance
_config: Optional[OverridesConfig] = None


def _load_config() -> OverridesConfig:
    """Load config from env var or unified defaults."""
    config_path = os.environ.get(OVERRIDES_ENV_VAR, DEFAULT_NORMALIZATION_PATH)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Normalization config not found: {config_path}")
    with open(config_path, "r") as f:
        overrides = yaml.safe_load(f) or {}
    
    normalization_stats = overrides.get("normalization_stats", {}) or {}
    config = OverridesConfig(
        normalization_stats=normalization_stats,
    )
    print(f"[config_overrides] Loaded from {config_path}: {config}")
    return config


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
