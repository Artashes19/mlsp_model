"""
Centralized normalization configuration for MLSP.

Usage:
    Set env var INDOOR_OVERRIDES_CONFIG to path of a YAML file, e.g.:
        export INDOOR_OVERRIDES_CONFIG=/path/to/normalization.yaml
    
    If not set, defaults are used.
"""
import os
from dataclasses import dataclass
from typing import Optional

import yaml

# Environment variable name
OVERRIDES_ENV_VAR = "INDOOR_OVERRIDES_CONFIG"


@dataclass
class OverridesConfig:
    """Normalization configuration for MLSP."""
    pass


# Global singleton instance
_config: Optional[OverridesConfig] = None


def _load_config() -> OverridesConfig:
    """Load config from env var or return defaults."""
    config_path = os.environ.get(OVERRIDES_ENV_VAR)
    
    if config_path and os.path.isfile(config_path):
        config = OverridesConfig()
        print(f"[config_overrides] Loaded from {config_path}: {config}")
        return config
    
    return OverridesConfig()


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
