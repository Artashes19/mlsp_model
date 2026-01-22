"""
Centralized configuration overrides for korean-model equivalence testing.

Usage:
    Set env var INDOOR_OVERRIDES_CONFIG to path of a YAML file, e.g.:
        export INDOOR_OVERRIDES_CONFIG=/path/to/korean_mode.yaml
    
    If not set, defaults are used (torchvision resize, all 8 channels).

Example YAML config (korean_mode.yaml):
    channels: "rtd"
"""
import os
from dataclasses import dataclass
from typing import Optional

import yaml

# Environment variable name
OVERRIDES_ENV_VAR = "INDOOR_OVERRIDES_CONFIG"

# Default channels string (all 8 channels)
DEFAULT_CHANNELS = "rtdgfmps"


@dataclass
class OverridesConfig:
    """All configurable overrides for korean-model equivalence."""
    # Channels to use (default all 8: rtdgfmps)
    # r=reflectance, t=transmittance, d=distance, g=antenna gain, f=frequency,
    # m=mask, p=floor plan, s=sparse measurements
    channels: str = DEFAULT_CHANNELS
    
    @property
    def num_channels(self) -> int:
        """Number of channels (computed from channels string length)."""
        return len(self.channels)


# Global singleton instance
_config: Optional[OverridesConfig] = None


def _load_config() -> OverridesConfig:
    """Load config from env var or return defaults."""
    config_path = os.environ.get(OVERRIDES_ENV_VAR)
    
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        
        # Support both old num_channels and new channels format
        channels = overrides.get("channels", DEFAULT_CHANNELS)
        if "num_channels" in overrides and "channels" not in overrides:
            # Legacy support: convert num_channels to channels
            num_ch = int(overrides["num_channels"])
            if num_ch == 3:
                channels = "rtd"
            elif num_ch == 9:
                channels = DEFAULT_CHANNELS
            else:
                channels = DEFAULT_CHANNELS[:num_ch]
        
        config = OverridesConfig(channels=channels)
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
