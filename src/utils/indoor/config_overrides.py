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
from typing import Optional
from dataclasses import dataclass, field
from typing import Literal
import yaml

# Environment variable name
OVERRIDES_ENV_VAR = "INDOOR_OVERRIDES_CONFIG"

# Default channels string (10 channels with one-hot frequency)
DEFAULT_CHANNELS = "rtdgfmps"


@dataclass
class OverridesConfig:
    """All configurable overrides for korean-model equivalence."""
    # Channels to use (default "rtdgfmps" = 10 channels)
    # r=reflectance (1ch), t=transmittance (1ch), d=distance (1ch), g=antenna gain (1ch),
    # f=frequency one-hot (3ch), m=mask (1ch), p=floor plan (1ch), s=sparse (1ch)
    channels: str = DEFAULT_CHANNELS
    
    # Normalization: legacy (fixed scaling) or standardize (mean/std)
    normalization_mode: Literal["legacy", "standardize"] = "legacy"
    # Stats for standardize mode
    # Expected keys: r/t/s -> mean_nz,std_nz ; d -> log_mean,log_std
    normalization_stats: dict = field(default_factory=dict)
    
    @property
    def num_channels(self) -> int:
        """Number of channels (computed from channels string, 'f' counts as 3 for one-hot)."""
        count = 0
        for ch in self.channels:
            if ch == "f":
                count += 3  # one-hot frequency encoding
            else:
                count += 1
        return count


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
        
        config = OverridesConfig(
            channels=channels,
            normalization_mode=overrides.get("normalization_mode", "legacy"),
            normalization_stats=overrides.get("normalization_stats", {}) or {},
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
