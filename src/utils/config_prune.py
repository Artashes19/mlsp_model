import logging
import os
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)

PRUNABLE_GROUPS = [
    "trainer",
    "datamodule",
    "algorithm",
    "network",
    "optimizer",
    "scheduler",
    "strategy",
]


def prune_config(config: DictConfig) -> DictConfig:
    """Remove leaked inline keys from resolved config groups.

    When ``_self_`` is first in the defaults list, inline sections provide
    default values that config-group files override.  Keys present only in
    the inline section (from a different variant) leak into the resolved
    config.  This function removes those surplus keys.
    """
    choices = HydraConfig.get().runtime.choices
    config_dir = os.path.join(HydraConfig.get().runtime.cwd, "configs")

    with open_dict(config):
        for group in PRUNABLE_GROUPS:
            choice = choices.get(group)
            if not choice or group not in config:
                continue

            yaml_path = Path(config_dir) / group / f"{choice}.yaml"
            if not yaml_path.is_file():
                continue

            file_cfg = OmegaConf.load(yaml_path)
            if not isinstance(file_cfg, DictConfig):
                continue

            allowed_keys = set(file_cfg.keys())
            surplus = [k for k in config[group] if k not in allowed_keys]

            if surplus:
                log.info(
                    "prune_config: removing from '%s': %s", group, surplus
                )
                for key in surplus:
                    del config[group][key]

    return config
