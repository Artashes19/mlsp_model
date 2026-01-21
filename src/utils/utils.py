import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

import fcntl
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import termios
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str],
    resolve: bool = True,
):
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    
    style = "green bold"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    
    rich.print(tree)
    
    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)
    
    OmegaConf.save(config, "config_tree.yaml")


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    algorithm: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """ This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally, saves:
        - number of trainable model parameters
    """
    if trainer.logger is None:
        return
    
    hparams = dict()
    
    # choose which parts of hydra config will be saved to loggers
    hparams["run_dir"] = os.getcwd()
    hparams["trainer"] = config["trainer"]
    hparams["algorithm"] = config["algorithm"]
    hparams["network"] = config["network"]
    hparams["optimizer"] = config["optimizer"]
    hparams["datamodule"] = config["datamodule"]
    hparams["ckpt_path"] = config["ckpt_path"]
    if "strategy" in config:
        hparams["strategy"] = config["strategy"]
    if "scheduler" in config:
        hparams["scheduler"] = config["scheduler"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    
    # save number of model parameters
    hparams["algorithm/params_total"] = sum(p.numel() for p in algorithm.parameters())
    hparams["algorithm/params_trainable"] = sum(
        p.numel() for p in algorithm.parameters() if p.requires_grad
    )
    hparams["algorithm/params_not_trainable"] = sum(
        p.numel() for p in algorithm.parameters() if not p.requires_grad
    )
    
    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
    
    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


class EpochCounter:
    count = 0


def set_winsize(fd, col, row, xpix=0, ypix=0):
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


class ProgressBarTheme(RichProgressBarTheme):
    # Hydra doesn't work with RichProgressBarTheme's Union type annotations
    description: str = "white"
    progress_bar: str = "#6206E0"
    progress_bar_finished: str = "#6206E0"
    progress_bar_pulse: str = "#6206E0"
    batch_progress: str = "white"
    time: str = "grey54"
    processing_speed: str = "grey70"
    metrics: str = "white"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"


@dataclass
class CompileParams:
    fullgraph: bool
    dynamic: bool
    backend: str
    mode: str
    options: dict[str, Any]
    disable: bool


def load_experiment_config(
    exp_cfg: DictConfig,
    config_root: Union[str, Path] = "configs",
) -> DictConfig:
    """
    Given a DictConfig for a single experiment (e.g. cfg.exps.e2),
    read its `defaults` list, load the referenced YAML files from
    `config_root`, and return a new DictConfig where those defaults
    are merged and `exp_cfg` overrides them.

    Notes:
    - Handles simple dict entries like {'datamodule': 'indoor'}
    - Handles list entries like {'callbacks': ['model_checkpoint_0', 'model_checkpoint_every']}
    - Ignores `_self_` (we merge `exp_cfg` itself at the end)
    """
    
    config_root = Path(config_root)
    merged = OmegaConf.create()
    
    defaults = exp_cfg.get("defaults", [])
    
    # Override main entries in defaults with those from exp_cfg
    for item in defaults:
        if item == "_self_":
            continue
        for group, value in item.items():
            if group in exp_cfg and isinstance(exp_cfg[group], str):
                item[group] = exp_cfg[group]
                exp_cfg.pop(group)
    
    for item in defaults:
        if item == "_self_":
            # We'll merge the experiment itself after defaults
            continue
        
        if not isinstance(item, (dict, DictConfig)):
            raise ValueError(f"Unsupported defaults entry: {item!r}")
        
        for group, value in item.items():
            # Disabled entry: e.g. {'scheduler': null}
            if value is None:
                continue
            
            # List-style group: e.g. {'callbacks': ['model_checkpoint_0', 'every']}
            if isinstance(value, (list, ListConfig)):
                for v in value:
                    cfg_path = config_root / group / f"{v}.yaml"
                    if not cfg_path.is_file():
                        raise FileNotFoundError(f"Config file not found: {cfg_path}")
                    piece = OmegaConf.load(cfg_path)
                    
                    # We assume callbacks end up like callbacks.<name> = piece
                    merged = OmegaConf.merge(merged, OmegaConf.create({group: {v: piece}}))
            
            else:
                # Simple group: e.g. {'datamodule': 'indoor'}
                cfg_path = config_root / group / f"{value}.yaml"
                if not cfg_path.is_file():
                    raise FileNotFoundError(f"Config file not found: {cfg_path}")
                piece = OmegaConf.load(cfg_path)
                
                # Typical Hydra behavior without explicit @package:
                # group name becomes the key: datamodule: <contents of indoor.yaml>
                merged = OmegaConf.merge(merged, OmegaConf.create({group: piece}))
    
    # Now merge the experiment itself on top of all defaults
    result = OmegaConf.merge(merged, exp_cfg)
    
    # Optionally drop the defaults key from the result
    if "defaults" in result:
        result.pop("defaults")
    
    return result
