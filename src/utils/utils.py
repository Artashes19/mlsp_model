import os
import struct
from dataclasses import dataclass
from typing import Any, Sequence

import fcntl
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import termios
from omegaconf import DictConfig, OmegaConf
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


