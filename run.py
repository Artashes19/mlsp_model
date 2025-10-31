import logging
import os
import random
import sys
import warnings

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

log = logging.getLogger(__name__)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

hydra.core.global_hydra.GlobalHydra.instance().clear()

# Ensure local src package is importable even if Hydra changes CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import utils
    from src.train import train
    from src.experiments.splits import (
        ensure_exp_dir,
        generate_building_split,
        read_split_json,
        write_split_json,
    )
    
    warnings.filterwarnings("ignore", ".*beta state*")
    
    terminal_col = config.get("terminal_col")
    if terminal_col:
        terminal_row = config.get("terminal_row", 24)
        utils.set_winsize(sys.stdin, terminal_col, terminal_row)
        utils.set_winsize(sys.stderr, terminal_col, terminal_row)
        utils.set_winsize(sys.stdout, terminal_col, terminal_row)
    
    if config.seed == -1:
        config.seed = random.randint(0, 10 ** 8)
    
    seed_everything(config.seed)
    log.info(f"Run dir: {os.path.realpath('./')}")
    
    # Performance: enable fast conv algo search and use fast matmul on Ampere+
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    if config.get("print_config"):
        utils.print_config(config, fields=tuple(config.keys()), resolve=True)
    
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    
    # Orchestrated multi-experiment run
    exps = str(config.get("exps", "") or "").strip()
    if exps:
        # Resolve or create exp_dir
        experiments_root = config.get("experiments_root") or "experiments"
        exp_dir = config.get("exp_dir") or None
        exp_dir = ensure_exp_dir(exp_dir, root_dir=experiments_root)

        # Ensure split.json exists (sizes from config.split if present)
        split = read_split_json(exp_dir)
        if split is None:
            split_cfg = config.get("split") or {}
            tsn = int(split_cfg.get("train_small_n", 7))
            tfn = int(split_cfg.get("train_full_n", 20))
            # validation_n is implied as n_buildings - train_full_n
            split = generate_building_split(seed=int(config.seed), n_buildings=25, train_small_n=tsn, train_full_n=tfn)
            write_split_json(exp_dir, split)

        from omegaconf import OmegaConf

        def clone_cfg(cfg: DictConfig) -> DictConfig:
            return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        e3_best_ckpt: str | None = None
        for name in [s.strip() for s in exps.split(',') if s.strip()]:
            cfg_e = clone_cfg(config)
            # Common overrides
            # datamodule
            cfg_e.datamodule.val_buildings_override = list(split.validation)
            # Logging and checkpoints
            if "loggers" in cfg_e and "aim" in cfg_e.loggers:
                cfg_e.loggers.aim.repo = os.path.join(exp_dir, "aim")
                cfg_e.loggers.aim.experiment = name
            if "callbacks" in cfg_e and "model_checkpoint_0" in cfg_e.callbacks:
                cfg_e.callbacks.model_checkpoint_0.dirpath = os.path.join(exp_dir, name, "checkpoints")
                # Expect metric name based on dataloader label
                cfg_e.callbacks.model_checkpoint_0.monitor = "real_val_mse"
            if "trainer" in cfg_e:
                cfg_e.trainer.default_root_dir = os.path.join(exp_dir, name, "pl")
            # One validation stream name
            cfg_e.datamodule.validation_names = ["real_val"]

            # Data dirs
            data_cfg = cfg_e.get("data", {})
            real_dir = data_cfg.get("real_dir") or cfg_e.datamodule.get("data_dir")
            synth_dir = data_cfg.get("synthetic_dir") or cfg_e.datamodule.get("synthetic_dir")
            if real_dir:
                cfg_e.datamodule.data_dir = real_dir
            if synth_dir:
                cfg_e.datamodule.synthetic_dir = synth_dir

            if name == "e1":
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_small)
                e3_best_ckpt = train(cfg_e)
            elif name == "e2":
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_full)
                e3_best_ckpt = train(cfg_e)
            elif name == "e3":
                cfg_e.datamodule.use_synthetic_train = True
                cfg_e.datamodule.train_buildings = None
                e3_best_ckpt = train(cfg_e)
            elif name == "e4":
                # Finetuning on train_small using e3 best checkpoint
                if not e3_best_ckpt:
                    # Try to find previous e3 best checkpoint under exp_dir
                    ckpt_dir = os.path.join(exp_dir, "e3", "checkpoints")
                    if os.path.isdir(ckpt_dir):
                        # pick the most recent checkpoint
                        try:
                            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            e3_best_ckpt = files[0] if files else None
                        except Exception:
                            e3_best_ckpt = None
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_small)
                # Finetune knobs: enable weights-only load
                if "algorithm" in cfg_e:
                    ft = cfg_e.algorithm.get("finetune", {}) or {}
                    ft["enable"] = True
                    ft["ckpt_path"] = e3_best_ckpt
                    cfg_e.algorithm.finetune = ft
                _ = train(cfg_e)
            else:
                log.warning(f"Unknown experiment '{name}' - skipping")
        return None

    # Single run, original behavior
    if config.name == "train":
        return train(config)


if __name__ == "__main__":
    main()
