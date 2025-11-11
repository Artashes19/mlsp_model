import logging
import os
import random
import sys
import warnings

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, ListConfig
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
        ensure_experiments_dir,
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
    raw_exps = config.get("exps", "")
    exp_list = []
    if isinstance(raw_exps, (list, tuple)) or isinstance(raw_exps, ListConfig):
        exp_list = [str(x).strip() for x in raw_exps if str(x).strip()]
    elif isinstance(raw_exps, str):
        exp_list = [s.strip() for s in raw_exps.split(',') if s.strip()]
    if exp_list:
        # Resolve experiments root and exp_dir with strict rules:
        # - If exp_dir IS PROVIDED: it must exist and contain split.json, else crash.
        # - If exp_dir IS NOT PROVIDED: create a NEW timestamped dir and create split.json there.
        experiments_root = config.get("experiments_root") or "experiments"
        root_abs = ensure_experiments_dir(experiments_root)
        exp_dir_opt = config.get("exp_dir")
        if exp_dir_opt:
            exp_dir = exp_dir_opt if os.path.isabs(exp_dir_opt) else os.path.join(root_abs, exp_dir_opt)
            if not os.path.isdir(exp_dir):
                raise RuntimeError(f"Experiment directory does not exist: {exp_dir}. Provide a valid exp_dir or omit it to create a new experiments set.")
            split = read_split_json(exp_dir)
            if split is None:
                raise RuntimeError(f"Missing split.json in {exp_dir}. This experiments set is invalid. Create a new experiments set or generate the split explicitly.")
        else:
            # Create a NEW experiments set dir and write a split
            exp_dir = ensure_exp_dir(None, root_dir=root_abs)
            split_cfg = config.get("split") or {}
            tsn = int(split_cfg.get("train_small_n", 7))
            tfn = int(split_cfg.get("train_full_n", 20))
            split = generate_building_split(seed=int(config.seed), n_buildings=25, train_small_n=tsn, train_full_n=tfn)
            write_split_json(exp_dir, split)

        from omegaconf import OmegaConf

        def clone_cfg(cfg: DictConfig) -> DictConfig:
            return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        e2_best_ckpt: str | None = None
        # Fast-dev toggle
        fast_dev = bool(config.get("fast_dev")) or bool(os.environ.get("FAST_DEV"))
        for name in exp_list:
            cfg_e = clone_cfg(config)
            # Common overrides
            # datamodule
            cfg_e.datamodule.val_buildings_override = list(split.validation)
            # Logging and checkpoints
            if "loggers" in cfg_e and "aim" in cfg_e.loggers:
                cfg_e.loggers.aim.repo = os.path.join(exp_dir, "aim")
                cfg_e.loggers.aim.experiment = name
            # Normalize checkpoint dirs for all ModelCheckpoint callbacks
            if "callbacks" in cfg_e:
                ckpt_dir = os.path.join(exp_dir, name, "checkpoints")
                for cb_name, cb_conf in list(cfg_e.callbacks.items()):
                    try:
                        tgt = str(cb_conf.get("_target_", ""))
                    except Exception:
                        tgt = ""
                    if "ModelCheckpoint" in tgt:
                        cfg_e.callbacks[cb_name].dirpath = ckpt_dir
                # Expect metric name based on dataloader label for the primary checkpoint
                if "model_checkpoint_0" in cfg_e.callbacks:
                    cfg_e.callbacks.model_checkpoint_0.monitor = "real_val_mse"
            if "trainer" in cfg_e:
                cfg_e.trainer.default_root_dir = os.path.join(exp_dir, name, "pl")
                if fast_dev:
                    cfg_e.trainer.max_epochs = 1
                    cfg_e.trainer.limit_train_batches = 4
                    cfg_e.trainer.limit_val_batches = 2
            # One validation stream name
            # Keep datamodule-configured names

            # Data dirs
            data_cfg = cfg_e.get("data", {})
            real_dir = data_cfg.get("real_dir") or cfg_e.datamodule.get("data_dir")
            synth_dir = data_cfg.get("synthetic_dir") or cfg_e.datamodule.get("synthetic_dir")
            if real_dir:
                cfg_e.datamodule.data_dir = real_dir
            if synth_dir:
                cfg_e.datamodule.synthetic_dir = synth_dir

            if name == "e0":
                # Train on train_small (ICASSR real only)
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_small)
                _ = train(cfg_e)
            elif name == "e1":
                # Train on train_full (ICASSR real only)
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_full)
                _ = train(cfg_e)
            elif name == "e2":
                # Pretrain on synthetic only; validation on real held-out buildings
                cfg_e.datamodule.use_synthetic_train = True
                cfg_e.datamodule.train_buildings = None
                _ = train(cfg_e)
                # Try to capture the best checkpoint path for downstream e3
                ckpt_dir = os.path.join(exp_dir, "e2", "checkpoints")
                e2_best_ckpt = None
                if os.path.isdir(ckpt_dir):
                    try:
                        files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        e2_best_ckpt = files[0] if files else None
                    except Exception:
                        e2_best_ckpt = None
            elif name == "e3":
                # Finetune on train_small using e2 checkpoints
                ckpt_dir = os.path.join(exp_dir, "e2", "checkpoints")
                # If not recorded from this run, try filesystem
                if not e2_best_ckpt:
                    if os.path.isdir(ckpt_dir):
                        try:
                            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            e2_best_ckpt = files[0] if files else None
                        except Exception:
                            e2_best_ckpt = None
                if not e2_best_ckpt:
                    cmd = f"timeout 60 python3 run.py exps=e2 exp_dir={os.path.basename(exp_dir)}"
                    raise RuntimeError(
                        f"e3 requires a pretrained checkpoint from e2.\n"
                        f"Cause: no .ckpt found under {ckpt_dir}\n"
                        f"Run this first:\n  {cmd}"
                    )
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = list(split.train_small)
                # Finetune knobs: enable weights-only load
                if "algorithm" in cfg_e:
                    ft = cfg_e.algorithm.get("finetune", {}) or {}
                    ft["enable"] = True
                    ft["ckpt_path"] = e2_best_ckpt
                    cfg_e.algorithm.finetune = ft
                _ = train(cfg_e)
            else:
                log.warning(f"Unknown experiment '{name}' - skipping")
        return None

    # Single run, original behavior
    if config.name == "train":
        # Strict single-run handling mirroring orchestrator:
        # - If exp_dir provided: require existing split.json.
        # - If not provided: create a NEW experiments set and create split.json.
        # - If finetune is enabled but no ckpt_path is provided, raise an instructive error.
        experiments_root = config.get("experiments_root") or "experiments"
        root_abs = ensure_experiments_dir(experiments_root)
        exp_dir_opt = config.get("exp_dir")
        if exp_dir_opt:
            exp_dir_single = exp_dir_opt if os.path.isabs(exp_dir_opt) else os.path.join(root_abs, exp_dir_opt)
            if not os.path.isdir(exp_dir_single):
                raise RuntimeError(f"Experiment directory does not exist: {exp_dir_single}. Provide a valid exp_dir or omit it to create a new experiments set.")
            split_single = read_split_json(exp_dir_single)
            if split_single is None:
                raise RuntimeError(f"Missing split.json in {exp_dir_single}. This experiments set is invalid. Create a new experiments set or generate the split explicitly.")
        else:
            exp_dir_single = ensure_exp_dir(None, root_dir=root_abs)
            split_cfg = config.get("split") or {}
            tsn = int(split_cfg.get("train_small_n", 7))
            tfn = int(split_cfg.get("train_full_n", 20))
            split_single = generate_building_split(
                seed=int(config.seed), n_buildings=25, train_small_n=tsn, train_full_n=tfn
            )
            write_split_json(exp_dir_single, split_single)
        # Enforce validation to be exactly the held-out 5 buildings
        try:
            config.datamodule.val_buildings_override = list(split_single.validation)
        except Exception:
            pass
        # Optional: choose training buildings by declared split role
        split_role = str(config.get("split_role", "")).lower()
        train_buildings = None
        if split_role in ("train_small", "small", "e0"):
            train_buildings = list(split_single.train_small)
        elif split_role in ("train_full", "full", "e1"):
            train_buildings = list(split_single.train_full)
        if train_buildings is not None:
            try:
                config.datamodule.train_buildings = train_buildings
            except Exception:
                pass
        # If requested finetune but no checkpoint, hint to run e2 under the same experiments set
        try:
            ft = config.algorithm.get("finetune", {}) or {}
            if bool(ft.get("enable", False)) and not ft.get("ckpt_path"):
                cmd = f"timeout 60 python3 run.py exps=e2 exp_dir={os.path.basename(exp_dir_single)}"
                raise RuntimeError(
                    f"Finetune is enabled but no ckpt_path was provided.\n"
                    f"Run the pretraining stage first:\n  {cmd}"
                )
        except Exception:
            # If algorithm/finetune not present, nothing to enforce
            pass
        return train(config)


if __name__ == "__main__":
    main()
