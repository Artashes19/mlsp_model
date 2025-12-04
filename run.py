import logging
import os
import csv
import random
import sys
import warnings
import time

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


def _setup_hydra_run_dir():
    """
    Pre-process command line args BEFORE Hydra runs.
    If exp_dir is provided and exists, set hydra.run.dir to it so Hydra
    doesn't create a new timestamped directory.
    """
    exp_dir = None
    experiments_root = "experiments"  # default
    
    # Parse relevant args from command line
    for arg in sys.argv[1:]:
        if arg.startswith("exp_dir="):
            exp_dir = arg.split("=", 1)[1]
        elif arg.startswith("experiments_root="):
            experiments_root = arg.split("=", 1)[1]
        # Check if hydra.run.dir is already overridden
        elif arg.startswith("hydra.run.dir=") or arg.startswith("+hydra.run.dir="):
            return  # User already specified, don't override
    
    if not exp_dir:
        return  # No exp_dir provided, let Hydra create new dir
    
    # Resolve exp_dir to absolute path
    if os.path.isabs(exp_dir):
        exp_dir_abs = exp_dir
    else:
        # Relative to experiments_root
        exp_dir_abs = os.path.join(os.getcwd(), experiments_root, exp_dir)
    
    # Only override if the directory exists (resuming existing run)
    if os.path.isdir(exp_dir_abs):
        sys.argv.append(f"hydra.run.dir={exp_dir_abs}")


_setup_hydra_run_dir()


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import utils
    from src.train import train
    from src.experiments.splits import (
        ensure_experiments_dir,
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
        # - If exp_dir IS PROVIDED:
        #     * If absolute: use it directly (do NOT create a new experiments root).
        #     * If relative: resolve it under experiments_root, creating experiments_root if needed.
        # - If exp_dir IS NOT PROVIDED: create a NEW timestamped dir and create split.json there.
        experiments_root = config.get("experiments_root") or "experiments"
        log.info(f"[orchestrator] experiments_root={experiments_root}")
        exp_dir_opt = config.get("exp_dir")
        if exp_dir_opt:
            if os.path.isabs(exp_dir_opt):
                exp_dir = exp_dir_opt
            else:
                root_abs = ensure_experiments_dir(experiments_root)
                exp_dir = os.path.join(root_abs, exp_dir_opt)
            if not os.path.isdir(exp_dir):
                raise RuntimeError(f"Experiment directory does not exist: {exp_dir}. Provide a valid exp_dir or omit it to create a new experiments set.")
            split = read_split_json(exp_dir)
            if split is None:
                raise RuntimeError(f"Missing split.json in {exp_dir}. This experiments set is invalid. Create a new experiments set or generate the split explicitly.")
            else:
                log.info(
                    f"[split] loaded split.json from {exp_dir} "
                    f"(seed={split.seed}, train_small={len(split.train_small)}, train_full={len(split.train_full)}, "
                    f"validation={len(split.validation)})"
                )
        else:
            # Create a NEW experiments set dir and write a split
            # Use experiments/ directly (no nested timestamp since Hydra run dir already has one)
            exp_dir = ensure_experiments_dir(experiments_root)
            split_cfg = config.get("split") or {}
            tsn = int(split_cfg.get("train_small_n", 7))
            tfn = int(split_cfg.get("train_full_n", 20))
            split = generate_building_split(seed=int(config.seed), n_buildings=25, train_small_n=tsn, train_full_n=tfn)
            write_split_json(exp_dir, split)
            log.info(
                f"[split] created new experiments set at {exp_dir} "
                f"(seed={split.seed}, train_small={len(split.train_small)}, train_full={len(split.train_full)}, "
                f"validation={len(split.validation)})"
            )

        from omegaconf import OmegaConf

        def clone_cfg(cfg: DictConfig) -> DictConfig:
            return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # Prepare global manifests and per-experiments-set filtered manifests
        from src.data_exploration.generate_manifest import (
            ensure_icassp_manifest,
            ensure_manifest as ensure_synth_manifest,
            filter_icassp_manifest,
            filter_synthetic_manifest,
        )
        run_dir_abs = os.path.realpath("./")
        manifests_dir = os.path.join(run_dir_abs, "manifests")
        os.makedirs(manifests_dir, exist_ok=True)
        # Enforce required ICASSP root early (fail fast)
        icassp_root = os.path.expanduser(str(config.datamodule.get("data_dir", "")))
        if not icassp_root or not os.path.isdir(icassp_root):
            raise RuntimeError(f"datamodule.data_dir must point to an existing ICASSP root. Got: {config.datamodule.get('data_dir')}")
        # Global ICASSP manifest under ICASSP root
        icassp_global_manifest = None
        if icassp_root and os.path.isdir(icassp_root):
            icassp_global_manifest = os.path.join(icassp_root, "icassp_manifest.csv")
            t0 = time.perf_counter()
            _ = ensure_icassp_manifest(icassp_root, icassp_global_manifest, config.datamodule.get("freqs_mhz", []), task="Task_2_ICASSP")
            dt = time.perf_counter() - t0
            rows = 0
            try:
                with open(icassp_global_manifest, "r", newline="") as fp:
                    rows = sum(1 for _ in fp) - 1  # minus header
            except Exception:
                rows = -1
            log.info(f"[manifest] ensured ICASSP manifest at {icassp_global_manifest} "
                     f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)")
        # Global synthetic manifest under SYNTH root
        synth_root = os.path.expanduser(str(config.datamodule.get("synthetic_dir", "")))
        synth_global_manifest = None
        if synth_root and os.path.isdir(synth_root):
            synth_global_manifest = os.path.join(synth_root, "samples.csv")
            t0 = time.perf_counter()
            _ = ensure_synth_manifest(synth_root, synth_global_manifest, config.datamodule.get("freqs_mhz", []))
            dt = time.perf_counter() - t0
            rows = 0
            try:
                with open(synth_global_manifest, "r", newline="") as fp:
                    rows = sum(1 for _ in fp) - 1
            except Exception:
                rows = -1
            log.info(f"[manifest] ensured synthetic manifest at {synth_global_manifest} "
                     f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)")
        # Build per-experiment-set filtered manifests
        # Create ICASSP train_small and train_full manifests if possible
        icassp_limit = int(config.datamodule.get("icassp_limit_per_building", 0) or 0)
        synth_limit = int(config.datamodule.get("synthetic_limit", 0) or 0)
        icassp_small_manifest = None
        icassp_full_manifest = None
        icassp_val_manifest = None
        synth_filtered_manifest = None
        if icassp_global_manifest and os.path.exists(icassp_global_manifest):
            icassp_small_manifest = os.path.join(manifests_dir, "icassp_train_small.filtered.csv")
            icassp_full_manifest = os.path.join(manifests_dir, "icassp_train_full.filtered.csv")
            icassp_val_manifest = os.path.join(manifests_dir, "icassp_validation.filtered.csv")
            t0 = t1 = time.perf_counter()
            _ = filter_icassp_manifest(icassp_global_manifest, icassp_small_manifest, list(split.train_small), icassp_limit if icassp_limit > 0 else None)
            t1 = time.perf_counter()
            _ = filter_icassp_manifest(icassp_global_manifest, icassp_full_manifest, list(split.train_full), icassp_limit if icassp_limit > 0 else None)
            t2 = time.perf_counter()
            _ = filter_icassp_manifest(icassp_global_manifest, icassp_val_manifest, list(split.validation), icassp_limit if icassp_limit > 0 else None)
            t3 = time.perf_counter()
            def _count_rows(p):
                try:
                    with open(p, "r", newline="") as fp:
                        return max(0, sum(1 for _ in fp) - 1)
                except Exception:
                    return -1
            rows_small = _count_rows(icassp_small_manifest)
            rows_full = _count_rows(icassp_full_manifest)
            rows_val = _count_rows(icassp_val_manifest)
            log.info(f"[manifest] ICASSP filtered (small={len(split.train_small)} blds, limit_per_bld={icassp_limit or 'none'}) "
                     f"-> {icassp_small_manifest} (rows={rows_small}, took={(t1 - t0):.2f}s)")
            log.info(f"[manifest] ICASSP filtered (full={len(split.train_full)} blds, limit_per_bld={icassp_limit or 'none'}) "
                     f"-> {icassp_full_manifest} (rows={rows_full}, took={(t2 - t1):.2f}s)")
            log.info(f"[manifest] ICASSP filtered (validation={len(split.validation)} blds, limit_per_bld={icassp_limit or 'none'}) "
                     f"-> {icassp_val_manifest} (rows={rows_val}, took={(t3 - t2):.2f}s)")
        if synth_global_manifest and os.path.exists(synth_global_manifest):
            synth_filtered_manifest = os.path.join(manifests_dir, "synthetic.filtered.csv")
            t0 = time.perf_counter()
            _ = filter_synthetic_manifest(synth_global_manifest, synth_filtered_manifest, synth_limit if synth_limit > 0 else None)
            dt = time.perf_counter() - t0
            rows = 0
            try:
                with open(synth_filtered_manifest, "r", newline="") as fp:
                    rows = sum(1 for _ in fp) - 1
            except Exception:
                rows = -1
            log.info(f"[manifest] Synthetic filtered (limit={synth_limit or 'none'}) -> {synth_filtered_manifest} "
                     f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)")

        e2_best_ckpt: str | None = None
        # Fast-dev toggle
        fast_dev = bool(config.get("fast_dev")) or bool(os.environ.get("FAST_DEV"))
        if fast_dev:
            log.info("[orchestrator] fast_dev enabled: will cap epochs/batches for quick smoke run")
        for name in exp_list:
            cfg_e = clone_cfg(config)
            # Merge experiment-specific config (required for trainer and any per-exp overrides)
            exp_cfg_dir_opt = config.get("experiments_config_dir") or "configs/experiments"
            repo_root = os.path.dirname(os.path.abspath(__file__))
            exp_cfg_dir = exp_cfg_dir_opt if os.path.isabs(exp_cfg_dir_opt) else os.path.join(repo_root, exp_cfg_dir_opt)
            exp_cfg_path = os.path.join(exp_cfg_dir, f"{name}.yaml")
            if os.path.isfile(exp_cfg_path):
                exp_cfg = OmegaConf.load(exp_cfg_path)
                cfg_e = OmegaConf.merge(cfg_e, exp_cfg)
            else:
                # If no experiment config file, require trainer to be provided explicitly
                if "trainer" not in cfg_e or not cfg_e.get("trainer"):
                    raise RuntimeError(
                        f"Missing experiment config for '{name}' at {exp_cfg_path} and no trainer provided.\n"
                        f"Provide a per-experiment config file or pass trainer via CLI, e.g.:\n"
                        f"  python run.py exps={name} trainer.max_epochs=2 trainer.devices=[0]"
                    )
            # Common overrides
            # datamodule: resolve training buildings from split by role key if provided
            val_role = None
            train_role = None
            if "datamodule" in cfg_e:
                val_role = cfg_e.datamodule.get("icassp_val_buildings", None)
                train_role = cfg_e.datamodule.get("icassp_train_buildings", None)
            if val_role is None:
                val_role = cfg_e.get("icassp_val_buildings", None)
            if train_role is None:
                train_role = cfg_e.get("icassp_train_buildings", None)
            # Training buildings (only for real-data training)
            train_from_role: list[int] | None = None
            if train_role:
                grp = getattr(split, str(train_role), None)
                if grp is None:
                    raise RuntimeError(f"Unknown split key '{train_role}' requested for training. Expected one of: train_small, train_full, validation")
                train_from_role = list(grp)
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
                    if "ModelCheckpoint" in tgt or "ScheduledEpochModelCheckpoint" in tgt:
                        cfg_e.callbacks[cb_name].dirpath = ckpt_dir
            if "trainer" in cfg_e:
                cfg_e.trainer.default_root_dir = os.path.join(exp_dir, name, "pl")
                try:
                    log.info(f"[trainer@{name}] devices={cfg_e.trainer.devices}, accelerator={cfg_e.trainer.accelerator}, "
                             f"precision={cfg_e.trainer.precision}, max_epochs={cfg_e.trainer.max_epochs}")
                except Exception:
                    pass
                # Re-apply CLI-provided trainer overrides (e.g., max_epochs) after merging experiment config
                try:
                    if "trainer" in config and config.trainer is not None:
                        if config.trainer.get("max_epochs", None) is not None:
                            cfg_e.trainer.max_epochs = int(config.trainer.max_epochs)
                except Exception:
                    pass
                if fast_dev:
                    cfg_e.trainer.max_epochs = 1
                    cfg_e.trainer.limit_train_batches = 4
                    cfg_e.trainer.limit_val_batches = 2
            # Enforce required data roots (no fallbacks)
            data_dir_req = cfg_e.datamodule.get("data_dir")
            data_dir_req = os.path.expanduser(str(data_dir_req)) if data_dir_req is not None else ""
            if not data_dir_req or not os.path.isdir(data_dir_req):
                raise RuntimeError(
                    f"datamodule.data_dir must point to an existing ICASSP root. Got: {cfg_e.datamodule.get('data_dir')}"
                )
            if name == "e2":
                synth_dir_req = cfg_e.datamodule.get("synthetic_dir")
                synth_dir_req = os.path.expanduser(str(synth_dir_req)) if synth_dir_req is not None else ""
                if not synth_dir_req or not os.path.isdir(synth_dir_req):
                    raise RuntimeError(
                        f"datamodule.synthetic_dir must point to an existing synthetic root for e2. Got: {cfg_e.datamodule.get('synthetic_dir')}"
                    )
            # Wire per-experiment manifests when available
            # New explicit train/val manifests: no runtime splitting
            if name in ("e0", "e3"):
                if icassp_small_manifest:
                    cfg_e.datamodule.train_manifest_path = icassp_small_manifest
                if icassp_val_manifest:
                    cfg_e.datamodule.val_manifest_path = icassp_val_manifest
            elif name == "e1":
                if icassp_full_manifest:
                    cfg_e.datamodule.train_manifest_path = icassp_full_manifest
                if icassp_val_manifest:
                    cfg_e.datamodule.val_manifest_path = icassp_val_manifest
            elif name == "e2":
                if synth_filtered_manifest:
                    cfg_e.datamodule.synthetic_manifest_path = synth_filtered_manifest
                if icassp_val_manifest:
                    cfg_e.datamodule.val_manifest_path = icassp_val_manifest

            # Summarize datamodule plan for this experiment
            try:
                dm = cfg_e.datamodule
                plan = dict(
                    use_synthetic_train=bool(dm.get("use_synthetic_train", False)),
                    train_buildings=("len=" + str(len(dm.get("train_buildings", []))) if dm.get("train_buildings") else "None"),
                    train_manifest_path=dm.get("train_manifest_path", None),
                    val_manifest_path=dm.get("val_manifest_path", None),
                    synthetic_manifest_path=dm.get("synthetic_manifest_path", None),
                    data_dir=dm.get("data_dir", None),
                    synthetic_dir=dm.get("synthetic_dir", None),
                )
                log.info(f"[datamodule@{name}] plan={plan}")
            except Exception:
                pass

            if name == "e0":
                # Train on train_small (ICASSR real only)
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = train_from_role if 'train_from_role' in locals() and train_from_role else list(split.train_small)
                t0 = time.perf_counter()
                best = train(cfg_e)
                log.info(f"[train@{name}] finished in {(time.perf_counter()-t0):.2f}s; best_checkpoint={best}")
            elif name == "e1":
                # Train on train_full (ICASSR real only)
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = train_from_role if 'train_from_role' in locals() and train_from_role else list(split.train_full)
                t0 = time.perf_counter()
                best = train(cfg_e)
                log.info(f"[train@{name}] finished in {(time.perf_counter()-t0):.2f}s; best_checkpoint={best}")
            elif name == "e2":
                # Pretrain on synthetic only; validation on real held-out buildings
                cfg_e.datamodule.use_synthetic_train = True
                cfg_e.datamodule.train_buildings = None
                t0 = time.perf_counter()
                best = train(cfg_e)
                log.info(f"[train@{name}] finished in {(time.perf_counter()-t0):.2f}s; best_checkpoint={best}")
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
                log.info(f"[e2] selected checkpoint for finetune: {e2_best_ckpt if e2_best_ckpt else 'NONE FOUND'} (dir={ckpt_dir})")
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
                else:
                    log.info(f"[e3] using e2 checkpoint: {os.path.abspath(e2_best_ckpt)}")
                cfg_e.datamodule.use_synthetic_train = False
                cfg_e.datamodule.train_buildings = train_from_role if 'train_from_role' in locals() and train_from_role else list(split.train_small)
                # Finetune knobs: enable weights-only load
                if "algorithm" in cfg_e:
                    ft = cfg_e.algorithm.get("finetune", {}) or {}
                    ft["enable"] = True
                    ft["ckpt_path"] = os.path.abspath(e2_best_ckpt)
                    cfg_e.algorithm.finetune = ft
                t0 = time.perf_counter()
                best = train(cfg_e)
                log.info(f"[train@{name}] finished in {(time.perf_counter()-t0):.2f}s; best_checkpoint={best}")
            else:
                log.warning(f"Unknown experiment '{name}' - skipping")
        return None

    # Single run, original behavior
    if config.name == "train":
        # Strict single-run handling mirroring orchestrator:
        # - If exp_dir provided: require existing split.json.
        # - If not provided: create a NEW experiments set and create split.json.
        # - If finetune is enabled but no ckpt_path is provided, raise an instructive error.
        # Additionally: require an explicit trainer configuration (no global default).
        if "trainer" not in config or not config.get("trainer"):
            raise RuntimeError(
                "No trainer configuration was provided. This project requires an explicit trainer per run.\n"
                "Pass it via CLI (e.g., trainer.max_epochs=2 trainer.devices=[0]) or run with exps=e0 and an experiment config."
            )
        experiments_root = config.get("experiments_root") or "experiments"
        exp_dir_opt = config.get("exp_dir")
        if exp_dir_opt:
            if os.path.isabs(exp_dir_opt):
                exp_dir_single = exp_dir_opt
            else:
                root_abs = ensure_experiments_dir(experiments_root)
                exp_dir_single = os.path.join(root_abs, exp_dir_opt)
            if not os.path.isdir(exp_dir_single):
                raise RuntimeError(f"Experiment directory does not exist: {exp_dir_single}. Provide a valid exp_dir or omit it to create a new experiments set.")
            split_single = read_split_json(exp_dir_single)
            if split_single is None:
                raise RuntimeError(f"Missing split.json in {exp_dir_single}. This experiments set is invalid. Create a new experiments set or generate the split explicitly.")
        else:
            # Use experiments/ directly (no nested timestamp since Hydra run dir already has one)
            exp_dir_single = ensure_experiments_dir(experiments_root)
            split_cfg = config.get("split") or {}
            tsn = int(split_cfg.get("train_small_n", 7))
            tfn = int(split_cfg.get("train_full_n", 20))
            split_single = generate_building_split(
                seed=int(config.seed), n_buildings=25, train_small_n=tsn, train_full_n=tfn
            )
            write_split_json(exp_dir_single, split_single)
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
        # Enforce required data roots (no fallbacks)
        data_dir_req = config.datamodule.get("data_dir")
        data_dir_req = os.path.expanduser(str(data_dir_req)) if data_dir_req is not None else ""
        if not data_dir_req or not os.path.isdir(data_dir_req):
            raise RuntimeError(
                f"datamodule.data_dir must point to an existing ICASSP root. Got: {config.datamodule.get('data_dir')}"
            )
        use_synth = bool(config.datamodule.get("use_synthetic_train", False))
        if use_synth:
            synth_dir_req = config.datamodule.get("synthetic_dir")
            synth_dir_req = os.path.expanduser(str(synth_dir_req)) if synth_dir_req is not None else ""
            if not synth_dir_req or not os.path.isdir(synth_dir_req):
                raise RuntimeError(
                    f"datamodule.synthetic_dir must point to an existing synthetic root when use_synthetic_train=True. Got: {config.datamodule.get('synthetic_dir')}"
                )
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
