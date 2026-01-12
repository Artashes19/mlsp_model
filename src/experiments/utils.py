import logging
import os
import time
from typing import Optional

from omegaconf import DictConfig

from src.data_exploration.generate_manifest import (
    ensure_icassp_manifest,
    ensure_manifest as ensure_synth_manifest,
    filter_icassp_manifest,
    filter_synthetic_manifest,
)
from src.experiments.splits import (
    ensure_exp_dir, ensure_experiments_dir, generate_building_split, read_split_json,
    write_split_json,
)
from src.utils.utils import _count_rows

log = logging.getLogger(__name__)


def exp_root_prep(config: DictConfig):
    # Resolve exps root and exp_dir with strict rules:
    # - If exp_dir IS PROVIDED:
    #     * If absolute: use it directly (do NOT create a new exps root).
    #     * If relative: resolve it under experiments_root, creating experiments_root if needed.
    # - If exp_dir IS NOT PROVIDED: create a NEW timestamped dir and create split.json there.
    experiments_root = config.get("experiments_root") or "exps"
    log.info(f"[orchestrator] experiments_root={experiments_root}")
    exp_dir_opt = config.get("exp_dir")
    if exp_dir_opt:
        if os.path.isabs(exp_dir_opt):
            exp_dir = exp_dir_opt
        else:
            root_abs = ensure_experiments_dir(experiments_root)
            exp_dir = os.path.join(root_abs, exp_dir_opt)
        if not os.path.isdir(exp_dir):
            raise RuntimeError(
                f"Experiment directory does not exist: {exp_dir}. Provide a valid exp_dir or omit it to create a new exps set."
            )
        split = read_split_json(exp_dir)
        if split is None:
            raise RuntimeError(
                f"Missing split.json in {exp_dir}. This exps set is invalid. Create a new exps set or generate the split explicitly."
            )
        else:
            log.info(
                f"[split] loaded split.json from {exp_dir} "
                f"(seed={split.seed}, train_small={len(split.train_small)}, train_full={len(split.train_full)}, "
                f"validation={len(split.validation)})"
            )
    else:
        # Create NEW exps, set dir and write a split
        root_abs = ensure_experiments_dir(experiments_root)
        exp_dir = ensure_exp_dir(None, root_dir=root_abs)
        split_cfg = config.get("split") or {}
        tsn = int(split_cfg.get("train_small_n", 7))
        tfn = int(split_cfg.get("train_full_n", 20))
        split = generate_building_split(
            seed=int(config.seed), n_buildings=25, train_small_n=tsn,
            val_buildings=list(config["val_buildings"])
        )
        write_split_json(exp_dir, split)
        log.info(
            f"[split] created new exps set at {exp_dir} "
            f"(seed={split.seed}, train_small={len(split.train_small)}, train_full={len(split.train_full)}, "
            f"validation={len(split.validation)})"
        )
    
    return split, exp_dir


def create_exp_manifest(config: DictConfig, split, exp_list):
    # Prepare global manifests and per-exps-set filtered manifests
    run_dir_abs = os.path.realpath("./")
    manifests_dir = os.path.join(run_dir_abs, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)
    # Enforce the required ICASSP root early (fail fast)
    icassp_limit = 0
    synth_limit = 0
    icassp_global_manifest: Optional[str] = None
    synth_global_manifest: Optional[str] = None
    for exp in exp_list:
        if exp in ("e0", "e1", "e3"):
            icassp_root = os.path.expanduser(str(config["exps"][exp]["datamodule"].get("data_dir", "")))
            if not icassp_root or not os.path.isdir(icassp_root):
                raise RuntimeError(
                    f"datamodule.data_dir must point to an existing ICASSP root. Got: {config['exps'][exp].get('data_dir')}"
                )
            # Global ICASSP manifest under ICASSP root
            icassp_global_manifest = None
            if icassp_root and os.path.isdir(icassp_root):
                icassp_global_manifest = os.path.join(icassp_root, "icassp_manifest.csv")
                t0 = time.perf_counter()
                _ = ensure_icassp_manifest(
                    icassp_root, icassp_global_manifest, config["exps"][exp]["datamodule"].get("freqs_mhz", []),
                    task="Task_2_ICASSP"
                )
                dt = time.perf_counter() - t0
                rows = 0
                try:
                    with open(icassp_global_manifest, "r", newline="") as fp:
                        rows = sum(1 for _ in fp) - 1  # minus header
                except Exception:
                    rows = -1
                log.info(
                    f"[manifest] ensured ICASSP manifest at {icassp_global_manifest} "
                    f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)"
                )
                icassp_limit = int(config["exps"][exp]["datamodule"].get("icassp_limit_per_building", 0) or 0)
        if exp == "e2":
            # Global synthetic manifest under SYNTH root
            synth_root = os.path.expanduser(str(config["exps"][exp]["datamodule"].get("synthetic_dir", "")))
            synth_global_manifest = None
            if synth_root and os.path.isdir(synth_root):
                synth_global_manifest = os.path.join(synth_root, "samples.csv")
                t0 = time.perf_counter()
                _ = ensure_synth_manifest(
                    synth_root, synth_global_manifest, config["exps"][exp]["datamodule"].get("freqs_mhz", [])
                )
                dt = time.perf_counter() - t0
                rows = 0
                try:
                    with open(synth_global_manifest, "r", newline="") as fp:
                        rows = sum(1 for _ in fp) - 1
                except Exception:
                    rows = -1
                log.info(
                    f"[manifest] ensured synthetic manifest at {synth_global_manifest} "
                    f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)"
                )
                synth_limit = int(config["exps"][exp]["datamodule"].get("synthetic_limit", 0) or 0)
    
    # Build per-experiment-set filtered manifests
    # Create ICASSP train_small and train_full manifests if possible
    icassp_small_manifest = None
    icassp_full_manifest = None
    icassp_val_manifest = None
    synth_filtered_manifest = None
    if icassp_global_manifest and os.path.exists(icassp_global_manifest):
        icassp_small_manifest = os.path.join(manifests_dir, "icassp_train_small.filtered.csv")
        icassp_full_manifest = os.path.join(manifests_dir, "icassp_train_full.filtered.csv")
        icassp_val_manifest = os.path.join(manifests_dir, "icassp_validation.filtered.csv")
        t0 = t1 = time.perf_counter()
        _ = filter_icassp_manifest(
            icassp_global_manifest, icassp_small_manifest, list(split.train_small),
            icassp_limit if icassp_limit > 0 else None
        )
        t1 = time.perf_counter()
        _ = filter_icassp_manifest(
            icassp_global_manifest, icassp_full_manifest, list(split.train_full),
            icassp_limit if icassp_limit > 0 else None
        )
        t2 = time.perf_counter()
        _ = filter_icassp_manifest(
            icassp_global_manifest, icassp_val_manifest, list(split.validation),
            icassp_limit if icassp_limit > 0 else None
        )
        t3 = time.perf_counter()
        
        rows_small = _count_rows(icassp_small_manifest)
        rows_full = _count_rows(icassp_full_manifest)
        rows_val = _count_rows(icassp_val_manifest)
        log.info(
            f"[manifest] ICASSP filtered (small={len(split.train_small)} blds, limit_per_bld={icassp_limit or 'none'}) "
            f"-> {icassp_small_manifest} (rows={rows_small}, took={(t1 - t0):.2f}s)"
        )
        log.info(
            f"[manifest] ICASSP filtered (full={len(split.train_full)} blds, limit_per_bld={icassp_limit or 'none'}) "
            f"-> {icassp_full_manifest} (rows={rows_full}, took={(t2 - t1):.2f}s)"
        )
        log.info(
            f"[manifest] ICASSP filtered (validation={len(split.validation)} blds, limit_per_bld={icassp_limit or 'none'}) "
            f"-> {icassp_val_manifest} (rows={rows_val}, took={(t3 - t2):.2f}s)"
        )
    if synth_global_manifest and os.path.exists(synth_global_manifest):
        synth_filtered_manifest = os.path.join(manifests_dir, "synthetic.filtered.csv")
        t0 = time.perf_counter()
        _ = filter_synthetic_manifest(
            synth_global_manifest, synth_filtered_manifest, synth_limit if synth_limit > 0 else None
        )
        dt = time.perf_counter() - t0
        rows = _count_rows(synth_filtered_manifest)
        log.info(
            f"[manifest] Synthetic filtered (limit={synth_limit or 'none'}) -> {synth_filtered_manifest} "
            f"(rows={rows if rows >= 0 else 'unknown'}, took={dt:.2f}s)"
        )
    
    return icassp_small_manifest, icassp_val_manifest, icassp_full_manifest, synth_filtered_manifest
