import logging
import os
import time
from typing import Sequence

from omegaconf import DictConfig

from src.data_exploration.generate_manifest import (
    ensure_icassp_manifest,
    ensure_manifest as ensure_synth_manifest,
    filter_icassp_manifest,
    generate_icassp_test_manifest,
    split_synthetic_manifest,
)

log = logging.getLogger(__name__)


def _build_val_buildings_suffix(val_buildings: Sequence[int]) -> str:
    """Build a suffix string from validation building IDs, e.g., '21_22_23_24_25'."""
    return "_".join(str(b) for b in sorted(val_buildings))


def manifest_prep(
    config: DictConfig,
    project_root: str,
) -> dict[str, str]:
    """
    Generate manifest files for ICASSP and synthetic datasets.
    
    Iterates over all configured tasks and creates per-task manifests:
    - icassp_train_{task}_val_{buildings}.csv - ICASSP full training data
    - icassp_train_small_{task}_val_{buildings}_n{size}.csv - ICASSP small training data
    - icassp_val_{task}_{buildings}.csv - ICASSP validation data
    - synthetic_train.csv - full synthetic training manifest
    - synthetic_val_{size}.csv - synthetic validation subset
    
    Returns dict of manifest paths.
    """
    icassp_data_dir = os.path.expanduser(str(config["icassp_data_dir"]))
    synthetic_dir = os.path.expanduser(str(config["synthetic_dir"]))
    icassp_eval_dir = os.path.expanduser(str(config.get("icassp_eval_dir", "")))
    val_buildings = list(config["val_buildings"])
    train_small_n = int(config["train_small_n"])
    synthetic_val_size = int(config["synthetic_val_size"])
    freqs_mhz = list(config["freqs_mhz"])
    tasks = list(config["tasks"])
    generate_synthetic = bool(config["generate_synthetic"])
    generate_test = bool(config.get("generate_test", False))
    
    # Output directories
    icassp_manifest_dir = os.path.expanduser(str(config["icassp_manifest_dir"]))
    synthetic_manifest_dir = os.path.expanduser(str(config["synthetic_manifest_dir"]))
    
    # Build naming suffixes
    val_suffix = _build_val_buildings_suffix(val_buildings)
    
    result = {}
    
    # === ICASSP Manifests ===
    if icassp_data_dir and os.path.isdir(icassp_data_dir):
        os.makedirs(icassp_manifest_dir, exist_ok=True)
        
        # Compute train buildings (all buildings not in validation)
        all_buildings = list(range(1, 26))
        train_buildings = sorted(set(all_buildings) - set(val_buildings))
        train_small_buildings = train_buildings[:train_small_n]
        
        for task in tasks:
            log.info(f"[manifest] Processing task: {task}")
            
            # First, ensure the global ICASSP manifest exists
            icassp_global_manifest = os.path.join(icassp_data_dir, f"icassp_manifest_{task}.csv")
            t0 = time.perf_counter()
            ensure_icassp_manifest(
                root=icassp_data_dir,
                out_csv=icassp_global_manifest,
                freqs_mhz=freqs_mhz,
                task=task,
            )
            dt = time.perf_counter() - t0
            rows = _count_rows(icassp_global_manifest)
            log.info(
                f"[manifest] ensured ICASSP global manifest at {icassp_global_manifest} "
                f"(rows={rows}, took={dt:.2f}s)"
            )
            
            # Generate full training manifest (all non-validation buildings)
            icassp_train_manifest = os.path.join(
                icassp_manifest_dir,
                f"icassp_train_{task}_val_{val_suffix}.csv",
            )
            t0 = time.perf_counter()
            n_train = filter_icassp_manifest(
                src_csv=icassp_global_manifest,
                out_csv=icassp_train_manifest,
                allow_buildings=train_buildings,
                limit_per_building=None,
            )
            dt = time.perf_counter() - t0
            log.info(
                f"[manifest] ICASSP train manifest: {icassp_train_manifest} "
                f"(buildings={len(train_buildings)}, rows={n_train}, took={dt:.2f}s)"
            )
            result[f"icassp_train_manifest_{task}"] = icassp_train_manifest
            
            # Generate small training manifest (subset of training buildings)
            icassp_train_small_manifest = os.path.join(
                icassp_manifest_dir,
                f"icassp_train_small_{task}_val_{val_suffix}_n{train_small_n}.csv",
            )
            t0 = time.perf_counter()
            n_train_small = filter_icassp_manifest(
                src_csv=icassp_global_manifest,
                out_csv=icassp_train_small_manifest,
                allow_buildings=train_small_buildings,
                limit_per_building=None,
            )
            dt = time.perf_counter() - t0
            log.info(
                f"[manifest] ICASSP train_small manifest: {icassp_train_small_manifest} "
                f"(buildings={len(train_small_buildings)}, rows={n_train_small}, took={dt:.2f}s)"
            )
            result[f"icassp_train_small_manifest_{task}"] = icassp_train_small_manifest
            
            # Generate validation manifest (only val_buildings)
            icassp_val_manifest = os.path.join(
                icassp_manifest_dir,
                f"icassp_val_{task}_{val_suffix}.csv",
            )
            t0 = time.perf_counter()
            n_val = filter_icassp_manifest(
                src_csv=icassp_global_manifest,
                out_csv=icassp_val_manifest,
                allow_buildings=val_buildings,
                limit_per_building=None,
            )
            dt = time.perf_counter() - t0
            log.info(
                f"[manifest] ICASSP val manifest: {icassp_val_manifest} "
                f"(buildings={len(val_buildings)}, rows={n_val}, took={dt:.2f}s)"
            )
            result[f"icassp_val_manifest_{task}"] = icassp_val_manifest
    else:
        log.warning(f"[manifest] ICASSP data dir not found or not specified: {icassp_data_dir}")
    
    # === Synthetic Manifests ===
    if generate_synthetic:
        if synthetic_dir and os.path.isdir(synthetic_dir):
            os.makedirs(synthetic_manifest_dir, exist_ok=True)
            
            # First, ensure the global synthetic manifest exists
            synth_global_manifest = os.path.join(synthetic_dir, "samples.csv")
            t0 = time.perf_counter()
            ensure_synth_manifest(
                root=synthetic_dir,
                out=synth_global_manifest,
                freqs_mhz=freqs_mhz,
                limit=0,
            )
            dt = time.perf_counter() - t0
            rows = _count_rows(synth_global_manifest)
            log.info(
                f"[manifest] ensured synthetic global manifest at {synth_global_manifest} "
                f"(rows={rows}, took={dt:.2f}s)"
            )
            
            # Generate disjoint train/val manifests (shuffled, then split)
            synth_train_manifest = os.path.join(
                synthetic_manifest_dir,
                "synthetic_train.csv",
            )
            synth_val_manifest = os.path.join(
                synthetic_manifest_dir,
                f"synthetic_val_{synthetic_val_size}.csv",
            )
            t0 = time.perf_counter()
            n_train, n_val = split_synthetic_manifest(
                src_csv=synth_global_manifest,
                train_csv=synth_train_manifest,
                val_csv=synth_val_manifest,
                val_size=synthetic_val_size,
            )
            dt = time.perf_counter() - t0
            log.info(
                f"[manifest] Synthetic train manifest: {synth_train_manifest} "
                f"(rows={n_train}, took={dt:.2f}s)"
            )
            log.info(
                f"[manifest] Synthetic val manifest: {synth_val_manifest} "
                f"(rows={n_val}, took={dt:.2f}s)"
            )
            result["synth_train_manifest"] = synth_train_manifest
            result["synth_val_manifest"] = synth_val_manifest
        else:
            log.warning(f"[manifest] Synthetic data dir not found or not specified: {synthetic_dir}")
    else:
        log.info("[manifest] Synthetic manifest generation disabled")
    
    # === Test Manifests (evaluation data, no outputs) ===
    if generate_test:
        if icassp_eval_dir and os.path.isdir(icassp_eval_dir):
            eval_manifest_dir = os.path.join(icassp_eval_dir, "manifests")
            os.makedirs(eval_manifest_dir, exist_ok=True)
            
            # Mapping of task number to eval directory and task subfolder
            test_task_mapping = {
                "Task_1_ICASSP": ("Evaluation_Data_T1", "Task_1_ICASSP"),
                "Task_2_ICASSP": ("Evaluation_Data_T2", "Task_2_ICASSP"),
                "Task_3_ICASSP": ("Evaluation_Data_T3", "Task_3_ICASSP"),
            }
            
            for task in tasks:
                if task not in test_task_mapping:
                    log.warning(f"[manifest] Unknown task for test manifest: {task}")
                    continue
                
                eval_data_name, task_subfolder = test_task_mapping[task]
                task_short = task.replace("_ICASSP", "")
                
                # Generate test manifest (no sparse)
                test_manifest_path = os.path.join(
                    eval_manifest_dir,
                    f"icassp_test_{task_short}.csv",
                )
                t0 = time.perf_counter()
                n_test = generate_icassp_test_manifest(
                    root=icassp_eval_dir,
                    out_csv=test_manifest_path,
                    freqs_mhz=freqs_mhz,
                    eval_data_name=eval_data_name,
                    task=task_subfolder,
                    sparse_dir="",
                )
                dt = time.perf_counter() - t0
                log.info(
                    f"[manifest] Test manifest: {test_manifest_path} "
                    f"(rows={n_test}, took={dt:.2f}s)"
                )
                result[f"icassp_test_manifest_{task}"] = test_manifest_path
            
            # Generate MLSP test manifests with sparse measurements (Task 1 data dir)
            if "Task_1_ICASSP" in tasks:
                eval_data_name = "Evaluation_Data_T1"
                task_subfolder = "Task_1_ICASSP"
                
                # Rate 0.02% sparse
                sparse_dir_0_02 = os.path.join(icassp_eval_dir, eval_data_name, "rate0.02", "sampledGT")
                if os.path.isdir(sparse_dir_0_02):
                    mlsp_test_path_0_02 = os.path.join(eval_manifest_dir, "mlsp_test_rate0.02.csv")
                    t0 = time.perf_counter()
                    n_mlsp = generate_icassp_test_manifest(
                        root=icassp_eval_dir,
                        out_csv=mlsp_test_path_0_02,
                        freqs_mhz=freqs_mhz,
                        eval_data_name=eval_data_name,
                        task=task_subfolder,
                        sparse_dir=sparse_dir_0_02,
                    )
                    dt = time.perf_counter() - t0
                    log.info(
                        f"[manifest] MLSP test manifest (rate 0.02): {mlsp_test_path_0_02} "
                        f"(rows={n_mlsp}, took={dt:.2f}s)"
                    )
                    result["mlsp_test_manifest_rate_0_02"] = mlsp_test_path_0_02
                
                # Rate 0.5% sparse
                sparse_dir_0_5 = os.path.join(icassp_eval_dir, eval_data_name, "rate0.5", "sampledGT")
                if os.path.isdir(sparse_dir_0_5):
                    mlsp_test_path_0_5 = os.path.join(eval_manifest_dir, "mlsp_test_rate0.5.csv")
                    t0 = time.perf_counter()
                    n_mlsp = generate_icassp_test_manifest(
                        root=icassp_eval_dir,
                        out_csv=mlsp_test_path_0_5,
                        freqs_mhz=freqs_mhz,
                        eval_data_name=eval_data_name,
                        task=task_subfolder,
                        sparse_dir=sparse_dir_0_5,
                    )
                    dt = time.perf_counter() - t0
                    log.info(
                        f"[manifest] MLSP test manifest (rate 0.5): {mlsp_test_path_0_5} "
                        f"(rows={n_mlsp}, took={dt:.2f}s)"
                    )
                    result["mlsp_test_manifest_rate_0_5"] = mlsp_test_path_0_5
        else:
            log.warning(f"[manifest] Evaluation data dir not found or not specified: {icassp_eval_dir}")
    else:
        log.info("[manifest] Test manifest generation disabled")
    
    # Summary
    log.info(f"[manifest] Manifest generation complete. Created manifests: {list(result.keys())}")
    for key, path in result.items():
        log.info(f"  {key}: {path}")
    
    return result


def _count_rows(csv_path: str) -> int:
    """Count rows in a CSV file (excluding header)."""
    if not csv_path or not os.path.exists(csv_path):
        return -1
    with open(csv_path, "r", newline="") as fp:
        return sum(1 for _ in fp) - 1
