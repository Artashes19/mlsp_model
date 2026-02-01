"""
Minimal checkpoint evaluation for Kaggle test tasks.
"""
import csv
import logging
import os
import re
import sys
import tempfile

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.algorithms.indoor import Indoor
from src.datamodules.datasets.indoor import PathlossDataset
from src.datamodules.indoor import IndoorDatamodule
from src.evaluate import submit_to_kaggle
from src.inference import generate_csv_rows
from src.utils.indoor.channel_config import NUM_CHANNELS


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[FlushingStreamHandler(sys.stderr)],
    force=True,
)
log = logging.getLogger(__name__)


TASKS: dict[str, dict[str, object]] = {
    "icassp_task_1": {
        "manifest_path": "/nfs/dgx/raid/iot/new_manifests/icassp_test_Task_1.csv",
        "competition_id": "iprm-task-1",
    },
    "icassp_task_2": {
        "manifest_path": "/nfs/dgx/raid/iot/new_manifests/icassp_test_Task_2.csv",
        "competition_id": "indoor-pathloss-radio-map-prediction-task-2",
    },
    "icassp_task_3": {
        "manifest_path": "/nfs/dgx/raid/iot/new_manifests/icassp_test_Task_3.csv",
        "competition_id": "iprm-challenge",
    },
    "mlsp_rate_0.02": {
        "manifest_path": "/nfs/dgx/raid/iot/new_manifests/mlsp_test_rate0.02.csv",
        "competition_id": "the-sampling-assisted-pathloss-rm-prediction",
    },
    "mlsp_rate_0.5": {
        "manifest_path": "/nfs/dgx/raid/iot/new_manifests/mlsp_test_rate0.5.csv",
        "competition_id": "sampling-assisted-pathloss-rm-prediction-t-1-ii",
    },
}


def load_manifest(manifest_path: str) -> list[dict]:
    return IndoorDatamodule.get_inputs_list(manifest_path=manifest_path)


def _resolve_config_tree_path(ckpt_path: str, config_tree_path: str | None) -> str:
    if config_tree_path:
        if not os.path.isfile(config_tree_path):
            raise RuntimeError(f"Config tree not found: {config_tree_path}")
        return config_tree_path
    search_dir = os.path.abspath(os.path.dirname(ckpt_path))
    for _ in range(8):
        candidate = os.path.join(search_dir, "config_tree.yaml")
        if os.path.isfile(candidate):
            log.info(f"[model] Derived config_tree.yaml from checkpoint path: {candidate}")
            return candidate
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    raise RuntimeError(
        "Unable to locate config_tree.yaml. Provide --config-tree or place it in a parent directory."
    )


def _load_network_conf(
    config_tree_path: str,
    exp_name: str | None,
    ckpt_path: str,
) -> tuple[DictConfig, str]:
    config = OmegaConf.load(config_tree_path)
    exps = config.get("exps")
    if not exps:
        raise RuntimeError(f"No exps found in config_tree.yaml: {config_tree_path}")
    if exp_name is None:
        path_parts = set(os.path.normpath(ckpt_path).split(os.sep))
        for candidate in exps.keys():
            if candidate in path_parts:
                exp_name = candidate
                log.info(f"[model] Derived exp name from checkpoint path: {exp_name}")
                break
    if exp_name is None:
        exp_name = next(iter(exps.keys()))
    if exp_name not in exps:
        raise RuntimeError(f"exp '{exp_name}' not found in config_tree.yaml: {config_tree_path}")
    network_conf = exps[exp_name].get("network")
    if not network_conf:
        raise RuntimeError(f"No network section found for exp '{exp_name}' in {config_tree_path}")
    network_conf = OmegaConf.create(network_conf)
    if "in_ch" not in network_conf:
        network_conf["in_ch"] = NUM_CHANNELS
    return network_conf, exp_name


def load_model_from_checkpoint(
    ckpt_path: str,
    gpu: int,
    config_tree_path: str | None = None,
    exp_name: str | None = None,
) -> Indoor:
    log.info(f"[model] Loading checkpoint: {ckpt_path}")
    resolved_config = _resolve_config_tree_path(ckpt_path=ckpt_path, config_tree_path=config_tree_path)
    network_conf, resolved_exp = _load_network_conf(
        config_tree_path=resolved_config,
        exp_name=exp_name,
        ckpt_path=ckpt_path,
    )
    log.info(f"[model] Using network config from {resolved_config} (exp={resolved_exp})")
    compiled_conf = DictConfig({
        "_target_": "src.utils.CompileParams",
        "fullgraph": True,
        "dynamic": False,
        "backend": "inductor",
        "mode": "max-autotune",
        "options": None,
        "disable": True,
    })
    algorithm = Indoor(
        out_norm=160.0,
        use_sip2net=False,
        sip2net_params={},
        compiled=compiled_conf,
        optimizer_conf=None,
        scheduler_conf=None,
        network=None,
        network_conf=OmegaConf.to_yaml(network_conf),
        gpu=gpu,
    )
    ckpt = torch.load(ckpt_path, weights_only=False)
    algorithm.load_state_dict(ckpt["state_dict"])
    algorithm.eval()
    algorithm.cuda(gpu)
    return algorithm


def run_inference(
    algorithm: Indoor,
    dataset: PathlossDataset,
    output_dir: str,
) -> list[tuple[str, float]]:
    os.makedirs(output_dir, exist_ok=True)
    csv_rows = []
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            batch = dataset[idx]
            result = algorithm.pred(batch=batch)
            sample_meta = batch[3]
            file_name = sample_meta["file_name"]
            safe_file_name = re.sub(r"[^\w\-_.]", "_", str(file_name))
            output_path = os.path.join(output_dir, f"{safe_file_name}.npz")
            np.savez(
                output_path,
                pred=result["pred"],
                inputs=result["inputs"],
                targets=result["targets"],
                masks=result["masks"],
                file_name=result["sample"]["file_name"],
                pixel_size=result["sample"]["pixel_size"],
            )
            rows = generate_csv_rows(
                file_name=file_name,
                pred=result["pred"],
                mask=result["masks"],
            )
            csv_rows.extend(rows)
    return csv_rows


def write_csv(csv_rows: list[tuple[str, float]], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "PL"])
        for id_str, pl_value in csv_rows:
            writer.writerow([id_str, round(pl_value, 1)])


def evaluate_checkpoint(
    task_name: str,
    ckpt_path: str,
    config_tree_path: str | None = None,
    exp_name: str | None = None,
) -> float:
    task = TASKS[task_name]
    manifest_path = task["manifest_path"]
    competition_id = task["competition_id"]
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(f"Manifest not found: {manifest_path}")
    log.info(f"[{task_name}] Loading manifest: {manifest_path}")
    inputs_list = load_manifest(manifest_path=manifest_path)
    if len(inputs_list) == 0:
        raise RuntimeError(f"No valid inputs found in manifest: {manifest_path}")
    log.info(f"[{task_name}] Building dataset: {len(inputs_list)} samples")
    dataset = PathlossDataset(
        inputs_list=inputs_list,
        training=False,
        inference=True,
        augmentations=None,
        sparse_range=[0, 0],
        modality_dropout_prob=0,
        sparse_dropout_given_dropout=1.0,
        channels="rtdgps",
        force_drop_sparse=False,
        force_drop_trans_ref=False,
    )
    algorithm = load_model_from_checkpoint(
        ckpt_path=ckpt_path,
        gpu=0,
        config_tree_path=config_tree_path,
        exp_name=exp_name,
    )
    log.info(f"[{task_name}] Running inference over {len(dataset)} samples")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_rows = run_inference(
            algorithm=algorithm,
            dataset=dataset,
            output_dir=tmpdir,
        )
        csv_path = os.path.join(tmpdir, "predictions.csv")
        write_csv(csv_rows=csv_rows, output_path=csv_path)
        log.info(f"[{task_name}] Submitting to Kaggle (competition_id={competition_id})")
        mse = submit_to_kaggle(
            csv_path=csv_path,
            competition_id=competition_id,
        )
    if mse is None:
        raise RuntimeError("Kaggle submission failed or score unavailable")
    rmse = float(np.sqrt(mse))
    log.info(f"[{task_name}] Done -> RMSE={rmse:.6f}")
    return rmse


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on MLSP tasks and save RMSE to CSV")
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing .ckpt files (e.g. .../checkpoints/every)",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to write results CSV (columns: checkpoint, task, rmse)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mlsp_rate_0.02", "mlsp_rate_0.5"],
        help="Task names from TASKS (default: mlsp_rate_0.02 mlsp_rate_0.5)",
    )
    parser.add_argument(
        "--config-tree",
        type=str,
        default=None,
        help="Path to config_tree.yaml (defaults to searching parent directories of the checkpoint)",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment key under config_tree.yaml exps (defaults to the first entry)",
    )
    args = parser.parse_args()
    ckpt_dir = args.checkpoint_dir
    output_path = args.output_csv
    task_names = args.tasks
    config_tree_path = args.config_tree
    exp_name = args.exp_name
    for name in task_names:
        if name not in TASKS:
            raise RuntimeError(f"Unknown task: {name}. Valid: {list(TASKS)}")
    ckpt_files = sorted(
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt")
    )
    if not ckpt_files:
        raise RuntimeError(f"No .ckpt files in {ckpt_dir}")
    n_ckpts = len(ckpt_files)
    n_tasks = len(task_names)
    log.info(
        f"Starting batch: {n_ckpts} checkpoints x {n_tasks} tasks = {n_ckpts * n_tasks} evaluations. "
        f"Tasks: {task_names}"
    )
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "task", "rmse"])
        f.flush()
        for ckpt_idx, ckpt_name in enumerate(ckpt_files, start=1):
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            log.info(f"--- Checkpoint {ckpt_idx}/{n_ckpts}: {ckpt_name} ---")
            for task_name in task_names:
                log.info(f"  Task: {task_name}")
                rmse = evaluate_checkpoint(
                    task_name=task_name,
                    ckpt_path=ckpt_path,
                    config_tree_path=config_tree_path,
                    exp_name=exp_name,
                )
                writer.writerow([ckpt_name, task_name, f"{rmse:.6f}"])
                f.flush()
                log.info(f"  -> rmse={rmse:.6f} (saved)")
    log.info(f"Finished. Results in {output_path}")


if __name__ == "__main__":
    main()
