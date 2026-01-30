import json
import logging
import os
import time
from glob import glob
from typing import Optional

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# Mapping from (dataset_name, split_name) to Kaggle competition ID
KAGGLE_COMPETITIONS: dict[tuple[str, str], str] = {
    ("icassp_task1_dataset", "test"): "iprm-task-1",
    ("icassp_task2_dataset", "test"): "indoor-pathloss-radio-map-prediction-task-2",
    ("icassp_task3_dataset", "test"): "iprm-challenge",
    ("mlsp_dataset", "test_0.02"): "the-sampling-assisted-pathloss-rm-prediction",
    ("mlsp_dataset", "test_0.5"): "sampling-assisted-pathloss-rm-prediction-t-1-ii",
}


def get_competition_id(
    inference_dir: str,
) -> Optional[str]:
    """
    Extract Kaggle competition ID from directory path.
    
    Parses the path to get dataset_name and split_name, then looks up
    the competition ID from KAGGLE_COMPETITIONS mapping.
    
    Args:
        inference_dir: Path like .../mlsp_dataset/test_0.02
        
    Returns:
        Competition ID string or None if not found
    """
    # Get the last two components of the path: dataset_name/split_name
    parts = os.path.normpath(inference_dir).split(os.sep)
    if len(parts) < 2:
        return None
    
    split_name = parts[-1]
    dataset_name = parts[-2]
    
    key = (dataset_name, split_name)
    return KAGGLE_COMPETITIONS.get(key)


def submit_to_kaggle(
    csv_path: str,
    competition_id: str,
    max_retries: int = 10,
    retry_delay: float = 30.0,
    submit_retries: int = 5,
    submit_retry_delay: float = 60.0,
) -> Optional[float]:
    """
    Submit CSV to Kaggle competition and return the MSE score.
    
    Uses the Kaggle Python API directly for more reliable operation.
    
    Args:
        csv_path: Path to the predictions.csv file
        competition_id: Kaggle competition ID
        max_retries: Maximum number of retries to check submission status
        retry_delay: Delay in seconds between retries
        submit_retries: Maximum number of retries for submission (rate limiting)
        submit_retry_delay: Delay in seconds between submission retries
        
    Returns:
        MSE score from Kaggle or None if submission failed
    """
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    log.info(f"[evaluate] Submitting to Kaggle competition: {competition_id}")
    log.info(f"[evaluate] CSV file: {csv_path}")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Submit the CSV with retry logic for rate limiting
    for submit_attempt in range(submit_retries):
        try:
            api.competition_submit(
                file_name=csv_path,
                message="auto submission",
                competition=competition_id,
            )
            break  # Success, exit retry loop
        except requests.exceptions.HTTPError as e:
            log.error(f"[evaluate] Kaggle submission HTTPError: {e}")
            if e.response is not None and e.response.status_code in (403, 429):
                if submit_attempt < submit_retries - 1:
                    log.warning(
                        f"[evaluate] Rate limited (HTTP {e.response.status_code}), "
                        f"retrying in {submit_retry_delay}s (attempt {submit_attempt + 1}/{submit_retries})"
                    )
                    time.sleep(submit_retry_delay)
                else:
                    log.error(f"[evaluate] Failed to submit after {submit_retries} attempts: {e}")
                    raise
            else:
                raise
    
    log.info(f"[evaluate] Submission successful, waiting for scoring...")
    
    # Wait and check submission status
    for attempt in range(max_retries):
        time.sleep(retry_delay)
        
        # Get submissions list using Python API
        submissions = api.competition_submissions(competition=competition_id)
        
        if not submissions:
            log.warning(f"[evaluate] No submissions found yet (attempt {attempt + 1}/{max_retries})")
            continue
        
        # Get latest submission (first in list)
        latest = submissions[0]
        
        # Debug: log all attributes to identify the correct score field
        if attempt == 0:
            attrs = vars(latest) if hasattr(latest, "__dict__") else {a: getattr(latest, a, None) for a in dir(latest) if not a.startswith("_")}
            log.info(f"[evaluate] Submission attributes: {attrs}")
        
        # Check multiple possible score attributes
        score = None
        for attr in ["publicScore", "public_score", "score", "publicScoreNullable"]:
            val = getattr(latest, attr, None)
            if val is not None and str(val).lower() not in ["", "none", "pending"]:
                score = val
                break
        
        if score is None:
            log.info(f"[evaluate] Score pending (attempt {attempt + 1}/{max_retries})")
            continue
        
        # Parse score (handle both string and numeric)
        mse = float(score)
        log.info(f"[evaluate] Kaggle MSE score: {mse}")
        return mse
    
    log.error(f"[evaluate] Timed out waiting for Kaggle score after {max_retries} attempts")
    return None


def calculate_rmse(
    pred: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    out_norm: float,
) -> float:
    """
    Calculate RMSE between prediction and targets.
    
    Predictions and targets are normalized [0, 1], multiply by out_norm for dB.
    """
    # Convert from normalized to dB scale
    pred_db = pred * out_norm
    targets_db = targets * out_norm
    
    # Masked squared error
    se = ((pred_db - targets_db) ** 2) * masks
    mse = se.sum() / masks.sum()
    rmse = float(np.sqrt(mse))
    return rmse


def find_npz_directories(root_dir: str) -> list[str]:
    """
    Find all directories containing .npz files.
    
    If root_dir contains .npz files directly, return [root_dir].
    Otherwise, recursively find all subdirectories with .npz files.
    """
    # Check if root has .npz files directly
    if glob(os.path.join(root_dir, "*.npz")):
        return [root_dir]
    
    # Recursively find directories with .npz files
    npz_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".npz") for f in filenames):
            npz_dirs.append(dirpath)
    
    return sorted(npz_dirs)


def is_test_directory(
    inference_dir: str,
) -> bool:
    """
    Check if a directory contains test data (no ground truth targets).
    
    Returns True if:
    - predictions.csv exists, AND
    - First .npz file has targets=None
    """
    csv_path = os.path.join(inference_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        return False
    
    # Check first .npz file for targets
    npz_files = sorted(glob(os.path.join(inference_dir, "*.npz")))
    if not npz_files:
        return False
    
    data = np.load(npz_files[0], allow_pickle=True)
    targets = data["targets"]
    
    # targets will be None or an array with None value for test data
    return targets is None or (hasattr(targets, "item") and targets.item() is None)


def evaluate_test_directory(
    inference_dir: str,
) -> Optional[tuple[float, int]]:
    """
    Evaluate test directory by submitting to Kaggle.
    
    Args:
        inference_dir: Directory containing predictions.csv and .npz files
        
    Returns:
        (mse_score, num_samples) or None if submission failed
    """
    # Check for predictions.csv
    csv_path = os.path.join(inference_dir, "predictions.csv")
    if not os.path.exists(csv_path):
        log.warning(f"[evaluate] No predictions.csv found in {inference_dir}")
        return None
    
    # Get competition ID from path
    competition_id = get_competition_id(inference_dir)
    if competition_id is None:
        log.warning(f"[evaluate] Unknown competition for path: {inference_dir}")
        return None
    
    # Count samples from .npz files
    npz_files = glob(os.path.join(inference_dir, "*.npz"))
    num_samples = len(npz_files)
    
    # Submit to Kaggle
    mse = submit_to_kaggle(
        csv_path=csv_path,
        competition_id=competition_id,
    )
    
    if mse is None:
        return None
    
    # Create empty JSON file with MSE value in name
    output_filename = f"MSE_{mse:.6f}.json"
    output_path = os.path.join(inference_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump({}, f)
    
    log.info(f"[evaluate] Created result file: {output_filename}")
    
    return mse, num_samples


def evaluate_directory(
    inference_dir: str,
    out_norm: float,
) -> tuple[float, int]:
    """
    Evaluate a single directory containing .npz files.
    
    Returns (average_rmse, num_samples).
    """
    # Find all .npz files
    npz_files = sorted(glob(os.path.join(inference_dir, "*.npz")))
    
    if len(npz_files) == 0:
        log.warning(f"[evaluate] No .npz files found in {inference_dir}")
        return 0.0, 0
    
    # Calculate RMSE for each sample
    results = {}
    total_rmse = 0.0
    
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        
        pred = data["pred"]
        targets = data["targets"]
        masks = data["masks"]
        file_name = str(data["file_name"])
        
        # Skip if targets is None
        if targets is None:
            log.warning(f"[evaluate] Skipping {file_name}: no targets")
            continue
        
        # Calculate RMSE
        rmse = calculate_rmse(
            pred=pred,
            targets=targets,
            masks=masks,
            out_norm=out_norm,
        )
        
        results[file_name] = rmse
        total_rmse += rmse
    
    if len(results) == 0:
        log.warning(f"[evaluate] No valid samples with targets found in {inference_dir}")
        return 0.0, 0
    
    # Calculate average RMSE
    avg_rmse = total_rmse / len(results)
    
    # Write results to JSON file named with average RMSE
    output_filename = f"RMSE_{avg_rmse:.6f}.json"
    output_path = os.path.join(inference_dir, output_filename)
    
    with open(output_path, "w") as f:
        json.dump(dict(sorted(results.items())), f, indent=4)
    
    return avg_rmse, len(results)


def evaluate_prep(
    config: DictConfig,
    project_root: str,
) -> None:
    """
    Evaluate inference predictions against ground truth or via Kaggle submission.
    
    Supports recursive evaluation - if the provided directory doesn't contain
    .npz files directly, it will find and evaluate all subdirectories that do.
    
    For test directories (no ground truth), submits to Kaggle to get MSE scores.
    """
    inference_dir = os.path.abspath(str(config["inference_dir"]))
    out_norm = float(config.get("out_norm", 160.0))
    
    # Validate inference directory exists
    if not os.path.isdir(inference_dir):
        raise RuntimeError(f"Inference directory not found: {inference_dir}")
    
    log.info(f"[evaluate] Inference directory: {inference_dir}")
    log.info(f"[evaluate] Output normalization: {out_norm}")
    
    # Find all directories with .npz files
    npz_dirs = find_npz_directories(inference_dir)
    
    if not npz_dirs:
        raise RuntimeError(f"No .npz files found in {inference_dir} or subdirectories")
    
    log.info(f"[evaluate] Found {len(npz_dirs)} directories to evaluate")
    
    # Evaluate each directory
    all_results = {}
    test_results = {}
    total_samples = 0
    total_test_samples = 0
    
    for npz_dir in npz_dirs:
        rel_path = os.path.relpath(npz_dir, inference_dir)
        if rel_path == ".":
            rel_path = os.path.basename(inference_dir)
        
        # Check if this is a test directory (no ground truth)
        if is_test_directory(npz_dir):
            log.info(f"[evaluate] Evaluating test data via Kaggle: {rel_path}")
            result = evaluate_test_directory(inference_dir=npz_dir)
            
            if result is not None:
                mse, num_samples = result
                test_results[npz_dir] = {
                    "mse": mse,
                    "samples": num_samples,
                }
                total_test_samples += num_samples
                log.info(f"[evaluate]   MSE: {mse:.6f} ({num_samples} samples)")
        else:
            log.info(f"[evaluate] Evaluating: {rel_path}")
            avg_rmse, num_samples = evaluate_directory(
                inference_dir=npz_dir,
                out_norm=out_norm,
            )
            
            if num_samples > 0:
                all_results[npz_dir] = {
                    "rmse": avg_rmse,
                    "samples": num_samples,
                }
                total_samples += num_samples
                log.info(f"[evaluate]   RMSE: {avg_rmse:.6f} dB ({num_samples} samples)")
    
    # Summary for regular evaluation
    if all_results:
        log.info(f"[evaluate] === Validation Summary ===")
        for path, result in all_results.items():
            rel_path = os.path.relpath(path, inference_dir)
            if rel_path == ".":
                rel_path = os.path.basename(inference_dir)
            log.info(f"[evaluate]   {rel_path}: RMSE={result['rmse']:.6f} dB ({result['samples']} samples)")
        log.info(f"[evaluate] Total validation: {total_samples} samples across {len(all_results)} directories")
    
    # Summary for test evaluation (Kaggle)
    if test_results:
        log.info(f"[evaluate] === Test Summary (Kaggle) ===")
        for path, result in test_results.items():
            rel_path = os.path.relpath(path, inference_dir)
            if rel_path == ".":
                rel_path = os.path.basename(inference_dir)
            log.info(f"[evaluate]   {rel_path}: MSE={result['mse']:.6f} ({result['samples']} samples)")
        log.info(f"[evaluate] Total test: {total_test_samples} samples across {len(test_results)} directories")
    
    if total_samples == 0 and total_test_samples == 0:
        raise RuntimeError("No valid samples found in any directory")
