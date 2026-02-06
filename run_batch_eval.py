"""
Resumable batch evaluation of checkpoints on MLSP tasks via Kaggle submission.

Usage:
    python run_batch_eval.py [--results results.csv] [--max-retries 3]

On each run, builds the full task queue, skips already-completed entries
found in results.csv, and processes the rest. If a single evaluation fails
it is pushed to the back of the queue and retried up to --max-retries times.
You can Ctrl-C at any time and re-run to resume.
"""
import argparse
import collections
import csv
import logging
import os
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Define the evaluation jobs:
#   (checkpoint_dir, task_name)
# ---------------------------------------------------------------------------
JOBS: list[tuple[str, str]] = []

# sr0_0002 checkpoints → mlsp_rate_0.02
_DIR_SR0_0002 = "/nfs/dgx/raid/iot/mlsp_wair_d_data/runs/mlsp_new_runs/v4_mlsp_finetune_0639_sr0_0002"
# sr0_005  checkpoints → mlsp_rate_0.02  AND  mlsp_rate_0.5
_DIR_SR0_005 = "/nfs/dgx/raid/iot/mlsp_wair_d_data/runs/mlsp_new_runs/v4_mlsp_finetune_0639_sr0_005"
# sr005    checkpoints → mlsp_rate_0.5
_DIR_SR005 = "/nfs/dgx/raid/iot/mlsp_wair_d_data/runs/mlsp_new_runs/v4_mlsp_finetune_0639_sr005"

def _ckpts(directory: str) -> list[str]:
    """Return sorted list of .ckpt file paths in *directory*."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".ckpt")
    )

def build_all_jobs() -> list[tuple[str, str]]:
    """Return the full list of (ckpt_path, task_name) pairs."""
    jobs: list[tuple[str, str]] = []
    for ckpt in _ckpts(_DIR_SR0_0002):
        jobs.append((ckpt, "mlsp_rate_0.02"))
    for ckpt in _ckpts(_DIR_SR0_005):
        jobs.append((ckpt, "mlsp_rate_0.02"))
        jobs.append((ckpt, "mlsp_rate_0.5"))
    for ckpt in _ckpts(_DIR_SR005):
        jobs.append((ckpt, "mlsp_rate_0.5"))
    return jobs

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
CSV_HEADER = ["checkpoint_dir", "checkpoint_file", "task", "rmse"]

def load_completed(csv_path: str) -> set[tuple[str, str, str]]:
    """Return set of (checkpoint_dir, checkpoint_file, task) already in csv."""
    completed: set[tuple[str, str, str]] = set()
    if not os.path.isfile(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["checkpoint_dir"], row["checkpoint_file"], row["task"])
            completed.add(key)
    return completed

def append_result(csv_path: str, ckpt_dir: str, ckpt_file: str, task: str, rmse: float) -> None:
    """Append a single result row (creates file + header if needed)."""
    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow([ckpt_dir, ckpt_file, task, f"{rmse:.6f}"])
        f.flush()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=str, default="results.csv",
                        help="Path to the results CSV (default: results.csv)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retry attempts per failed evaluation (default: 3)")
    parser.add_argument("--config-tree", type=str, default="config_tree.yaml",
                        help="Path to config_tree.yaml (default: config_tree.yaml)")
    args = parser.parse_args()

    csv_path = args.results
    max_retries = args.max_retries
    config_tree_path = os.path.abspath(args.config_tree)

    if not os.path.isfile(config_tree_path):
        log.error(f"Config tree not found: {config_tree_path}")
        sys.exit(1)

    # Lazy import so arg parsing is instant
    from src.eval_checkpoint import evaluate_checkpoint

    all_jobs = build_all_jobs()
    completed = load_completed(csv_path)

    # Build queue of remaining jobs  (ckpt_path, task, retries_left)
    queue: collections.deque[tuple[str, str, int]] = collections.deque()
    for ckpt_path, task in all_jobs:
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_file = os.path.basename(ckpt_path)
        if (ckpt_dir, ckpt_file, task) not in completed:
            queue.append((ckpt_path, task, max_retries))

    total = len(all_jobs)
    already_done = total - len(queue)
    log.info(f"Total jobs: {total}  |  Already done: {already_done}  |  Remaining: {len(queue)}")

    if not queue:
        log.info("Nothing to do – all evaluations are already in the results CSV.")
        return

    # Ensure CSV exists with header
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    done_this_run = 0
    failed_permanently = 0

    while queue:
        ckpt_path, task, retries_left = queue.popleft()
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_file = os.path.basename(ckpt_path)

        log.info(
            f"[{done_this_run + 1 + already_done}/{total}] "
            f"Evaluating {ckpt_file} on {task}  "
            f"(retries_left={retries_left}, queue_remaining={len(queue)})"
        )

        try:
            rmse = evaluate_checkpoint(
                task_name=task,
                ckpt_path=ckpt_path,
                config_tree_path=config_tree_path,
            )
            append_result(csv_path, ckpt_dir, ckpt_file, task, rmse)
            done_this_run += 1
            log.info(f"  -> RMSE={rmse:.6f}  (saved to {csv_path})")
        except KeyboardInterrupt:
            log.warning("Interrupted by user – progress saved. Re-run to resume.")
            sys.exit(1)
        except Exception:
            log.error(f"  -> FAILED:\n{traceback.format_exc()}")
            if retries_left > 1:
                log.info(f"  -> Re-queuing ({retries_left - 1} retries left)")
                queue.append((ckpt_path, task, retries_left - 1))
            else:
                log.error(f"  -> Permanently failed: {ckpt_file} / {task}")
                failed_permanently += 1

    log.info(
        f"Run complete.  Evaluated this run: {done_this_run}  |  "
        f"Previously done: {already_done}  |  "
        f"Failed permanently: {failed_permanently}  |  "
        f"Results: {csv_path}"
    )


if __name__ == "__main__":
    main()
