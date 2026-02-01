"""
Plot RMSE vs epoch (1-based) from results/eval_results.csv.
Both MLSP 0.02 and 0.5 on the same axes. Saves results/eval_rmse.png.
"""
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "eval_results.csv"
OUT_PATH = RESULTS_DIR / "eval_rmse.png"


def parse_epoch(checkpoint: str) -> int:
    m = re.match(r"epoch_(\d+)_every\.ckpt", checkpoint)
    if m is None:
        raise ValueError(f"Cannot parse epoch from checkpoint: {checkpoint}")
    return int(m.group(1))


def load_results() -> list[tuple[str, str, float]]:
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["checkpoint"], row["task"], float(row["rmse"])))
    return rows


def main() -> None:
    rows = load_results()
    tasks = [
        ("mlsp_rate_0.02", "MLSP rate 0.02"),
        ("mlsp_rate_0.5", "MLSP rate 0.5"),
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    all_epochs_1based = []
    for task_key, label in tasks:
        data = [(parse_epoch(c), r) for c, t, r in rows if t == task_key]
        data.sort(key=lambda x: x[0])
        epochs_1based = [e + 1 for e, _ in data]
        all_epochs_1based.extend(epochs_1based)
        rmse = [r for _, r in data]
        ax.plot(epochs_1based, rmse, "o-", markersize=4, label=label)
    min_ep, max_ep = min(all_epochs_1based), max(all_epochs_1based)
    tick_epochs = list(range(10, max_ep + 1, 10))
    if min_ep < 10 or min_ep not in tick_epochs:
        tick_epochs = [min_ep] + [e for e in tick_epochs if e > min_ep]
    tick_epochs = sorted(set(tick_epochs))
    ax.set_xticks(tick_epochs)
    ax.set_xticklabels([str(e) for e in tick_epochs])
    ax.set_xlabel("Epoch (1-based)")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
