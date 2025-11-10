from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Ensure non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorBoard is required to read event files. Please install tensorboard (pip install tensorboard)."
    ) from exc


def find_latest_run_dir(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    run_dirs: List[Path] = [p for p in runs_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")
    # Sort by modification time, descending
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def find_tb_dir(run_dir: Path) -> Path:
    """
    Resolve the TensorBoard directory within a run directory.
    Default is run_dir / "tb"; if missing, fall back to run_dir itself if it contains event files.
    """
    tb_dir = run_dir / "tb"
    if tb_dir.exists():
        return tb_dir
    # Fallback: use run_dir if it contains TB event files
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if event_files:
        return run_dir
    # Try any subdir containing TB events
    for sub in run_dir.rglob("*"):
        if sub.is_dir() and list(sub.glob("events.out.tfevents.*")):
            return sub
    raise FileNotFoundError(f"TensorBoard event files not found under: {run_dir}")


def load_scalars(tb_dir: Path, tags: List[str]) -> Dict[str, Tuple[List[int], List[float]]]:
    """
    Load scalar series for provided tags from TensorBoard event files in tb_dir.
    Returns a mapping: tag -> (steps, values)
    """
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    for tag in tags:
        tag_to_use = None
        if tag in available:
            tag_to_use = tag
        else:
            # Try common variants (with/without leading slash and suffixes)
            base = tag.lstrip("/")
            candidates = [base]
            for suf in ("_epoch", "_step"):
                if base.endswith(suf):
                    candidates.append(base[: -len(suf)])
            # Search available tags for best match
            for t in available:
                if t in candidates or t.lstrip("/") in candidates:
                    tag_to_use = t
                    break
            if tag_to_use is None:
                for t in available:
                    if any(t.endswith(c) for c in candidates):
                        tag_to_use = t
                        break
            if tag_to_use is None:
                raise KeyError(
                    f"Scalar tag not found in TB logs: {tag} | available: {sorted(list(available))[:20]} ..."
                )
        evts = ea.Scalars(tag_to_use)
        steps = [int(e.step) for e in evts]
        vals = [float(e.value) for e in evts]
        series[tag] = (steps, vals)
    return series


def plot_val_metrics(
    run_dir: Path,
    tags: Tuple[str, str] = ("val/l1_epoch", "val/rmse_db_epoch"),
    output: Path | None = None,
) -> Path:
    tb_dir = find_tb_dir(run_dir)
    data = load_scalars(tb_dir, list(tags))

    l1_steps, l1_vals = data[tags[0]]
    rmse_steps, rmse_vals = data[tags[1]]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color_l1 = "tab:blue"
    color_rmse = "tab:red"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val L1", color=color_l1)
    ax1.plot(l1_steps, l1_vals, marker="o", linestyle="-", color=color_l1, label="val L1")
    ax1.tick_params(axis="y", labelcolor=color_l1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val RMSE (dB)", color=color_rmse)
    ax2.plot(rmse_steps, rmse_vals, marker="s", linestyle="--", color=color_rmse, label="val RMSE dB")
    ax2.tick_params(axis="y", labelcolor=color_rmse)

    title = f"Validation Metrics — {run_dir.name}"
    plt.title(title)

    # Build a shared legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()

    if output is None:
        output = run_dir / "val_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot val L1 and val RMSE dB from TensorBoard logs")
    parser.add_argument("--run_dir", type=str, default="", help="Path to a specific run directory (containing tb/)")
    parser.add_argument("--runs_root", type=str, default="runs", help="Root directory containing runs (used if --run_dir is empty)")
    parser.add_argument("--l1_tag", type=str, default="val/l1_epoch", help="TensorBoard tag for validation L1")
    parser.add_argument(
        "--rmse_tag", type=str, default="val/rmse_db_epoch", help="TensorBoard tag for validation RMSE in dB"
    )
    parser.add_argument("--output", type=str, default="", help="Output PNG path; defaults to <run_dir>/val_metrics.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run_dir(runs_root)

    output_path = Path(args.output).resolve() if args.output else None

    out = plot_val_metrics(run_dir, tags=(args.l1_tag, args.rmse_tag), output=output_path)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


