import math
import os

import matplotlib.pyplot as plt
import pandas as pd

# Pre-training samples in increasing order (skipping 63K)
pretraining_samples = [0, 124800, 249600, 449280, 998400, 1996800, 3993600]

# X-axis labels
x_labels = ["0", "125K", "250K", "500K", "1M", "2M", "4M"]

# ICASSP Task 1 checkpoint directories (skipping 63K)
icassp_task1 = [
    "2026-02-03_16-10-10.784625",
    # "2026-02-03_15-44-20.716149",  # 63K - skipped
    "2026-02-03_16-10-10.504235",
    "2026-02-03_16-10-10.548471",
    "2026-02-03_16-10-10.785072",
    "2026-02-03_15-44-20.724854",
    "2026-02-04_08-19-54.191878",
    "2026-02-03_15-44-22.356921",
]

# ICASSP Task 2 checkpoint directories (skipping 63K)
icassp_task2 = [
    "2026-02-03_14-12-42.246996",
    # "2026-02-03_14-12-31.648463",  # 63K - skipped
    "2026-02-03_14-12-31.647847",
    "2026-02-03_14-12-42.252207",
    "2026-02-03_14-12-42.188006",
    "2026-02-03_16-01-22.947431",
    "2026-02-03_14-12-42.189492",
    "2026-02-03_14-12-42.183745",
]

# ICASSP Task 3 checkpoint directories (skipping 63K)
icassp_task3 = [
    "2026-02-04_08-19-54.743152",
    # "2026-02-04_08-19-54.747471",  # 63K - skipped
    "2026-02-04_08-19-54.744595",
    "2026-02-04_08-19-54.810973",
    "2026-02-04_08-19-54.809236",
    "2026-02-04_08-19-54.808990",
    "2026-02-04_08-19-54.746124",
    "2026-02-04_08-19-54.808384",
]

# MLSP checkpoint directories (skipping 63K)
mlsp = [
    "2026-02-03_14-14-22.157466",
    # "2026-02-03_14-12-42.251825",  # 63K - skipped
    "2026-02-03_14-12-42.200486",
    "2026-02-03_16-04-21.588777",
    "2026-02-03_16-04-21.538168",
    "2026-02-03_16-04-24.961390",
    "2026-02-03_14-14-22.161092",
    "2026-02-03_14-14-22.103916",
]

# Kaggle baselines (sqrt of MSE)
kaggle_baseline_task1 = math.sqrt(24.08821)
kaggle_baseline_task2 = math.sqrt(106.22743)
kaggle_baseline_task3 = math.sqrt(144.44216)
kaggle_baseline_mlsp_002 = math.sqrt(18.93175)
kaggle_baseline_mlsp_05 = math.sqrt(13.91998)

# CSV paths
csv_task1 = "/nfs/dgx/raid/iot/preds/indoor/eval/icassp1_2026-02-05_16-11-06.132601.csv"
csv_task2 = "/nfs/dgx/raid/iot/preds/indoor/eval/icassp2_2026-02-05_16-27-03.045418.csv"
csv_task3 = "/nfs/dgx/raid/iot/preds/indoor/eval/icassp3_2026-02-05_20-40-04.748176.csv"
csv_mlsp = "/nfs/dgx/raid/iot/preds/indoor/eval/mlsp_2026-02-05_17-52-42.528549.csv"


def get_best_rmse(
    csv_path: str,
    ckpt_dirs: list,
    dataset_filter: str = None,
) -> tuple:
    """Read CSV and get best RMSE for each checkpoint directory."""
    df = pd.read_csv(csv_path)
    
    # Filter by dataset if specified
    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter]
    
    best_rmse = df.groupby("timestamp")["RMSE"].min()
    
    ckpt_to_samples = dict(zip(ckpt_dirs, pretraining_samples))
    
    x_values = []
    y_values = []
    
    for ckpt_dir in ckpt_dirs:
        if ckpt_dir in best_rmse.index:
            x_values.append(ckpt_to_samples[ckpt_dir])
            y_values.append(best_rmse[ckpt_dir])
    
    return x_values, y_values


def create_plot(
    x_values: list,
    y_values: list,
    kaggle_baseline: float,
    title: str,
    output_prefix: str,
):
    """Create and save a plot with log scale x-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For log scale, divide by 6240 and replace 0 with 1
    x_values_log = [1 if x == 0 else x / 6240 for x in x_values]
    
    # Plot our results
    ax.plot(x_values_log, y_values, marker="o", label="Our Results")
    
    # Add Kaggle baseline horizontal line
    ax.axhline(
        y=kaggle_baseline,
        color="red",
        linestyle="--",
        label=f"Kaggle Baseline ({kaggle_baseline:.2f})",
    )
    
    ax.set_xscale("log")
    
    # Annotate with only RMSE values
    for x, y in zip(x_values_log, y_values):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    
    # Set custom x-axis ticks (only labeled ticks, no minor ticks)
    ax.set_xticks(x_values_log)
    ax.set_xticklabels(x_labels)
    ax.minorticks_off()
    
    # Add Kaggle baseline to y-axis ticks with 2 decimal places
    yticks = list(ax.get_yticks())
    if kaggle_baseline not in yticks:
        yticks.append(kaggle_baseline)
        yticks.sort()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks])
    
    ax.set_xlabel("Pre-training Samples")
    ax.set_ylabel("Best RMSE")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    png_path = os.path.join(output_dir, f"{output_prefix}.png")
    pdf_path = os.path.join(output_dir, f"{output_prefix}.pdf")
    
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# Generate Task 1 plot
x1, y1 = get_best_rmse(csv_task1, icassp_task1)
create_plot(
    x_values=x1,
    y_values=y1,
    kaggle_baseline=kaggle_baseline_task1,
    title="ICASSP Task 1: Best RMSE vs Pre-training Samples",
    output_prefix="icassp1_pretraining",
)

# Generate Task 2 plot
x2, y2 = get_best_rmse(csv_task2, icassp_task2)
create_plot(
    x_values=x2,
    y_values=y2,
    kaggle_baseline=kaggle_baseline_task2,
    title="ICASSP Task 2: Best RMSE vs Pre-training Samples",
    output_prefix="icassp2_pretraining",
)

# Generate Task 3 plot
x3, y3 = get_best_rmse(csv_task3, icassp_task3)
create_plot(
    x_values=x3,
    y_values=y3,
    kaggle_baseline=kaggle_baseline_task3,
    title="ICASSP Task 3: Best RMSE vs Pre-training Samples",
    output_prefix="icassp3_pretraining",
)

# Generate MLSP 0.02% plot
x_mlsp_002, y_mlsp_002 = get_best_rmse(
    csv_path=csv_mlsp,
    ckpt_dirs=mlsp,
    dataset_filter="test_0.02",
)
create_plot(
    x_values=x_mlsp_002,
    y_values=y_mlsp_002,
    kaggle_baseline=kaggle_baseline_mlsp_002,
    title="MLSP 0.02%: Best RMSE vs Pre-training Samples",
    output_prefix="mlsp_002_pretraining",
)

# Generate MLSP 0.5% plot
x_mlsp_05, y_mlsp_05 = get_best_rmse(
    csv_path=csv_mlsp,
    ckpt_dirs=mlsp,
    dataset_filter="test_0.5",
)
create_plot(
    x_values=x_mlsp_05,
    y_values=y_mlsp_05,
    kaggle_baseline=kaggle_baseline_mlsp_05,
    title="MLSP 0.5%: Best RMSE vs Pre-training Samples",
    output_prefix="mlsp_05_pretraining",
)
