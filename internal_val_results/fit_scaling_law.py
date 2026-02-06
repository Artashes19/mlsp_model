"""
Scaling Law Fitting: Generates two plots:
1. full_task_law.png - Power law fits using all available data
2. task_extrapolation.png - Hold out largest n point to test extrapolation

Model: L(n) = a * n^(-b) + c   (power law with irreducible loss floor)
First two rows (n=1, n=62500) excluded from fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path(__file__).parent

# Load only task CSVs (not pretrain_loss or other result files)
TASK_NAMES = ["icassp1", "icassp2", "icassp3", "icass3", "mlsp002", "mlsp05"]
TASKS = {}
for name in TASK_NAMES:
    csv_path = OUT / f"{name}.csv"
    if csv_path.exists():
        TASKS[name] = pd.read_csv(csv_path)

# Rename icass3 -> icassp3 if present
if "icass3" in TASKS and "icassp3" not in TASKS:
    TASKS["icassp3"] = TASKS.pop("icass3")
elif "icass3" in TASKS:
    TASKS.pop("icass3")

COLORS = {"icassp1": "#e63946", "icassp2": "#457b9d", "icassp3": "#6a4c93",
          "mlsp002": "#2a9d8f", "mlsp05": "#e9c46a"}

# Display names for plots (formal naming)
DISPLAY_NAMES = {
    "icassp1": "ICASSP 1",
    "icassp2": "ICASSP 2",
    "icassp3": "ICASSP 3",
    "mlsp002": "MLSP 0.02",
    "mlsp05": "MLSP 0.5"
}

# X-axis tick configuration (matching pretrain style)
TICK_VALS = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
TICK_LABELS = ["100k", "250k", "500k", "1M", "2M", "4M", "8M"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "font.family": "serif",
})

def setup_xticks(ax):
    """Configure x-axis ticks to match pretrain plot style"""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICK_VALS))
    ax.xaxis.set_major_formatter(FixedFormatter(TICK_LABELS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis='x', rotation=45)


# ── Functional form ─────────────────────────────────────────────────────────
def power_law(n, a, b, c):
    """L(n) = a * n^(-b) + c"""
    return a * np.power(n, -b) + c


# ── Fit one task ─────────────────────────────────────────────────────────────
def fit_task(name, df):
    n_all = df["n_samples"].values.astype(float)
    L_all = df["RMSE"].values.astype(float)

    # Exclude first two rows from fitting
    mask = n_all > 62500
    n = n_all[mask]
    L = L_all[mask]

    # Seed with log-linear regression: log(L) ~ log(a) - b*log(n)
    log_n, log_L = np.log(n), np.log(L)
    slope, intercept = np.polyfit(log_n, log_L, 1)
    a_seed = np.exp(intercept)
    b_seed = max(-slope, 0.001)
    c_seed = max(L.min() * 0.5, 0.01)

    try:
        popt, pcov = curve_fit(power_law, n, L,
                               p0=[a_seed, b_seed, c_seed],
                               bounds=([0, 1e-6, 0], [np.inf, 2.0, L.max()]),
                               method="trf", maxfev=10000,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        perr = np.sqrt(np.diag(pcov))
        L_pred = power_law(n, *popt)
        mae = np.mean(np.abs(L - L_pred))
        mape = np.mean(np.abs((L - L_pred) / L)) * 100
        return dict(
            a=popt[0], b=popt[1], c=popt[2],
            a_se=perr[0], b_se=perr[1], c_se=perr[2],
            MAE=mae, MAPE_pct=mape,
        )
    except Exception as e:
        return {"error": str(e)}


# ── Fit all tasks ────────────────────────────────────────────────────────────
all_results = {}
for name, df in TASKS.items():
    all_results[name] = fit_task(name, df)

task_names = sorted(all_results.keys())


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_curve(name, n_arr):
    p = all_results[name]
    if "error" in p:
        return None
    return power_law(n_arr, p["a"], p["b"], p["c"])


def formula_str(name):
    p = all_results[name]
    if "error" in p:
        return ""
    return f"$L(n) = {p['a']:.2f} \\cdot n^{{-{p['b']:.4f}}} + {p['c']:.2f}$"

def display_name(name):
    """Get formal display name for plot labels"""
    return DISPLAY_NAMES.get(name, name)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Full Scaling Law Fits (using all available data)
# ══════════════════════════════════════════════════════════════════════════════
n_dense = np.logspace(np.log10(5e4), np.log10(8e6), 500)
FIT_MASK_THR = 62500  # points above this were used for fitting

fig1, ax = plt.subplots(1, 1, figsize=(11, 8))

for name in task_names:
    df = TASKS[name]
    n, L = df["n_samples"].values.astype(float), df["RMSE"].values
    mask = n > FIT_MASK_THR
    p = all_results[name]
    curve = get_curve(name, n_dense)
    if curve is not None:
        ax.plot(n_dense, curve, color=COLORS[name], alpha=0.7, linewidth=2.5,
                label=f"{display_name(name)}: {formula_str(name)}")
        ax.axhline(p["c"], color=COLORS[name], linestyle=":", alpha=0.35, linewidth=1)
    ax.scatter(n[mask], L[mask], color=COLORS[name], s=55, zorder=5,
               edgecolors="k", linewidths=0.5)

setup_xticks(ax)
ax.set_xlabel("Training samples", fontsize=12)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("Scaling Law Fits (All Data): $L(n) = a \\cdot n^{-b} + c$", fontsize=14, pad=15)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.25, which="both")

fig1.tight_layout()
fig1.savefig(OUT / "full_task_law.png")
plt.close(fig1)

print(f"Saved full_task_law.png to {OUT}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Extrapolation Test (hold out largest n point)
# ══════════════════════════════════════════════════════════════════════════════
# Fit models with largest point held out, then predict it
holdout_results = {}

for name, df in TASKS.items():
    n_all = df["n_samples"].values.astype(float)
    L_all = df["RMSE"].values.astype(float)

    # Exclude first two rows AND largest point from fitting
    mask = n_all > FIT_MASK_THR
    n_fit = n_all[mask]
    L_fit = L_all[mask]

    # Hold out the largest point
    if len(n_fit) > 1:
        holdout_idx = np.argmax(n_fit)
        n_holdout = n_fit[holdout_idx]
        L_holdout = L_fit[holdout_idx]

        n_train = np.delete(n_fit, holdout_idx)
        L_train = np.delete(L_fit, holdout_idx)

        # Fit on training data (without holdout point)
        ss_tot = np.sum((L_train - L_train.mean()) ** 2)
        log_n, log_L = np.log(n_train), np.log(L_train)
        slope, intercept = np.polyfit(log_n, log_L, 1)
        a_seed = np.exp(intercept)
        b_seed = max(-slope, 0.001)
        c_seed = max(L_train.min() * 0.5, 0.01)

        try:
            popt, pcov = curve_fit(power_law, n_train, L_train,
                                   p0=[a_seed, b_seed, c_seed],
                                   bounds=([0, 1e-6, 0], [np.inf, 2.0, L_train.max()]),
                                   method="trf", maxfev=10000,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)

            # Predict holdout point
            L_pred = power_law(n_holdout, *popt)
            pred_error = L_pred - L_holdout
            pct_error = (pred_error / L_holdout) * 100

            holdout_results[name] = {
                'popt': popt,
                'n_holdout': n_holdout,
                'L_holdout': L_holdout,
                'L_pred': L_pred,
                'pred_error': pred_error,
                'pct_error': pct_error,
                'n_train': n_train,
                'L_train': L_train,
                'n_fit': n_fit,  # All points including holdout
                'L_fit': L_fit
            }
        except Exception as e:
            holdout_results[name] = {'exception': str(e)}

# Plot extrapolation test
fig2, ax2 = plt.subplots(1, 1, figsize=(11, 8))

for name in task_names:
    if name in holdout_results and 'exception' not in holdout_results[name]:
        res = holdout_results[name]

        # Dense curve for plotting
        n_plot = np.logspace(np.log10(res['n_train'].min() * 0.9),
                             np.log10(res['n_holdout'] * 1.2), 500)
        L_curve = power_law(n_plot, *res['popt'])

        # Split curve at holdout boundary
        fit_mask = n_plot <= res['n_holdout']
        ext_mask = n_plot >= res['n_holdout']

        # Plot fit curve (solid) and extrapolation (dashed) - only show label once per task
        ax2.plot(n_plot[fit_mask], L_curve[fit_mask], color=COLORS[name],
                alpha=0.8, linewidth=2.5, label=f"{display_name(name)}", zorder=4)
        ax2.plot(n_plot[ext_mask], L_curve[ext_mask], color=COLORS[name],
                alpha=0.8, linewidth=2.5, linestyle="--", zorder=4)

        # Training points (used for fitting) - no label to reduce clutter
        ax2.scatter(res['n_train'], res['L_train'], color=COLORS[name], s=40,
                   alpha=0.6, zorder=5, edgecolors="k", linewidths=0.3)

        # Holdout point (actual) - show with circle marker
        ax2.scatter(res['n_holdout'], res['L_holdout'], color=COLORS[name], s=100,
                   marker='o', zorder=6, edgecolors="k", linewidths=1.5)

        # Predicted holdout point - show with square marker
        ax2.scatter(res['n_holdout'], res['L_pred'], color=COLORS[name], s=80,
                   marker='s', zorder=6, edgecolors="k", linewidths=1)

# Add generic legend entries for markers (once only)
ax2.scatter([], [], color="gray", s=100, marker='o', edgecolors="k", linewidths=1.5,
           label="Actual (held out)")
ax2.scatter([], [], color="gray", s=80, marker='s', edgecolors="k", linewidths=1,
           label="Predicted")

# Build info box with per-task statistics for held-out 4M point
info_lines = ["Held-out 4M errors:"]
for name in sorted(task_names):
    if name in holdout_results and 'exception' not in holdout_results[name]:
        res = holdout_results[name]
        mae = abs(res['pred_error'])
        mape = abs(res['pct_error'])
        info_lines.append(f"{display_name(name)}: MAPE={mape:.2f}%, MAE={mae:.3f}")

if len(info_lines) > 1:
    info_text = "\n".join(info_lines)
    ax2.text(0.98, 0.97, info_text,
             transform=ax2.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="gray", alpha=0.95, linewidth=1.5))

setup_xticks(ax2)
ax2.set_xlabel("Training samples", fontsize=12)
ax2.set_ylabel("RMSE", fontsize=12)
ax2.set_title("Extrapolation Test: Largest Point Held Out", fontsize=14, pad=15)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.25, which="both")

fig2.tight_layout()
fig2.savefig(OUT / "task_extrapolation.png")
plt.close(fig2)

print(f"Saved task_extrapolation.png to {OUT}")

# Print extrapolation results
print("\n" + "=" * 70)
print("EXTRAPOLATION TEST RESULTS")
print("=" * 70)
for name in sorted(holdout_results.keys()):
    if 'exception' not in holdout_results[name]:
        res = holdout_results[name]
        print(f"\n{name}:")
        print(f"  Holdout point: n={res['n_holdout']:.0f}")
        print(f"  Actual RMSE: {res['L_holdout']:.4f}")
        print(f"  Predicted RMSE: {res['L_pred']:.4f}")
        print(f"  Prediction error: {res['pred_error']:+.4f} ({res['pct_error']:+.2f}%)")
print("=" * 70)
