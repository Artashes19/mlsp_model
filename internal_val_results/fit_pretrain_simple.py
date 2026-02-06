"""
Simple Pretrain Loss Scaling Law Fitting
- Uses ALL clean points (no log-spaced subsampling)
- Unregularized fit only (3 free parameters)
- Generates: pretrain_extrapolation.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path

OUT = Path(__file__).parent
PT_COLOR = "#d62828"
MIN_N_SAMPLES = 200_000
PT_BOUNDARY = 2e6

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "font.family": "serif",
})

def power_law(n, a, b, c):
    """L(n) = a * n^(-b) + c"""
    return a * np.power(n, -b) + c

def do_fit(n, L):
    log_n, log_L = np.log(n), np.log(L)
    slope, intercept = np.polyfit(log_n, log_L, 1)
    a0 = np.exp(intercept)
    b0 = max(-slope, 0.001)
    c0 = max(L.min() * 0.5, 0.01)

    popt, pcov = curve_fit(power_law, n, L, p0=[a0, b0, c0],
                           bounds=([0, 1e-6, 0], [np.inf, 2.0, L.max()]),
                           method="trf", maxfev=10000,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

# Load data
pretrain_df = pd.read_csv(OUT / "pretrain_loss.csv")
n_raw = pretrain_df["n_samples"].values.astype(float)
if "RMSE" in pretrain_df.columns:
    L_raw = pretrain_df["RMSE"].values.astype(float)
elif "value" in pretrain_df.columns:
    L_raw = pretrain_df["value"].values.astype(float)
else:
    L_raw = pretrain_df["loss"].values.astype(float)

# Clean data
mask = n_raw >= MIN_N_SAMPLES
n_clean = n_raw[mask]
L_clean = L_raw[mask]

print(f"\nPretrain loss scaling law:")
print(f"  Raw data: {len(n_raw)} points")
print(f"  Clean (≥{MIN_N_SAMPLES/1e3:.0f}K): {len(n_clean)} points")

# Dense curve for plotting
n_dense = np.logspace(np.log10(n_clean.min() * 0.9),
                      np.log10(n_clean.max() * 1.3), 500)

PT_TICK_VALS = [200_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000]
PT_TICK_LABELS = ["200k", "500k", "1M", "2M", "3M", "4M"]

def setup_xticks(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(PT_TICK_VALS))
    ax.xaxis.set_major_formatter(FixedFormatter(PT_TICK_LABELS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis='x', rotation=45)

# ══════════════════════════════════════════════════════════════════════════════
# FULL FIT (for comparison in report)
# ══════════════════════════════════════════════════════════════════════════════
popt_full, perr_full = do_fit(n_clean, L_clean)
a_f, b_f, c_f = popt_full

# ══════════════════════════════════════════════════════════════════════════════
# EXTRAPOLATION PLOT (fit ≤2M, predict >2M)
# ══════════════════════════════════════════════════════════════════════════════
fit_mask = n_clean <= PT_BOUNDARY
ext_mask = n_clean > PT_BOUNDARY
n_fit = n_clean[fit_mask]
L_fit = L_clean[fit_mask]
n_ext = n_clean[ext_mask]
L_ext = L_clean[ext_mask]

popt_extrap, perr_extrap = do_fit(n_fit, L_fit)
a_e, b_e, c_e = popt_extrap
curve_extrap = power_law(n_dense, *popt_extrap)

L_ext_pred = power_law(n_ext, *popt_extrap)
ext_errors = L_ext_pred - L_ext
ext_mae = np.mean(np.abs(ext_errors))
ext_mape = np.mean(np.abs(ext_errors / L_ext)) * 100
ext_bias = np.mean(ext_errors)

fit_m = n_dense <= PT_BOUNDARY
ext_m = n_dense >= PT_BOUNDARY

fig2, ax2 = plt.subplots(1, 1, figsize=(11, 7))
ax2.scatter(n_fit, L_fit, s=10, alpha=0.5, color="#457b9d",
            label=f"Fit data (≤2M, n={fit_mask.sum()})", zorder=2)
ax2.scatter(n_ext, L_ext, s=15, alpha=0.6, color="#2a9d8f",
            edgecolors="k", linewidths=0.3,
            label=f"Held-out (>2M, n={ext_mask.sum()})", zorder=3)
ax2.plot(n_dense[fit_m], curve_extrap[fit_m], color=PT_COLOR, linewidth=3,
         label="Fit curve", zorder=4)
ax2.plot(n_dense[ext_m], curve_extrap[ext_m], color=PT_COLOR, linewidth=3,
         linestyle="--", label="Extrapolated", zorder=4)
ax2.axvline(PT_BOUNDARY, color="gray", linestyle=":", alpha=0.5, linewidth=2,
            label="2M boundary")
if c_e > 0.01:
    ax2.axhline(c_e, color=PT_COLOR, linestyle=":", alpha=0.4, linewidth=1.5,
                label=f"Floor (c={c_e:.2f})")

# Add comprehensive info box
sign = "+" if ext_bias >= 0 else ""
info_text = (
    f"Model: $L(n) = {a_e:.1f} \\cdot n^{{-{b_e:.4f}}} + {c_e:.2f}$\n"
    f"Extrapolation MAPE = {ext_mape:.2f}%\n"
    f"MAE = {ext_mae:.3f},  Bias = {sign}{ext_bias:.3f}"
)
ax2.text(0.98, 0.97, info_text,
         transform=ax2.transAxes, fontsize=10, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                   edgecolor="gray", alpha=0.95, linewidth=1.5))

setup_xticks(ax2)
ax2.set_xlabel("Training samples", fontsize=12)
ax2.set_ylabel("Pretrain loss", fontsize=12)
ax2.set_title("Pretrain Loss Scaling Law: Forward Extrapolation", fontsize=14, pad=15)

# Add space on y-axis so legend doesn't cover data
y_min = min(L_clean.min(), curve_extrap.min())
y_max = max(L_clean.max(), curve_extrap.max())
y_range = y_max - y_min
ax2.set_ylim(y_min - 0.05*y_range, y_max + 0.35*y_range)  # Extra space at top for legend

ax2.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
ax2.grid(True, alpha=0.25, which="both")
fig2.tight_layout()
fig2.savefig(OUT / "pretrain_extrapolation.png")
plt.close(fig2)

print(f"\n✓ Saved: {OUT / 'pretrain_extrapolation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PRETRAIN LOSS SCALING LAW (UNREGULARIZED)")
print("=" * 70)

print(f"\nFULL FIT (all {len(n_clean)} clean points):")
print(f"  L(n) = {a_f:.1f} * n^(-{b_f:.6f}) + {c_f:.2f}")
print(f"  Parameters:")
print(f"    a = {a_f:.1f} ± {perr_full[0]:.1f}")
print(f"    b = {b_f:.6f} ± {perr_full[1]:.6f}")
print(f"    c = {c_f:.2f} ± {perr_full[2]:.2f}")

print(f"\nEXTRAPOLATION (fit {fit_mask.sum()} pts ≤2M, predict {ext_mask.sum()} pts >2M):")
print(f"  L(n) = {a_e:.1f} * n^(-{b_e:.6f}) + {c_e:.2f}")
print(f"  MAPE on held-out: {ext_mape:.2f}%")
print(f"  MAE:  {ext_mae:.4f}")
print(f"  Bias: {sign}{ext_bias:.4f}")

print(f"\nFORWARD PREDICTIONS (using full-fit model):")
for ns in [5e6, 8e6, 10e6, 15e6, 20e6, 30e6, 50e6]:
    pred = power_law(ns, *popt_full)
    print(f"  {ns/1e6:>4.0f}M samples: {pred:.3f}")

print("=" * 70)
