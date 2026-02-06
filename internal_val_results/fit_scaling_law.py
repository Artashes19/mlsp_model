"""
Scaling Law Fitting: Generates plot_summary.png showing power law fits.

Model: L(n) = a * n^(-b) + c   (power law with irreducible loss floor)
First two rows (n=1, n=62500) excluded from fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path(__file__).parent
CSVS = sorted(OUT.glob("*.csv"))
TASKS = {p.stem: pd.read_csv(p) for p in CSVS}
COLORS = {"icassp1": "#e63946", "icassp2": "#457b9d", "mlsp002": "#2a9d8f", "mlsp05": "#e9c46a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})


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
    n_pts = len(L)

    ss_tot = np.sum((L - L.mean()) ** 2)

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
        ss_res = np.sum((L - L_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        k = 3
        adj_r2 = 1 - (1 - r2) * (n_pts - 1) / (n_pts - k - 1) if n_pts > k + 1 else r2
        rmse_fit = np.sqrt(ss_res / n_pts)
        mae = np.mean(np.abs(L - L_pred))
        mape = np.mean(np.abs((L - L_pred) / L)) * 100
        ll = -n_pts / 2 * (np.log(2 * np.pi * ss_res / n_pts) + 1)
        aic = 2 * k - 2 * ll
        bic = k * np.log(n_pts) - 2 * ll
        return dict(
            a=popt[0], b=popt[1], c=popt[2],
            a_se=perr[0], b_se=perr[1], c_se=perr[2],
            R2=r2, adj_R2=adj_r2, RMSE_fit=rmse_fit, MAE=mae, MAPE_pct=mape,
            AIC=aic, BIC=bic,
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


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PLOT: 2 subplots (log-x fits + formulas, log-log)
# ══════════════════════════════════════════════════════════════════════════════
n_dense = np.logspace(np.log10(5e4), np.log10(8e6), 500)
FIT_MASK_THR = 62500  # points above this were used for fitting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Subplot 1: All tasks log-x with fits + formulas -------------------------
for name in task_names:
    df = TASKS[name]
    n, L = df["n_samples"].values.astype(float), df["RMSE"].values
    mask = n > FIT_MASK_THR
    p = all_results[name]
    curve = get_curve(name, n_dense)
    if curve is not None:
        ax1.plot(n_dense, curve, color=COLORS[name], alpha=0.7, linewidth=2,
                 label=f"{name}: {formula_str(name)}")
        ax1.axhline(p["c"], color=COLORS[name], linestyle=":", alpha=0.35, linewidth=1)
    ax1.scatter(n[mask], L[mask], color=COLORS[name], s=55, zorder=5,
                edgecolors="k", linewidths=0.5)
ax1.set_xscale("log")
ax1.set_xlabel("Number of training samples (log)")
ax1.set_ylabel("RMSE")
ax1.yaxis.set_major_locator(MultipleLocator(1))
ax1.set_title("Scaling Law Fits")
ax1.legend(fontsize=7.5, loc="upper right")
ax1.grid(True, alpha=0.3, which="both")

# --- Subplot 2: Log-log view -------------------------------------------------
for name in task_names:
    df = TASKS[name]
    n, L = df["n_samples"].values.astype(float), df["RMSE"].values
    mask = n > FIT_MASK_THR
    p = all_results[name]
    curve = get_curve(name, n_dense)
    if curve is not None:
        ax2.plot(n_dense, curve, color=COLORS[name], alpha=0.7, linewidth=2,
                 label=f"{name}  R²={p['R2']:.4f}")
    ax2.scatter(n[mask], L[mask], color=COLORS[name], s=55, zorder=5,
                edgecolors="k", linewidths=0.5)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_minor_formatter(ScalarFormatter())
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.ticklabel_format(axis="y", style="plain")
ax2.set_xlabel("Number of training samples (log)")
ax2.set_ylabel("RMSE (log)")
ax2.set_title("Log-Log View")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which="both")

fig.suptitle("Scaling Law Summary: $L(n) = a \\cdot n^{-b} + c$", fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "plot_summary.png")
plt.close(fig)

print(f"Saved plot_summary.png to {OUT}")
