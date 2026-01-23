#!/usr/bin/env python3
"""
Visualize RBF (Radial Basis Function) frequency encoding.

Instead of one-hot encoding frequencies [868, 1800, 3500] MHz,
we use Gaussian kernels in log-frequency space for smooth interpolation.
"""
import math
import numpy as np
import matplotlib.pyplot as plt


# Reference frequencies (MHz)
FREQ_CENTERS = [868, 1800, 3500]
LOG_CENTERS = [math.log(f) for f in FREQ_CENTERS]
SIGMA = 0.5  # Bandwidth in log-space


def encode_frequency_rbf(freq_mhz: float, sigma: float = SIGMA) -> list[float]:
    """
    Encode frequency using RBF kernels in log-space.
    
    Args:
        freq_mhz: Input frequency in MHz
        sigma: RBF bandwidth in log-space
    
    Returns:
        List of 3 activations, one per reference frequency
    """
    log_freq = math.log(freq_mhz)
    two_sigma_sq = 2.0 * sigma * sigma
    
    activations = []
    for log_center in LOG_CENTERS:
        dist_sq = (log_freq - log_center) ** 2
        activation = math.exp(-dist_sq / two_sigma_sq)
        # Center and scale to [-1, 1]
        activation = (activation - 0.5) * 2.0
        activations.append(activation)
    
    return activations


def main():
    # Frequency range to visualize
    freqs = np.linspace(100, 7000, 1000)
    
    # Compute RBF activations for each frequency
    activations = np.array([encode_frequency_rbf(f) for f in freqs])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # --- Plot 1: RBF activations vs frequency ---
    ax1 = axes[0]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = [f'{f} MHz' for f in FREQ_CENTERS]
    
    for i in range(3):
        ax1.plot(freqs, activations[:, i], color=colors[i], linewidth=2, label=labels[i])
    
    # Mark the reference frequencies
    for i, fc in enumerate(FREQ_CENTERS):
        ax1.axvline(fc, color=colors[i], linestyle='--', alpha=0.5)
        ax1.scatter([fc], [1.0], color=colors[i], s=100, zorder=5)
    
    ax1.set_xlabel('Frequency (MHz)', fontsize=12)
    ax1.set_ylabel('Channel Activation', fontsize=12)
    ax1.set_title('RBF Frequency Encoding: Channel Activations vs Input Frequency', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 7000)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(0, color='gray', linewidth=0.5)
    
    # --- Plot 2: Log-space view ---
    ax2 = axes[1]
    log_freqs = np.log(freqs)
    
    for i in range(3):
        ax2.plot(log_freqs, activations[:, i], color=colors[i], linewidth=2, label=labels[i])
    
    # Mark the reference frequencies in log-space
    for i, lc in enumerate(LOG_CENTERS):
        ax2.axvline(lc, color=colors[i], linestyle='--', alpha=0.5)
        ax2.scatter([lc], [1.0], color=colors[i], s=100, zorder=5)
    
    # Add frequency labels on top x-axis
    ax2_top = ax2.twiny()
    tick_freqs = [200, 500, 868, 1800, 3500, 7000]
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks([math.log(f) for f in tick_freqs])
    ax2_top.set_xticklabels([f'{f}' for f in tick_freqs])
    ax2_top.set_xlabel('Frequency (MHz)', fontsize=10)
    
    ax2.set_xlabel('log(Frequency)', fontsize=12)
    ax2.set_ylabel('Channel Activation', fontsize=12)
    ax2.set_title('RBF Frequency Encoding in Log-Space (Gaussian kernels)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('freq_rbf_encoding.png', dpi=150, bbox_inches='tight')
    print("Saved: freq_rbf_encoding.png")
    
    # --- Print example values ---
    print("\nExample encodings:")
    print("-" * 50)
    print(f"{'Freq (MHz)':<12} {'Ch0 (868)':<12} {'Ch1 (1800)':<12} {'Ch2 (3500)':<12}")
    print("-" * 50)
    
    for freq in [500, 868, 1200, 1800, 2500, 3500, 5000]:
        rbf = encode_frequency_rbf(freq)
        print(f"{freq:<12} {rbf[0]:+.3f}       {rbf[1]:+.3f}        {rbf[2]:+.3f}")


def comparison_plot():
    """Create a side-by-side comparison of one-hot vs RBF encoding."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    freqs = np.linspace(100, 7000, 1000)
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    # --- One-hot encoding ---
    ax1 = axes[0]
    for i, fc in enumerate(FREQ_CENTERS):
        # One-hot: only active at exact frequency
        y = np.zeros_like(freqs)
        # Show as narrow spike at the center frequency
        mask = np.abs(freqs - fc) < 50
        y[mask] = 1.0
        ax1.fill_between(freqs, y, alpha=0.7, color=colors[i], label=f'{fc} MHz')
        ax1.axvline(fc, color=colors[i], linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Frequency (MHz)', fontsize=12)
    ax1.set_ylabel('Channel Activation', fontsize=12)
    ax1.set_title('One-Hot Encoding\n(discrete, no interpolation)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_xlim(100, 7000)
    ax1.set_ylim(-0.1, 1.2)
    ax1.grid(True, alpha=0.3)
    
    # --- RBF encoding ---
    ax2 = axes[1]
    activations = np.array([encode_frequency_rbf(f) for f in freqs])
    
    for i in range(3):
        ax2.plot(freqs, activations[:, i], color=colors[i], linewidth=2.5, 
                 label=f'{FREQ_CENTERS[i]} MHz')
        ax2.axvline(FREQ_CENTERS[i], color=colors[i], linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Frequency (MHz)', fontsize=12)
    ax2.set_ylabel('Channel Activation', fontsize=12)
    ax2.set_title('RBF Encoding\n(continuous, smooth interpolation)', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.set_xlim(100, 7000)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('freq_onehot_vs_rbf.png', dpi=150, bbox_inches='tight')
    print("Saved: freq_onehot_vs_rbf.png")


if __name__ == "__main__":
    main()
    comparison_plot()
