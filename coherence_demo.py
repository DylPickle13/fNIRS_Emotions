"""
Simple Demonstration: What is Coherence?

Think of coherence like measuring how "in sync" two brain regions are.

Imagine two people clapping:
- If they clap at the same rhythm = HIGH coherence
- If they clap randomly = LOW coherence

In brain imaging, coherence tells us if two brain areas are working together
at the same rhythm (frequency). High coherence = teamwork!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import coherence
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Simple example: Two brain regions during a thinking task
fs = 10        # How often we measure (10 times per second)
T = 120        # 2 minutes of data
t = np.linspace(0, T, int(T * fs), endpoint=False)

# Brain Region A: Active during the task (slow rhythm) + some noise
region_A = (np.sin(2 * np.pi * 0.1 * t) +          # 0.1 Hz = slow thinking rhythm (shared)
            0.3 * np.random.randn(len(t)))          # Random brain noise

# Brain Region B: Also active during the same task + different noise
region_B = (0.8 * np.sin(2 * np.pi * 0.1 * t + np.pi/6) +  # Same thinking rhythm (shared)
            0.3 * np.random.randn(len(t)))                   # Different random noise

# Compute coherence (how "in sync" the regions are)
f, Cxy = coherence(region_A, region_B, fs, nperseg=128)

# Create a simple figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('fNIRS Functional Connectivity Example', fontsize=24, fontweight='bold')

# Show the brain signals (first 30 seconds)
time_display = t[:300]  # First 30 seconds
ax1.plot(time_display, region_A[:300], label='Channel 1', linewidth=2, alpha=0.8, color='blue')
ax1.plot(time_display, region_B[:300], label='Channel 2', linewidth=2, alpha=0.8, color='red')
ax1.set_title('fNIRS HbO₂ Time Series', fontsize=20)
ax1.set_xlabel('Time (seconds)', fontsize=16)
ax1.set_ylabel('ΔHbO₂ Concentration (μM)', fontsize=16)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show how "in sync" the regions are at different rhythms
ax2.plot(f, Cxy, 'purple', linewidth=4)
ax2.set_xlim([0, 0.5])  # Focus on slow brain rhythms
ax2.set_ylim([0, 1.1])
ax2.set_title('Coherence Between Channels', fontsize=20)
ax2.set_xlabel('Frequency (Hz)', fontsize=16)
ax2.set_ylabel('Coherence', fontsize=16)
ax2.grid(True, alpha=0.3)

# Find the peak where they're most in sync
peak_idx = np.argmax(Cxy[(f >= 0.05) & (f <= 0.15)])
peak_freq_idx = np.where((f >= 0.05) & (f <= 0.15))[0][peak_idx]
freq_peak = f[peak_freq_idx]
coh_peak = Cxy[peak_freq_idx]

# Point out where they work together
ax2.annotate(f'Strong Functional Connectivity\nCoherence: {coh_peak:.2f}',
            xy=(freq_peak, coh_peak), 
            xytext=(freq_peak, coh_peak + 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=3),
            fontsize=16, fontweight='bold', color='red',
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.9))

plt.tight_layout()
plt.savefig(r'C:\Users\super\OneDrive - Ontario Tech University\fNIRS_Emotions\plots\figures\fNIRS Coherence Example.png', dpi=300, bbox_inches='tight')

# Helper function to check sync level
def get_coherence_at_freq(target_freq, tolerance=0.02):
    """Get how in sync the regions are at a specific rhythm"""
    freq_mask = (f >= target_freq - tolerance) & (f <= target_freq + tolerance)
    if np.any(freq_mask):
        return np.mean(Cxy[freq_mask])
    return 0.0