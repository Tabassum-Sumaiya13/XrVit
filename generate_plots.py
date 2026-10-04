import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Set style for IEEE paper aesthetics
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

# ==========================================
# 1. Trade-off Plot (Macro AUROC vs Cost)
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)

models = [
    {"name": "CNN alone", "cost": 0.18, "auroc": 0.8263, "ci_low": 0.8200, "ci_high": 0.8317, "color": "#1f77b4", "marker": "s"},
    {"name": "Distilled CNN", "cost": 0.18, "auroc": 0.8339, "ci_low": 0.8283, "ci_high": 0.8387, "color": "#d62728", "marker": "*"},
    {"name": "RAD-DINO", "cost": 1.00, "auroc": 0.8364, "ci_low": 0.8303, "ci_high": 0.8415, "color": "#ff7f0e", "marker": "^"},
    {"name": "Teacher Average", "cost": 1.18, "auroc": 0.8428, "ci_low": 0.8373, "ci_high": 0.8478, "color": "#9467bd", "marker": "D"},
    {"name": "BioViL-T", "cost": 0.50, "auroc": 0.8010, "ci_low": 0.7950, "ci_high": 0.8060, "color": "#7f7f7f", "marker": "o"}
]

for m in models:
    yerr = [[m["auroc"] - m["ci_low"]], [m["ci_high"] - m["auroc"]]]
    size = 120 if m["marker"] == "*" else 70
    ax.errorbar(m["cost"], m["auroc"], yerr=yerr, fmt=m["marker"], color=m["color"], 
                ecolor=m["color"], elinewidth=1.5, capsize=3, markersize=8 if m["marker"] != "*" else 12,
                label=f"{m['name']} ({m['auroc']:.4f})", zorder=4)

# Draw arrow for KD gain
ax.annotate('+0.0076 KD Gain\n(5.7× faster than RAD-DINO)',
            xy=(0.18, 0.8265), xytext=(0.35, 0.8290),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.8, linestyle='--'),
            fontsize=9.5, fontweight='bold', color='#d62728',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe6e6', edgecolor='#d62728', alpha=0.9))

ax.set_xlabel('Relative Inference Cost (× RAD-DINO)', fontweight='bold')
ax.set_ylabel('Macro AUROC (NIH ChestX-ray14 Test)', fontweight='bold')
ax.set_xlim(0.0, 1.35)
ax.set_ylim(0.790, 0.852)
ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
plt.title('Accuracy vs. Inference Cost Trade-off', fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig("figures/tradeoff_plot.png", dpi=300)
plt.savefig("figures/tradeoff_plot.pdf")
plt.close()
print("Saved figures/tradeoff_plot.png")

# ==========================================
# 2. Per-finding Bar Plot
# ==========================================
fig, ax = plt.subplots(figsize=(6.5, 6), dpi=300)

findings = [
    'Infiltration', 'Effusion', 'Atelectasis', 'Pneumothorax', 
    'Consolidation', 'Mass', 'Nodule', 'Pleural Thick.', 
    'Emphysema', 'Cardiomegaly', 'Edema', 'Pneumonia', 
    'Fibrosis', 'Hernia*'
]

cnn_auroc = [0.709, 0.839, 0.790, 0.882, 0.761, 0.846, 0.798, 0.794, 0.936, 0.888, 0.858, 0.727, 0.829, 0.910]
dist_auroc = [0.711, 0.841, 0.796, 0.889, 0.767, 0.851, 0.803, 0.801, 0.939, 0.895, 0.864, 0.735, 0.843, 0.940]
rad_auroc =  [0.709, 0.843, 0.798, 0.904, 0.761, 0.851, 0.803, 0.814, 0.949, 0.898, 0.856, 0.729, 0.849, 0.944]

y = np.arange(len(findings))
height = 0.25

rects1 = ax.barh(y + height, cnn_auroc, height, label='CNN Baseline', color='#1f77b4', alpha=0.85)
rects2 = ax.barh(y, dist_auroc, height, label='Distilled CNN (Ours)', color='#d62728', alpha=0.9)
rects3 = ax.barh(y - height, rad_auroc, height, label='RAD-DINO Teacher', color='#ff7f0e', alpha=0.85)

ax.set_xlabel('Test AUROC', fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(findings, fontweight='medium')
ax.set_xlim(0.68, 0.97)
ax.invert_yaxis()  # top-down
ax.grid(True, linestyle='--', alpha=0.4, axis='x')
ax.legend(loc='lower right', frameon=True, framealpha=0.9)
plt.title('Per-Finding AUROC Comparison', fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig("figures/per_finding_plot.png", dpi=300)
plt.savefig("figures/per_finding_plot.pdf")
plt.close()
print("Saved figures/per_finding_plot.png")
