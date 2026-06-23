"""
Generate publication-quality figures for FaceFlash paper/README.

Produces:
  1. Pareto frontier (recall vs latency, memory as bubble size)
  2. Scaling plot (latency vs database size)
  3. Recall vs candidates curve
  4. Memory comparison bar chart

Usage:
    .venv/bin/python scripts/plot_figures.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGS_DIR = Path(__file__).parent.parent / "docs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def plot_pareto():
    """Recall vs Latency Pareto frontier (100K VGGFace2)."""
    data = [
        ("FAISS-Flat", 100.0, 4.04, 195),
        ("FAISS-IVF", 97.0, 0.66, 195),
        ("HNSWLIB (ef=32)", 99.0, 0.45, 292),
        ("HNSWLIB (ef=64)", 100.0, 0.73, 292),
        ("USearch", 99.0, 0.21, 253),
        ("NMSLIB (ef=64)", 100.0, 0.21, 292),
        ("FaceFlash (100)", 99.5, 0.49, 6),
        ("FaceFlash (300)", 100.0, 0.55, 6),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, recall, latency, memory in data:
        color = '#e74c3c' if 'FaceFlash' in name else '#3498db'
        marker = 'D' if 'FaceFlash' in name else 'o'
        size = max(30, memory * 0.8)
        ax.scatter(latency, recall, s=size, c=color, marker=marker,
                  alpha=0.8, edgecolors='black', linewidths=0.5, zorder=5)
        offset = (0.03, -0.8) if 'FAISS-Flat' in name else (0.03, 0.3)
        ax.annotate(name, (latency, recall), fontsize=8,
                   xytext=offset, textcoords='offset points')

    ax.set_xlabel('Latency (ms) →')
    ax.set_ylabel('Recall@1 (%) ↑')
    ax.set_title('FaceFlash vs Baselines — VGGFace2 100K\n(bubble size = memory)')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(96, 101)
    ax.grid(True, alpha=0.3)
    ax.legend(['FaceFlash (6 MB)', 'Others (195-292 MB)'],
             loc='lower right', markerscale=0.8)

    plt.savefig(FIGS_DIR / 'pareto_recall_latency.png')
    plt.close()
    print(f"  Saved: pareto_recall_latency.png")


def plot_scaling():
    """Latency vs database size."""
    scales = [13000, 50000, 100000, 500000, 1000000]
    ff_total = [0.08, 0.24, 0.46, 2.21, 5.68]
    faiss_flat = [0.54, 2.01, 4.14, 20.3, 39.9]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(scales, faiss_flat, 'o-', color='#3498db', label='FAISS-Flat (exact)', linewidth=2)
    ax.plot(scales, ff_total, 'D-', color='#e74c3c', label='FaceFlash (top-100)', linewidth=2)

    ax.set_xlabel('Database Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Search Latency vs Scale')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(scales)
    ax.set_xticklabels(['13K', '50K', '100K', '500K', '1M'])

    plt.savefig(FIGS_DIR / 'scaling_latency.png')
    plt.close()
    print(f"  Saved: scaling_latency.png")


def plot_recall_vs_candidates():
    """Recall@1 vs number of rerank candidates."""
    # VGGFace2 100K (post-ITQ)
    cands = [10, 30, 50, 100, 200, 300, 500, 1000]
    recall = [94.5, 99.0, 99.3, 99.7, 100.0, 100.0, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(cands, recall, 'D-', color='#e74c3c', linewidth=2, markersize=6)
    ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% target')
    ax.axhline(y=100, color='#3498db', linestyle='--', alpha=0.5, label='Exact (FAISS-Flat)')

    ax.set_xlabel('Rerank Candidates')
    ax.set_ylabel('Recall@1 (%)')
    ax.set_title('FaceFlash Recall vs Candidate Count — VGGFace2 100K')
    ax.set_xscale('log')
    ax.set_ylim(85, 101)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(FIGS_DIR / 'recall_vs_candidates.png')
    plt.close()
    print(f"  Saved: recall_vs_candidates.png")


def plot_memory():
    """Memory comparison bar chart."""
    methods = ['FAISS-Flat', 'FAISS-IVF', 'HNSWLIB', 'USearch', 'NMSLIB', 'FaceFlash']
    memory = [195, 195, 292, 253, 292, 6]
    colors = ['#3498db'] * 5 + ['#e74c3c']

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(methods, memory, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Index Memory at 100K Faces')

    for bar, mem in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{mem} MB', ha='center', fontsize=9)

    ax.set_ylim(0, 340)
    plt.xticks(rotation=15)

    plt.savefig(FIGS_DIR / 'memory_comparison.png')
    plt.close()
    print(f"  Saved: memory_comparison.png")


if __name__ == "__main__":
    print("Generating publication figures...")
    plot_pareto()
    plot_scaling()
    plot_recall_vs_candidates()
    plot_memory()
    print(f"\nAll figures saved to: {FIGS_DIR}")
