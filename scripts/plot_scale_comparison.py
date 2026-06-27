"""Generate the multi-scale comparison chart: FaceFlash vs competitors 100K-1M."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent.parent / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "bench_ann_comparison_ms1m.json") as f:
    data = json.load(f)

scales_order = ["100K", "200K", "300K", "500K", "1M"]
scale_labels = []
ff_mem = []
hnsw_mem = []
ff_latency = []
hnsw_latency = []
ff_batched_qps = []
hnsw_batched_qps = []
ff_recall = []
hnsw_recall = []

for scale in scales_order:
    if scale not in data["scales"]:
        continue
    methods = data["scales"][scale]["methods"]
    scale_labels.append(scale)

    # FaceFlash best config (512b, single-query)
    ff = next((m for m in methods if "512b/100c" in m["method"] and "parallel" not in m["method"] and "batched" not in m["method"]), None)
    ff_batched = next((m for m in methods if "batched" in m["method"] and "FaceFlash" in m["method"]), None)
    hnsw = next((m for m in methods if "HNSWLIB (ef=128)" == m["method"]), None)
    hnsw_b = next((m for m in methods if "HNSWLIB (ef=128, batched)" in m["method"]), None)

    ff_mem.append(ff["memory_mb"] if ff else 0)
    hnsw_mem.append(hnsw["memory_mb"] if hnsw else 0)
    ff_latency.append(ff["latency_mean_ms"] if ff else 0)
    hnsw_latency.append(hnsw["latency_mean_ms"] if hnsw else 0)
    ff_batched_qps.append(ff_batched["qps"] if ff_batched else 0)
    hnsw_batched_qps.append(hnsw_b["qps"] if hnsw_b else 0)
    ff_recall.append(ff["recall_at_1"] * 100 if ff else 0)
    hnsw_recall.append(hnsw["recall_at_1"] * 100 if hnsw else 0)

x = np.arange(len(scale_labels))
width = 0.35

# --- Chart 1: Memory comparison across scales ---
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, ff_mem, width, label='FaceFlash (512b)', color='#2dd4bf', edgecolor='white')
bars2 = ax.bar(x + width/2, hnsw_mem, width, label='HNSWLIB (ef=128)', color='#94a3b8', edgecolor='white')

ax.set_xlabel('Database Scale')
ax.set_ylabel('Memory (MB)')
ax.set_title('Memory Usage: FaceFlash vs HNSW (100K to 1M)\nBoth at 100% Recall@1', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scale_labels)
ax.legend()
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# Add ratio labels
for i in range(len(scale_labels)):
    ratio = hnsw_mem[i] / ff_mem[i] if ff_mem[i] > 0 else 0
    ax.annotate(f'{ratio:.0f}x less', xy=(x[i] - width/2, ff_mem[i]),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color='#0d9488', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / "chart_memory_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_memory_scale.svg", bbox_inches='tight')
print(f"  Saved: chart_memory_scale.png")

# --- Chart 2: Batched throughput comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, ff_batched_qps, width, label='FaceFlash (batched)', color='#2dd4bf', edgecolor='white')
bars2 = ax.bar(x + width/2, hnsw_batched_qps, width, label='HNSWLIB (batched)', color='#94a3b8', edgecolor='white')

ax.set_xlabel('Database Scale')
ax.set_ylabel('Throughput (queries/sec)')
ax.set_title('Batched Throughput: FaceFlash vs HNSW (100K to 1M)\nAll cores, 100% Recall@1', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scale_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add ratio labels
for i in range(len(scale_labels)):
    ratio = ff_batched_qps[i] / hnsw_batched_qps[i] if hnsw_batched_qps[i] > 0 else 0
    label = f'{ratio:.1f}x' if ratio >= 1 else f'{1/ratio:.1f}x slower'
    color = '#0d9488' if ratio >= 1 else '#ef4444'
    ax.annotate(label, xy=(x[i] - width/2, ff_batched_qps[i]),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / "chart_throughput_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_throughput_scale.svg", bbox_inches='tight')
print(f"  Saved: chart_throughput_scale.png")

# --- Chart 3: Single-query latency comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(scale_labels, ff_latency, 'o-', color='#2dd4bf', linewidth=2, markersize=8, label='FaceFlash (512b/100c)')
ax.plot(scale_labels, hnsw_latency, 's--', color='#94a3b8', linewidth=2, markersize=8, label='HNSWLIB (ef=128)')

ax.set_xlabel('Database Scale')
ax.set_ylabel('Latency (ms)')
ax.set_title('Single-Query Latency: FaceFlash vs HNSW (100K to 1M)\nFaceFlash is O(N), HNSW is O(log N)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Annotate crossover
ax.axhline(y=0.7, color='#94a3b8', linestyle=':', alpha=0.5)
ax.annotate('HNSW ~0.7ms (constant)', xy=(2, 0.72), fontsize=9, color='#64748b')

plt.tight_layout()
plt.savefig(FIGURES / "chart_latency_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_latency_scale.svg", bbox_inches='tight')
print(f"  Saved: chart_latency_scale.png")

print("\n  All scale comparison charts generated.")
