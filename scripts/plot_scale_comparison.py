"""Generate multi-scale comparison charts: FaceFlash vs ALL competitors 100K-1M."""
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

# Define methods to track (name pattern → display label, color, marker)
METHODS = {
    "FaceFlash (512b/100c)": {"label": "FaceFlash (single)", "color": "#0d9488", "marker": "o", "ls": "-"},
    "FaceFlash (512b/100c, batched)": {"label": "FaceFlash (batched)", "color": "#2dd4bf", "marker": "D", "ls": "-"},
    "HNSWLIB (ef=128)": {"label": "HNSW ef=128", "color": "#6366f1", "marker": "s", "ls": "--"},
    "HNSWLIB (ef=128, batched)": {"label": "HNSW batched", "color": "#a5b4fc", "marker": "s", "ls": ":"},
    "USearch (HNSW)": {"label": "USearch", "color": "#f59e0b", "marker": "^", "ls": "--"},
    "ScaNN": {"label": "ScaNN", "color": "#ef4444", "marker": "v", "ls": "--"},
    "FAISS-IVF (nprobe=64)": {"label": "FAISS-IVF", "color": "#8b5cf6", "marker": "P", "ls": "--"},
}

def find_method(methods_list, pattern):
    """Find a method by exact or partial match."""
    # Try exact match first
    for m in methods_list:
        if m["method"] == pattern:
            return m
    # Partial match for ScaNN (has varying leaf counts)
    for m in methods_list:
        if pattern.lower() in m["method"].lower() and pattern.lower() != "faceflash":
            return m
    return None

# Collect data per method per scale
results = {k: {"latency": [], "memory": [], "recall": [], "qps": []} for k in METHODS}
valid_scales = []

for scale in scales_order:
    if scale not in data["scales"]:
        continue
    valid_scales.append(scale)
    methods_list = data["scales"][scale]["methods"]

    for key, meta in METHODS.items():
        m = find_method(methods_list, key)
        if m:
            results[key]["latency"].append(m.get("latency_mean_ms", 0))
            results[key]["memory"].append(m.get("memory_mb", 0))
            results[key]["recall"].append(m.get("recall_at_1", 0) * 100)
            results[key]["qps"].append(m.get("qps", 0))
        else:
            results[key]["latency"].append(None)
            results[key]["memory"].append(None)
            results[key]["recall"].append(None)
            results[key]["qps"].append(None)

x = np.arange(len(valid_scales))

# --- Chart 1: Memory comparison (bar chart, all competitors) ---
fig, ax = plt.subplots(figsize=(12, 6))
mem_methods = ["FaceFlash (512b/100c)", "ScaNN", "HNSWLIB (ef=128)", "USearch (HNSW)", "FAISS-IVF (nprobe=64)"]
n_bars = len(mem_methods)
width = 0.15
offsets = np.arange(n_bars) - (n_bars - 1) / 2

for i, key in enumerate(mem_methods):
    meta = METHODS[key]
    vals = [v if v else 0 for v in results[key]["memory"]]
    bars = ax.bar(x + offsets[i] * width, vals, width, label=meta["label"], color=meta["color"], alpha=0.85)

ax.set_xlabel('Database Scale', fontsize=11)
ax.set_ylabel('Memory (MB, log scale)', fontsize=11)
ax.set_title('Index Memory: FaceFlash vs All Competitors (100K to 1M)\nAll at their best recall configuration', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(valid_scales)
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "chart_memory_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_memory_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_memory_scale.png")

# --- Chart 2: Single-query latency (line chart, all competitors) ---
fig, ax = plt.subplots(figsize=(11, 6))
latency_methods = ["FaceFlash (512b/100c)", "HNSWLIB (ef=128)", "USearch (HNSW)", "ScaNN", "FAISS-IVF (nprobe=64)"]

for key in latency_methods:
    meta = METHODS[key]
    vals = results[key]["latency"]
    # Filter out None
    xs = [valid_scales[i] for i in range(len(vals)) if vals[i] is not None]
    ys = [v for v in vals if v is not None]
    ax.plot(xs, ys, marker=meta["marker"], linestyle=meta["ls"], color=meta["color"],
            linewidth=2, markersize=8, label=meta["label"])

ax.set_xlabel('Database Scale', fontsize=11)
ax.set_ylabel('Latency (ms, log scale)', fontsize=11)
ax.set_title('Single-Query Latency: All Methods (100K to 1M)\nLower is better — FaceFlash is O(N), graph methods are O(log N)', fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "chart_latency_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_latency_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_latency_scale.png")

# --- Chart 3: Batched throughput (all that have it) ---
fig, ax = plt.subplots(figsize=(11, 6))
batch_methods = ["FaceFlash (512b/100c, batched)", "HNSWLIB (ef=128, batched)"]

for key in batch_methods:
    meta = METHODS[key]
    vals = results[key]["qps"]
    xs = [valid_scales[i] for i in range(len(vals)) if vals[i] is not None and vals[i] > 0]
    ys = [v for v in vals if v is not None and v > 0]
    ax.plot(xs, ys, marker=meta["marker"], linestyle=meta["ls"], color=meta["color"],
            linewidth=2.5, markersize=10, label=meta["label"])

# Also add single-query methods for context
for key in ["HNSWLIB (ef=128)", "USearch (HNSW)", "ScaNN"]:
    meta = METHODS[key]
    vals = results[key]["qps"]
    xs = [valid_scales[i] for i in range(len(vals)) if vals[i] is not None and vals[i] > 0]
    ys = [v for v in vals if v is not None and v > 0]
    ax.plot(xs, ys, marker=meta["marker"], linestyle=meta["ls"], color=meta["color"],
            linewidth=1.5, markersize=6, alpha=0.7, label=f"{meta['label']} (single)")

ax.set_xlabel('Database Scale', fontsize=11)
ax.set_ylabel('Throughput (queries/sec)', fontsize=11)
ax.set_title('Throughput: FaceFlash Batched vs All Competitors (100K to 1M)\nHigher is better — batched lines are thick, single-query are thin', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "chart_throughput_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_throughput_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_throughput_scale.png")

# --- Chart 4: Recall vs Memory (scatter, all scales combined, latest = 500K) ---
fig, ax = plt.subplots(figsize=(11, 6))
# Use 500K scale for this (real data, meaningful scale)
target_scale = "500K" if "500K" in data["scales"] else valid_scales[-1]
methods_500k = data["scales"][target_scale]["methods"]

colors_map = {
    "FaceFlash": "#0d9488",
    "HNSWLIB": "#6366f1",
    "USearch": "#f59e0b",
    "ScaNN": "#ef4444",
    "FAISS": "#8b5cf6",
}

for m in methods_500k:
    name = m["method"]
    recall = m.get("recall_at_1", 0) * 100
    memory = m.get("memory_mb", 0)
    if memory == 0 or recall < 85:
        continue

    # Determine color
    color = "#94a3b8"
    for prefix, c in colors_map.items():
        if prefix in name:
            color = c
            break

    size = 120 if "FaceFlash" in name else 60
    alpha = 1.0 if "FaceFlash" in name else 0.7
    ax.scatter(memory, recall, s=size, c=color, alpha=alpha, edgecolors='white', linewidths=0.5)

    # Label key points
    if any(k in name for k in ["512b/100c", "ef=128", "USearch (HNSW)", "ScaNN"]) and "parallel" not in name and "batched" not in name:
        short = name.split("(")[0].strip() if "FaceFlash" not in name else "FaceFlash"
        ax.annotate(short, (memory, recall), fontsize=8, ha='left', va='bottom',
                    xytext=(5, 3), textcoords='offset points')

ax.set_xlabel('Memory (MB, log scale) — left is better', fontsize=11)
ax.set_ylabel('Recall@1 (%) — higher is better', fontsize=11)
ax.set_title(f'Recall vs Memory at {target_scale}: FaceFlash Dominates the Top-Left Corner', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(84, 101)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "chart_recall_memory_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_recall_memory_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_recall_memory_scale.png")

print("\n  All scale comparison charts generated.")
