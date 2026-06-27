"""Generate polished multi-scale comparison charts: FaceFlash vs ALL competitors."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12})

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent.parent / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "bench_ann_comparison_ms1m.json") as f:
    data = json.load(f)

scales_order = ["100K", "200K", "300K", "500K", "1M"]

# Vibrant, distinct colors
COLORS = {
    "FaceFlash": "#10b981",       # emerald green
    "FaceFlash_batch": "#06d6a0", # bright teal
    "HNSW": "#6366f1",            # indigo
    "HNSW_batch": "#a78bfa",      # light purple
    "USearch": "#f59e0b",         # amber
    "ScaNN": "#ef4444",           # red
    "FAISS-IVF": "#8b5cf6",       # violet
    "FAISS-Flat": "#94a3b8",      # gray
}


def find_method(methods_list, pattern):
    for m in methods_list:
        if m["method"] == pattern:
            return m
    # Partial for ScaNN
    if "ScaNN" in pattern:
        for m in methods_list:
            if "ScaNN" in m["method"]:
                return m
    return None


def collect_across_scales(method_key):
    """Collect latency, memory, recall, qps across scales for one method."""
    lat, mem, rec, qps_list = [], [], [], []
    found_scales = []
    for scale in scales_order:
        if scale not in data["scales"]:
            continue
        m = find_method(data["scales"][scale]["methods"], method_key)
        if m:
            lat.append(m.get("latency_mean_ms", 0))
            mem.append(m.get("memory_mb", 0))
            rec.append(m.get("recall_at_1", 0) * 100)
            qps_list.append(m.get("qps", 0))
            found_scales.append(scale)
    return found_scales, lat, mem, rec, qps_list


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Memory (horizontal bar at 500K, all methods, with recall labels)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))
target = "500K"
methods_500k = data["scales"][target]["methods"]

# Pick representative methods
bar_data = []
for name, color in [
    ("FaceFlash (512b/100c)", COLORS["FaceFlash"]),
    ("FaceFlash (256b/300c)", COLORS["FaceFlash_batch"]),
    ("ScaNN", COLORS["ScaNN"]),
    ("FAISS-IVF (nprobe=64)", COLORS["FAISS-IVF"]),
    ("USearch (HNSW)", COLORS["USearch"]),
    ("HNSWLIB (ef=128)", COLORS["HNSW"]),
    ("FAISS-Flat (exact)", COLORS["FAISS-Flat"]),
]:
    m = find_method(methods_500k, name)
    if m:
        bar_data.append((m["method"].replace(" (HNSW)", ""), m["memory_mb"],
                         m["recall_at_1"] * 100, color))

bar_data.sort(key=lambda x: x[1])  # sort by memory ascending

labels = [d[0] for d in bar_data]
mems = [d[1] for d in bar_data]
recalls = [d[2] for d in bar_data]
colors = [d[3] for d in bar_data]

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, mems, color=colors, edgecolor='white', height=0.6)

for i, (mem, recall) in enumerate(zip(mems, recalls)):
    label = f"  {mem:.0f} MB   R@1={recall:.1f}%"
    ax.text(mem + 15, i, label, va='center', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Memory (MB) — lower is better')
ax.set_title(f'Index Memory at {target} Faces (All Competitors)\nFaceFlash: 30 MB at 100% recall vs HNSW: 1,465 MB', fontweight='bold')
ax.set_xlim(0, max(mems) * 1.35)
ax.grid(axis='x', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "chart_memory_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_memory_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_memory_scale.png (500K horizontal bar)")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Latency across scales (line, all methods)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

plot_methods = [
    ("FaceFlash (512b/100c)", "FaceFlash", COLORS["FaceFlash"], "o", "-", 2.5),
    ("HNSWLIB (ef=128)", "HNSW ef=128", COLORS["HNSW"], "s", "--", 2),
    ("USearch (HNSW)", "USearch", COLORS["USearch"], "^", "--", 2),
    ("FAISS-IVF (nprobe=64)", "FAISS-IVF", COLORS["FAISS-IVF"], "P", "--", 1.5),
    ("ScaNN", "ScaNN", COLORS["ScaNN"], "v", "--", 1.5),
]

for key, label, color, marker, ls, lw in plot_methods:
    scales, lat, _, _, _ = collect_across_scales(key)
    if scales:
        ax.plot(scales, lat, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=9, label=label)

ax.set_xlabel('Database Scale')
ax.set_ylabel('Latency (ms)')
ax.set_title('Single-Query Latency Across Scales (All Competitors)\nFaceFlash is O(N) linear scan; graph methods are O(log N)', fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# Annotate FaceFlash win at 100K
ax.annotate('FaceFlash\nfaster here', xy=(0, 0.43), xytext=(0.5, 1.5),
            fontsize=9, color=COLORS["FaceFlash"], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS["FaceFlash"]))

plt.tight_layout()
plt.savefig(FIGURES / "chart_latency_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_latency_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_latency_scale.png")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Throughput (QPS) across scales — batched + single
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

throughput_methods = [
    ("FaceFlash (512b/100c, batched)", "FaceFlash BATCHED", COLORS["FaceFlash"], "D", "-", 3),
    ("HNSWLIB (ef=128, batched)", "HNSW BATCHED", COLORS["HNSW_batch"], "s", "-", 2.5),
    ("FaceFlash (512b/100c)", "FaceFlash (single)", COLORS["FaceFlash"], "o", ":", 1.5),
    ("HNSWLIB (ef=128)", "HNSW (single)", COLORS["HNSW"], "s", ":", 1.5),
    ("USearch (HNSW)", "USearch (single)", COLORS["USearch"], "^", ":", 1.5),
    ("ScaNN", "ScaNN (single)", COLORS["ScaNN"], "v", ":", 1.5),
]

for key, label, color, marker, ls, lw in throughput_methods:
    scales, _, _, _, qps_list = collect_across_scales(key)
    qps_valid = [(s, q) for s, q in zip(scales, qps_list) if q > 0]
    if qps_valid:
        sx, sy = zip(*qps_valid)
        ax.plot(sx, sy, marker=marker, linestyle=ls, color=color,
                linewidth=lw, markersize=8 if lw > 2 else 6, label=label,
                alpha=1.0 if lw > 2 else 0.7)

ax.set_xlabel('Database Scale')
ax.set_ylabel('Throughput (queries/sec, log scale)')
ax.set_title('Throughput: FaceFlash Batched vs All (100K to 1M)\nBold = batched (all cores), dotted = single-query', fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "chart_throughput_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_throughput_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_throughput_scale.png")

# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Recall vs Memory scatter at 500K (all methods, colorful)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
methods_500k = data["scales"]["500K"]["methods"]

for m in methods_500k:
    name = m["method"]
    recall = m.get("recall_at_1", 0) * 100
    memory = m.get("memory_mb", 0)
    if memory == 0 or recall < 85:
        continue

    # Color by family
    if "FaceFlash" in name:
        color = COLORS["FaceFlash"]
        size = 180
        alpha = 1.0
        zorder = 10
    elif "HNSW" in name:
        color = COLORS["HNSW"]
        size = 90
        alpha = 0.8
        zorder = 5
    elif "USearch" in name:
        color = COLORS["USearch"]
        size = 90
        alpha = 0.8
        zorder = 5
    elif "ScaNN" in name:
        color = COLORS["ScaNN"]
        size = 90
        alpha = 0.8
        zorder = 5
    elif "FAISS-IVF" in name:
        color = COLORS["FAISS-IVF"]
        size = 70
        alpha = 0.7
        zorder = 4
    elif "FAISS-Flat" in name:
        color = COLORS["FAISS-Flat"]
        size = 70
        alpha = 0.6
        zorder = 3
    else:
        color = "#cbd5e1"
        size = 50
        alpha = 0.5
        zorder = 2

    ax.scatter(memory, recall, s=size, c=color, alpha=alpha, edgecolors='white',
               linewidths=1, zorder=zorder)

    # Label key points
    if any(k in name for k in ["512b/100c", "256b/300c"]) and "parallel" not in name and "batched" not in name:
        ax.annotate("FaceFlash", (memory, recall), fontsize=9, fontweight='bold',
                    ha='right', color=COLORS["FaceFlash"],
                    xytext=(-8, 3), textcoords='offset points')
    elif name == "HNSWLIB (ef=128)":
        ax.annotate("HNSW", (memory, recall), fontsize=9, color=COLORS["HNSW"],
                    xytext=(5, -12), textcoords='offset points')
    elif "USearch (HNSW)" == name:
        ax.annotate("USearch", (memory, recall), fontsize=9, color=COLORS["USearch"],
                    xytext=(5, -12), textcoords='offset points')
    elif "ScaNN" in name:
        ax.annotate("ScaNN", (memory, recall), fontsize=9, color=COLORS["ScaNN"],
                    xytext=(5, 3), textcoords='offset points')
    elif "FAISS-Flat" in name:
        ax.annotate("FAISS-Flat", (memory, recall), fontsize=9, color=COLORS["FAISS-Flat"],
                    xytext=(5, 3), textcoords='offset points')

# Draw the "good corner" box
ax.axhspan(99, 101, xmin=0, xmax=0.15, alpha=0.08, color=COLORS["FaceFlash"])
ax.text(8, 99.3, "The good corner\n(high recall, low memory)", fontsize=9,
        color=COLORS["FaceFlash"], fontstyle='italic', alpha=0.8)

ax.set_xlabel('Memory (MB, log scale) — left is better')
ax.set_ylabel('Recall@1 (%) — higher is better')
ax.set_title('Recall vs Memory at 500K Faces\nFaceFlash: 100% recall at 30 MB. Competitors need 1,000+ MB.', fontweight='bold')
ax.set_xscale('log')
ax.set_ylim(84, 101)
ax.grid(alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES / "chart_recall_memory_scale.png", dpi=150, bbox_inches='tight')
plt.savefig(FIGURES / "chart_recall_memory_scale.svg", bbox_inches='tight')
plt.close()
print("  Saved: chart_recall_memory_scale.png")

print("\n  All charts generated successfully.")
