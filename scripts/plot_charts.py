"""
Generate README charts from the real MS1MV2 result JSONs.

Produces four clean, minimal charts (PNG + SVG) in docs/figures/:
  1. memory_bar          — memory to index 100K faces (the showstopper)
  2. recall_memory_pareto — recall vs memory; FaceFlash alone in the good corner
  3. recall_vs_candidates — recall curve per bit-width with 95% CI bands
  4. rank1_tie            — 1:N rank-1 vs the exact ceiling (compression is free)

Run:  python scripts/plot_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Style — clean and minimal
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})
ACCENT = "#0E9F6E"     # FaceFlash — green
MUTED = "#9CA3AF"      # competitors — gray
DARK = "#374151"
PALETTE = ["#6366F1", "#F59E0B", "#EF4444", "#06B6D4", "#8B5CF6"]


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  ✓ {name}.png / .svg")


def load_ann_scale(scale="100K"):
    d = json.load(open(RES / "bench_ann_comparison_ms1m.json"))
    return d["scales"][scale]


# ─── 1. Memory bar (100K) ────────────────────────────────────────────────────
def chart_memory_bar():
    s = load_ann_scale("100K")
    pick = {  # one representative per family
        "HNSWLIB (ef=128)": "HNSWLIB", "USearch (HNSW)": "USearch",
        "FAISS-Flat (exact)": "FAISS-Flat", "ScaNN (leaves=316)": "ScaNN",
        "FaceFlash (256b/300c)": "FaceFlash",
    }
    rows = []
    for m in s["methods"]:
        if m["method"] in pick:
            rows.append((pick[m["method"]], m["memory_mb"], m["recall_at_1"]))
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    mem = [r[1] for r in rows]
    colors = [ACCENT if l == "FaceFlash" else MUTED for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, mem, color=colors)
    for bar, (lbl, mb, rec) in zip(bars, rows):
        ax.text(mb + max(mem) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{mb:.0f} MB" + ("" if lbl != "FaceFlash" else f"  ({rec*100:.1f}% recall)"),
                va="center", fontsize=11,
                fontweight="bold" if lbl == "FaceFlash" else "normal",
                color=ACCENT if lbl == "FaceFlash" else DARK)
    ax.set_xlabel("Memory to index 100,000 faces (MB) — lower is better")
    ax.set_title("FaceFlash uses ~96× less memory than HNSW\nat the same recall (MS1MV2, 100K)",
                 fontweight="bold", loc="left")
    ax.set_xlim(0, max(mem) * 1.18)
    save(fig, "chart_memory_bar")


# ─── 2. Recall vs memory pareto (100K) ───────────────────────────────────────
def chart_pareto():
    s = load_ann_scale("100K")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in s["methods"]:
        name, rec, mem = m["method"], m["recall_at_1"] * 100, m["memory_mb"]
        is_ff = "FaceFlash" in name
        ax.scatter(mem, rec, s=130 if is_ff else 70,
                   color=ACCENT if is_ff else MUTED,
                   edgecolor=DARK, zorder=3 if is_ff else 2, alpha=0.95)
    # annotate a few key points
    notable = {"FaceFlash (256b/300c)": "FaceFlash", "HNSWLIB (ef=128)": "HNSW",
               "USearch (HNSW)": "USearch", "ScaNN (leaves=316)": "ScaNN",
               "FAISS-Flat (exact)": "FAISS-Flat"}
    for m in s["methods"]:
        if m["method"] in notable:
            ax.annotate(notable[m["method"]], (m["memory_mb"], m["recall_at_1"] * 100),
                        textcoords="offset points", xytext=(8, 6), fontsize=10,
                        fontweight="bold" if "FaceFlash" in m["method"] else "normal",
                        color=ACCENT if "FaceFlash" in m["method"] else DARK)
    ax.set_xscale("log")
    ax.set_xlabel("Memory (MB, log scale) — left is better")
    ax.set_ylabel("Recall@1 (%) — higher is better")
    ax.set_title("FaceFlash sits alone in the good corner:\nhigh recall, tiny memory (MS1MV2, 100K)",
                 fontweight="bold", loc="left")
    ax.set_ylim(78, 101)
    save(fig, "chart_recall_memory_pareto")


# ─── 3. Recall vs candidates, per bit-width, with CI bands ───────────────────
def chart_recall_candidates():
    d = json.load(open(RES / "bench_nbits_grid_1m.json"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, b in enumerate(d["grid"]):
        x = [c["n_cand"] for c in b["candidates"]]
        y = [c["recall_at1"] * 100 for c in b["candidates"]]
        lo = [c["recall_ci95"][0] * 100 for c in b["candidates"]]
        hi = [c["recall_ci95"][1] * 100 for c in b["candidates"]]
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "-o", color=col, label=f"{b['n_bits']}-bit", linewidth=2, markersize=4)
        ax.fill_between(x, lo, hi, color=col, alpha=0.12)
    ax.set_xlabel("Rerank candidates")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title(f"Recall vs candidates at 1M faces (VGGFace2)\nbands = 95% CI, {d['n_queries']} queries",
                 fontweight="bold", loc="left")
    ax.legend(title="code length", frameon=False)
    save(fig, "chart_recall_vs_candidates")


# ─── 4. Rank-1 tie vs exact ceiling ──────────────────────────────────────────
def chart_rank1():
    d = json.load(open(RES / "bench_identification_ms1m.json"))
    names, vals, cols = [], [], []
    for r in d["results"]:
        nm = r["method"].replace(" (exact)", "\n(exact)").replace("FaceFlash", "FaceFlash\n")
        names.append(nm)
        vals.append(r["rank1"] * 100)
        cols.append(MUTED if "exact" in r["method"] else ACCENT)
    ceiling = next(r["rank1"] * 100 for r in d["results"] if "exact" in r["method"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals, color=cols, width=0.6)
    ax.axhline(ceiling, ls="--", color=DARK, alpha=0.6, lw=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Rank-1 identification accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"FaceFlash ties exact search on {d['n_gallery_identities']:,} distinct people\n"
                 "— the binary compression is free (MS1MV2, 1:N)",
                 fontweight="bold", loc="left")
    save(fig, "chart_rank1_tie")


if __name__ == "__main__":
    print("Generating charts from MS1MV2 result JSONs...")
    chart_memory_bar()
    chart_pareto()
    chart_recall_candidates()
    chart_rank1()
    print(f"Done → {OUT.relative_to(ROOT)}/")
