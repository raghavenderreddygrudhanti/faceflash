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

# ── Style — clean, modern, editorial ─────────────────────────────────────────
INK = "#0F172A"        # near-black for text
SUBINK = "#64748B"     # muted slate for subtitles/secondary
ACCENT = "#10B981"     # FaceFlash — emerald
ACCENT_DK = "#047857"  # darker emerald for emphasis
MUTED = "#CBD5E1"      # competitors — light slate
GRID = "#E2E8F0"       # faint grid
PALETTE = ["#10B981", "#6366F1", "#F59E0B", "#EF4444", "#0EA5E9", "#8B5CF6"]
CARD_BG = "#FFFFFF"

plt.rcParams.update({
    "font.size": 13,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "axes.labelcolor": SUBINK,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "xtick.color": SUBINK,
    "ytick.color": SUBINK,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "figure.facecolor": CARD_BG,
    "axes.facecolor": CARD_BG,
})


def titled(ax, title, subtitle=None):
    """Editorial title: bold headline + muted subtitle, left-aligned."""
    ax.set_title(title, fontweight="bold", loc="left", color=INK, pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=11, color=SUBINK, va="bottom", ha="left")


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    facecolor=CARD_BG, pad_inches=0.25)
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

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars = ax.barh(labels, mem, color=colors, height=0.62,
                   edgecolor="white", linewidth=1.5, zorder=3)
    for bar, (lbl, mb, rec) in zip(bars, rows):
        is_ff = lbl == "FaceFlash"
        ax.text(mb + max(mem) * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{mb:.0f} MB" + ("" if not is_ff else f"   {rec*100:.1f}% recall"),
                va="center", fontsize=11.5,
                fontweight="bold" if is_ff else "medium",
                color=ACCENT_DK if is_ff else SUBINK)
    ax.set_xlabel("Memory to index 100,000 faces (MB) — lower is better")
    titled(ax, "FaceFlash uses ~96× less memory than HNSW",
           "Same recall, a fraction of the RAM · MS1MV2, 100K faces")
    ax.set_xlim(0, max(mem) * 1.20)
    ax.grid(axis="y", visible=False)
    ax.tick_params(left=False)
    save(fig, "chart_memory_bar")


# ─── 2. Recall vs memory pareto (100K) ───────────────────────────────────────
def chart_pareto():
    s = load_ann_scale("100K")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for m in s["methods"]:
        name, rec, mem = m["method"], m["recall_at_1"] * 100, m["memory_mb"]
        is_ff = "FaceFlash" in name
        ax.scatter(mem, rec, s=240 if is_ff else 95,
                   color=ACCENT if is_ff else MUTED,
                   edgecolor="white", linewidth=1.6,
                   zorder=4 if is_ff else 2, alpha=1.0)
    # annotate a few key points — per-label offsets so the crowded
    # top-right corner (FAISS/HNSW/USearch) doesn't overlap
    notable = {"FaceFlash (256b/300c)": ("FaceFlash", (10, 10)),
               "HNSWLIB (ef=128)": ("HNSW", (8, 10)),
               "USearch (HNSW)": ("USearch", (10, -18)),
               "ScaNN (leaves=316)": ("ScaNN", (10, 8)),
               "FAISS-Flat (exact)": ("FAISS-Flat", (-12, 12))}
    for m in s["methods"]:
        if m["method"] in notable:
            lbl, off = notable[m["method"]]
            is_ff = "FaceFlash" in m["method"]
            ax.annotate(lbl, (m["memory_mb"], m["recall_at_1"] * 100),
                        textcoords="offset points", xytext=off, fontsize=10.5,
                        ha="right" if off[0] < 0 else "left",
                        fontweight="bold" if is_ff else "medium",
                        color=ACCENT_DK if is_ff else SUBINK)
    ax.set_xscale("log")
    ax.set_xlabel("Memory (MB, log scale) — left is better")
    ax.set_ylabel("Recall@1 (%) — higher is better")
    titled(ax, "FaceFlash sits alone in the good corner",
           "High recall at tiny memory · MS1MV2, 100K faces")
    ax.set_ylim(78, 101)
    save(fig, "chart_recall_memory_pareto")


# ─── 3. Recall vs candidates, per bit-width, with CI bands ───────────────────
def chart_recall_candidates():
    # Prefer the fresh MS1MV2 grid; fall back to the older VGGFace2 1M run.
    fresh = RES / "bench_nbits_grid.json"
    src = fresh if fresh.exists() else RES / "bench_nbits_grid_1m.json"
    d = json.load(open(src))
    n_db = d.get("n_database") or d.get("database")
    scale_lbl = (f"{n_db/1e6:.0f}M" if n_db and n_db >= 1e6
                 else f"{n_db/1e3:.0f}K" if n_db else "1M")
    dataset = "MS1MV2" if "ms1m" in str(src).lower() or fresh.exists() else "VGGFace2"
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, b in enumerate(d["grid"]):
        x = [c["n_cand"] for c in b["candidates"]]
        y = [c["recall_at1"] * 100 for c in b["candidates"]]
        lo = [c["recall_ci95"][0] * 100 for c in b["candidates"]]
        hi = [c["recall_ci95"][1] * 100 for c in b["candidates"]]
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "-o", color=col, label=f"{b['n_bits']}-bit",
                linewidth=2.4, markersize=6, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
        ax.fill_between(x, lo, hi, color=col, alpha=0.10, zorder=1)
    ax.set_xlabel("Rerank candidates")
    ax.set_ylabel("Recall@1 (%)")
    titled(ax, "More bits or more candidates both buy recall",
           f"{scale_lbl} faces ({dataset}) · bands = 95% CI over {d['n_queries']} queries")
    ax.legend(title="code length", frameon=False, loc="lower right")
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

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(names, vals, color=cols, width=0.62,
                  edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(ceiling, ls="--", color=SUBINK, alpha=0.7, lw=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("Rank-1 identification accuracy (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    titled(ax, f"FaceFlash ties exact search on {d['n_gallery_identities']:,} people",
           "The binary compression is free · MS1MV2, 1:N identification")
    save(fig, "chart_rank1_tie")


# ─── 5. Clustering recall/speed tradeoff (real MS1MV2) ───────────────────────
def chart_clustering():
    d = json.load(open(RES / "bench_clustering.json"))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, (scale, s) in enumerate(d["scales"].items()):
        x = [c["n_probe"] for c in s["clustered"]]
        y = [c["recall_at_1"] * 100 for c in s["clustered"]]
        col = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "-o", color=col, linewidth=2.4, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, label=scale, zorder=3)
        for c in s["clustered"]:
            ax.annotate(f"{c['speedup_vs_full']:.0f}×",
                        (c["n_probe"], c["recall_at_1"] * 100),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, fontweight="medium", color=col)
    ax.axhline(99, ls="--", color=SUBINK, alpha=0.7, lw=1.2)
    ax.text(ax.get_xlim()[1], 99, " 99%", va="center", fontsize=9, color=SUBINK)
    ax.set_xlabel("n_probe (buckets scanned)")
    ax.set_ylabel("Recall@1 (%)")
    titled(ax, "Clustering trades recall for speed",
           "Real MS1MV2 · labels = speedup vs full scan")
    ax.legend(title="database", frameon=False, loc="lower right")
    save(fig, "chart_clustering_tradeoff")


# ─── 6. SIMD scan speed: single vs parallel ──────────────────────────────────
def chart_simd():
    import numpy as np
    d = json.load(open(RES / "bench_simd.json"))
    scales = list(d["scales"].keys())
    single = [d["scales"][s]["single_threaded"]["mean_ms"] for s in scales]
    par = [d["scales"][s]["parallel"]["mean_ms"] for s in scales]
    x = np.arange(len(scales))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    b1 = ax.bar(x - w/2, single, w, label="single thread", color=MUTED,
                edgecolor="white", linewidth=1.5, zorder=3)
    b2 = ax.bar(x + w/2, par, w, label="multi-core", color=ACCENT,
                edgecolor="white", linewidth=1.5, zorder=3)
    for bar, v in zip(b1, single):   # single-thread: value above bar
        ax.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f} ms",
                ha="center", va="bottom", fontsize=10.5, color=SUBINK)
    for i, (bar, v) in enumerate(zip(b2, par)):   # multi-core: value + speedup
        sp = d["scales"][scales[i]]["parallel"].get("speedup_vs_single")
        tag = f"{v:.2f} ms\n{sp:.1f}× faster" if sp else f"{v:.2f} ms"
        ax.text(bar.get_x() + bar.get_width()/2, v, tag,
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=ACCENT_DK)
    ax.set_xticks(x); ax.set_xticklabels(scales)
    ax.set_ylabel("Hamming scan latency (ms) — lower is better")
    ax.set_ylim(0, max(single) * 1.28)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    titled(ax, "Binary-scan speed: single vs multi-core",
           f"{d['arch']} · AVX2 on x86 · NEON on ARM")
    save(fig, "chart_simd_speed")


if __name__ == "__main__":
    print("Generating charts from MS1MV2 result JSONs...")
    # Each chart is independent — a missing/changed JSON shouldn't kill the rest.
    for fn in (chart_memory_bar, chart_pareto, chart_recall_candidates,
               chart_rank1, chart_clustering, chart_simd):
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__} skipped: {e}")
    print(f"Done → {OUT.relative_to(ROOT)}/")
