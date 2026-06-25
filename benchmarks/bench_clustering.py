"""
FaceFlash — Coarse Clustering (IVF) Benchmark on REAL face embeddings.

Validates `FaceIndex.build_clusters()` on real data (MS1MV2 by default), not
synthetic vectors. For each scale it measures:
  - full-scan recall@1 + latency (the baseline, clustering off)
  - clustered recall@1 + latency at a sweep of n_probe values

This is the honest counterpart to the README "Scaling to Millions" numbers —
real ArcFace embeddings cluster more tightly than random vectors, so recall at a
given n_probe is typically higher here than on synthetic data.

Usage:
    DATA_TAG=ms1m python benchmarks/bench_clustering.py --scales 100K,500K
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.index import FaceIndex, _HAS_RUST
# Reuse the exact data pipeline the ANN benchmark uses (single source of truth)
from benchmarks.bench_ann_comparison import load_data, split_queries

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCALE_MAP = {"100K": 100_000, "500K": 500_000, "1M": 1_000_000}
N_PROBE_SWEEP = [4, 8, 16, 32, 64]
N_CANDIDATES = 100
N_BITS = 512


def _timed_recall(index, queries, gt, n_probe=None, warmup=10):
    for q in queries[:warmup]:
        index.search(q, k=1, n_candidates=N_CANDIDATES, n_probe=n_probe)
    correct, t0 = 0, time.perf_counter()
    for i, q in enumerate(queries):
        r = index.search(q, k=1, n_candidates=N_CANDIDATES, n_probe=n_probe)
        if r and r[0][2] == gt[i]:
            correct += 1
    latency_ms = (time.perf_counter() - t0) / len(queries) * 1000
    return correct / len(queries), latency_ms


def run_scale(embs, labels, n_db, n_queries):
    database, queries = split_queries(embs, labels, n_queries)
    n_db = min(n_db, len(database))
    db = database[:n_db]
    # Ground truth = exact brute-force cosine argmax within this database slice
    gt = np.array([int(np.argmax(db @ q)) for q in queries], dtype=np.intp)

    idx = FaceIndex(dim=db.shape[1], n_bits=N_BITS)
    idx.add_batch(db, [str(i) for i in range(n_db)])
    idx._flush_pending()

    # Baseline: full scan (clustering off)
    full_recall, full_ms = _timed_recall(idx, queries, gt)

    # Build clusters once (~sqrt(N) buckets), then sweep n_probe
    idx.build_clusters()
    n_clusters = idx.cluster.n_clusters
    probes = []
    for p in N_PROBE_SWEEP:
        if p > n_clusters:
            continue
        idx.cluster.n_probe = p
        rec, ms = _timed_recall(idx, queries, gt, n_probe=p)
        probes.append({
            "n_probe": p,
            "recall_at_1": round(float(rec), 4),
            "latency_mean_ms": round(float(ms), 4),
            "speedup_vs_full": round(full_ms / ms, 2) if ms else None,
        })
        print(f"      n_probe={p:<3} recall@1={rec:.3f}  {ms:.3f} ms  "
              f"({full_ms/ms:.1f}x faster)")

    return {
        "n_database": int(n_db),
        "n_queries": int(len(queries)),
        "n_clusters": int(n_clusters),
        "full_scan": {
            "recall_at_1": round(float(full_recall), 4),
            "latency_mean_ms": round(float(full_ms), 4),
        },
        "clustered": probes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="100K,500K")
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()

    print(f"  Rust backend: {_HAS_RUST}")
    embs, labels = load_data()

    out = {
        "benchmark": "coarse clustering (IVF) — real embeddings",
        "rust_backend": _HAS_RUST,
        "n_bits": N_BITS,
        "n_candidates": N_CANDIDATES,
        "note": "full scan vs clustered; n_probe sweeps the recall/speed tradeoff",
        "scales": {},
    }

    for tag in args.scales.split(","):
        tag = tag.strip()
        if tag not in SCALE_MAP:
            continue
        if SCALE_MAP[tag] > len(embs):
            print(f"  Skipping {tag}: only {len(embs):,} embeddings available")
            continue
        print(f"\n  ── Scale {tag} ──")
        out["scales"][tag] = run_scale(embs, labels, SCALE_MAP[tag], args.queries)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bench_clustering.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  ✓ Saved: results/bench_clustering.json")


if __name__ == "__main__":
    main()
