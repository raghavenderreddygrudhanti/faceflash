"""
Batch-QPS benchmark — single-query throughput vs cache-blocked batched search,
measured on REAL embeddings (no synthetic codes).

Isolates the part the batching optimization actually touches — the binary
Hamming scan — on real MS1MV2 (or whatever DATA_TAG points at). It loads the
real embeddings, quantizes them with the production PCA+ITQ pipeline, then
compares three ways of resolving the SAME workload (m queries, full-DB top-k):

    1. per-query        — loop calling hamming_topk once per query (status quo)
    2. batched-serial   — hamming_topk_batch(..., parallel=False)
    3. batched-parallel — hamming_topk_batch(..., parallel=True), cache-blocked

The cosine rerank is identical on every path, so it is deliberately left out;
this measures scan throughput in isolation. A correctness gate runs first on
real codes: the batched top-k DISTANCES must equal the true k-smallest before
any QPS is reported (ties may reorder indices, so distances are the invariant).

The win is memory-bandwidth driven, so it only appears once the DB exceeds
last-level cache (500K-1M). On a cache-resident DB the numbers converge — that
is expected, not a regression.

Writes results/bench_batch_qps.json so it checks in alongside the others.

Usage:
    python benchmarks/bench_batch_qps.py --data-tag ms1m \
        [--scales 100K,500K,1M] [--n-bits 512] [--queries 1000] \
        [--n-candidates 100]
"""
import sys
import os
import time
import json
import argparse
import platform
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
# Reuse the single-source-of-truth real-data loader + identity-aware split.
from bench_ann_comparison import load_data, split_queries

try:
    import faceflash._core as core
except ImportError:
    import faceflash_core as core

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCALE_MAP = {"100K": 100_000, "200K": 200_000, "300K": 300_000,
             "500K": 500_000, "1M": 1_000_000}

HAS_BATCH = hasattr(core, "hamming_topk_batch")


def cpu_info():
    """Best-effort CPU description + whether AVX-512 VPOPCNTDQ is available.

    VPOPCNTDQ matters here: cache-blocking turns the batched scan compute-bound,
    which re-opens native 512-bit popcount as a follow-on lever (it does NOT
    help the bandwidth-bound single-query path).
    """
    info = {"arch": platform.machine(), "cpu_count": os.cpu_count(),
            "avx512_vpopcntdq": False, "model": platform.processor()}
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        info["avx512_vpopcntdq"] = "avx512_vpopcntdq" in txt
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except (FileNotFoundError, OSError):
        pass  # non-Linux dev machine — flag stays False
    return info


def verify_correctness(db_codes, q_codes, k_values=(1, 10, 100)):
    """Batched top-k DISTANCES must equal the true k-smallest, on REAL codes."""
    print("  Verifying batched output vs ground truth (real codes)...")
    # Keep the gate fast: a real sub-slice of the DB + a handful of queries.
    sub = db_codes[: min(5000, len(db_codes))]
    qs = q_codes[: min(40, len(q_codes))]
    for k in k_values:
        kk = min(k, len(sub))
        for parallel in (True, False):
            flat = np.asarray(core.hamming_topk_batch(qs, sub, kk, parallel)).reshape(len(qs), kk)
            for i in range(len(qs)):
                di = np.asarray(core.hamming_distances(qs[i], sub))
                got = np.sort(di[flat[i]])
                truth = np.sort(di)[:kk]
                assert np.array_equal(got, truth), (
                    f"MISMATCH k={kk} parallel={parallel} query={i}")
        print(f"    k={kk:>3}: exact (parallel+serial) OK")
    print("  Batched outputs are byte-exact on distances\n")


def _qps_per_query(db, q, k, reps):
    m = len(q)
    for _ in range(min(m, 20)):  # warmup
        core.hamming_topk(q[0], db, k)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(m):
            core.hamming_topk(q[i], db, k)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return m / best


def _qps_batch(db, q, k, parallel, reps):
    m = len(q)
    core.hamming_topk_batch(q[:min(m, 32)], db, k, parallel)  # warmup
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        core.hamming_topk_batch(q, db, k, parallel)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return m / best


def run_scale(db_codes, q_codes, n_db, scale_name, n_ids, k):
    db = np.ascontiguousarray(db_codes[:n_db])
    q = np.ascontiguousarray(q_codes)
    m = len(q)
    db_mb = db.nbytes / 1024 / 1024

    reps = 5 if n_db <= 100_000 else (3 if n_db <= 500_000 else 2)

    per_q = _qps_per_query(db, q, k, reps)
    bser = _qps_batch(db, q, k, False, reps)
    bpar = _qps_batch(db, q, k, True, reps)

    print(f"  {scale_name}: {n_db:,} faces ({n_ids:,} identities)  "
          f"DB {db_mb:.1f} MB, {m} real queries")
    print(f"      per-query        : {per_q:>10,.0f} qps")
    print(f"      batched serial   : {bser:>10,.0f} qps   ({bser/per_q:4.1f}x)")
    print(f"      batched parallel : {bpar:>10,.0f} qps   ({bpar/per_q:4.1f}x)")
    return {
        "n_database": n_db,
        "n_identities": n_ids,
        "n_bits": db.shape[1] * 8,
        "db_mb": round(db_mb, 2),
        "queries": m,
        "n_candidates": k,
        "reps": reps,
        "per_query_qps": round(per_q, 1),
        "batched_serial_qps": round(bser, 1),
        "batched_parallel_qps": round(bpar, 1),
        "speedup_serial": round(bser / per_q, 2),
        "speedup_parallel": round(bpar / per_q, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="Batch-QPS benchmark (real embeddings)")
    ap.add_argument("--scales", default="100K,200K,300K,500K,1M")
    ap.add_argument("--n-bits", type=int, default=512)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--n-candidates", type=int, default=100)
    ap.add_argument("--data-tag", default=None,
                    help="Dataset tag to load (e.g. 'ms1m'); sets DATA_TAG")
    args = ap.parse_args()

    if not HAS_BATCH:
        print("ERROR: this build has no hamming_topk_batch — rebuild the Rust "
              "extension (maturin develop --release).")
        sys.exit(1)

    if args.data_tag:
        os.environ["DATA_TAG"] = args.data_tag

    cpu = cpu_info()
    print(f"Backend: {backend_info()}")
    print(f"CPU:     {cpu['model']}  ({cpu['cpu_count']} logical cores)")
    print(f"Arch:    {cpu['arch']}   AVX-512 VPOPCNTDQ: "
          f"{'yes' if cpu['avx512_vpopcntdq'] else 'no'}\n")

    # ── Real embeddings → identity-aware query split → PCA+ITQ codes ──────
    print("  Loading real embeddings...")
    embs, labels = load_data()
    all_db, queries = split_queries(embs, labels, n_queries=args.queries)
    n_ids_total = len(set(str(l) for l in labels))
    print(f"  Database pool: {len(all_db):,} | Queries: {len(queries)} | "
          f"Identities: {n_ids_total:,}")

    print(f"  Fitting PCA+ITQ quantizer (n_bits={args.n_bits})...")
    quant = PCABinaryQuantizer(n_bits=args.n_bits)
    quant.fit(all_db[:5000])
    db_codes = quant.encode(all_db)
    q_codes = quant.encode(queries)
    print(f"  Encoded: db {db_codes.shape}, queries {q_codes.shape}\n")

    verify_correctness(db_codes, q_codes)

    out = {
        "benchmark": "Batch QPS (per-query vs cache-blocked batched scan) — REAL data",
        "backend": backend_info(),
        "cpu": cpu,
        "data_tag": os.environ.get("DATA_TAG", "(default)"),
        "data_identities_total": n_ids_total,
        "n_bits": args.n_bits,
        "queries": len(queries),
        "n_candidates": args.n_candidates,
        "batched_correct_vs_truth": True,
        "note": ("real MS1MV2 codes; scan-only throughput (rerank excluded, "
                 "identical on all paths); bandwidth win appears once DB > LLC"),
        "scales": {},
    }

    ran_any = False
    for tag in args.scales.split(","):
        tag = tag.strip()
        n = SCALE_MAP.get(tag)
        if n is None:
            continue
        if n > len(db_codes):
            print(f"  WARNING: scale {tag} ({n:,}) exceeds available "
                  f"({len(db_codes):,}) — skipping")
            continue
        n_ids = len(set(str(labels[i]) for i in range(min(n, len(labels)))))
        out["scales"][tag] = run_scale(
            db_codes, q_codes, n, tag, n_ids, args.n_candidates)
        ran_any = True

    if not ran_any:
        # Fall back to whatever real data we have.
        n = len(db_codes)
        tag = f"{n // 1000}K"
        n_ids = len(set(str(l) for l in labels))
        print(f"  Falling back to single scale: {tag}")
        out["scales"][tag] = run_scale(db_codes, q_codes, n, tag, n_ids,
                                       args.n_candidates)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bench_batch_qps.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: results/bench_batch_qps.json")


if __name__ == "__main__":
    main()
