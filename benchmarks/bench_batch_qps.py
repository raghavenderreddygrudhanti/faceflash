"""
Batch-QPS benchmark — single-query throughput vs cache-blocked batched search.

Isolates the part the batching optimization actually touches: the binary
Hamming scan. It compares three ways of resolving the SAME workload (m queries,
full-DB top-k) on synthetic packed codes:

    1. per-query     — loop calling hamming_topk once per query (the status quo)
    2. batched-serial — hamming_topk_batch(..., parallel=False)
    3. batched-parallel — hamming_topk_batch(..., parallel=True), cache-blocked

The cosine rerank is identical on every path, so it is deliberately left out;
this measures the scan throughput in isolation. A correctness gate runs first:
the batched top-k DISTANCES must equal the true k-smallest before any QPS is
reported (ties may reorder indices, so distances are the invariant).

The win is memory-bandwidth driven, so it only appears once the DB exceeds
last-level cache (500K-1M on x86). On a cache-resident DB (small N, or a big
desktop LLC) the numbers converge — that is expected, not a regression.

Writes results/bench_batch_qps.json so it checks in alongside the others.

Usage:
    python benchmarks/bench_batch_qps.py [--scales 100K,500K,1M] \
        [--n-bits 512] [--queries 1000] [--n-candidates 100]
"""
import sys
import time
import json
import argparse
import platform
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import backend_info

try:
    import faceflash._core as core
except ImportError:
    import faceflash_core as core

RESULTS_DIR = Path(__file__).parent.parent / "results"
SCALE_MAP = {"100K": 100_000, "500K": 500_000, "1M": 1_000_000}

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
        # Linux: read flags from /proc/cpuinfo
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        info["avx512_vpopcntdq"] = "avx512_vpopcntdq" in txt
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
    except (FileNotFoundError, OSError):
        pass  # non-Linux (e.g. macOS dev machine) — flag stays False
    return info


def verify_correctness(rng):
    """Batched top-k DISTANCES must equal the true k-smallest before timing."""
    print("  Verifying batched output vs ground truth...")
    cases = [
        # (n_db, code_bytes, m, k)
        (5000, 64, 50, 10),
        (8000, 32, 40, 100),
        (4000, 48, 64, 1),     # k=1
        (3000, 64, 1, 16),     # m=1
        (2000, 17, 33, 7),     # awkward code length, non-multiple of 8
    ]
    for n, cb, m, k in cases:
        db = rng.integers(0, 256, (n, cb), dtype=np.uint8)
        q = rng.integers(0, 256, (m, cb), dtype=np.uint8)
        for parallel in (True, False):
            flat = np.asarray(core.hamming_topk_batch(q, db, k, parallel)).reshape(m, k)
            for i in range(m):
                di = np.asarray(core.hamming_distances(q[i], db))
                got = np.sort(di[flat[i]])
                truth = np.sort(di)[:k]
                assert np.array_equal(got, truth), (
                    f"MISMATCH n={n} cb={cb} m={m} k={k} parallel={parallel} query={i}")
        print(f"    n={n:>5} code={cb:>3}B m={m:>3} k={k:>3}: exact (parallel+serial) OK")
    print("  All batched outputs are byte-exact on distances\n")


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


def run_scale(n_db, n_bits, m, k, rng):
    code_bytes = n_bits // 8
    db = rng.integers(0, 256, (n_db, code_bytes), dtype=np.uint8)
    q = rng.integers(0, 256, (m, code_bytes), dtype=np.uint8)
    db_mb = db.nbytes / 1024 / 1024

    # Fewer reps as the DB grows (each pass is more expensive).
    reps = 5 if n_db <= 100_000 else (3 if n_db <= 500_000 else 2)

    per_q = _qps_per_query(db, q, k, reps)
    bser = _qps_batch(db, q, k, False, reps)
    bpar = _qps_batch(db, q, k, True, reps)

    print(f"  {n_db:>9,} x {n_bits}b  (DB {db_mb:6.1f} MB)")
    print(f"      per-query        : {per_q:>10,.0f} qps")
    print(f"      batched serial   : {bser:>10,.0f} qps   ({bser/per_q:4.1f}x)")
    print(f"      batched parallel : {bpar:>10,.0f} qps   ({bpar/per_q:4.1f}x)")
    return {
        "n_database": n_db,
        "n_bits": n_bits,
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
    ap = argparse.ArgumentParser(description="Batch-QPS benchmark")
    ap.add_argument("--scales", default="100K,500K,1M")
    ap.add_argument("--n-bits", type=int, default=512)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--n-candidates", type=int, default=100)
    args = ap.parse_args()

    if not HAS_BATCH:
        print("ERROR: this build has no hamming_topk_batch — rebuild the Rust "
              "extension (maturin develop --release).")
        sys.exit(1)

    rng = np.random.default_rng(42)
    cpu = cpu_info()
    print(f"Backend: {backend_info()}")
    print(f"CPU:     {cpu['model']}  ({cpu['cpu_count']} logical cores)")
    print(f"Arch:    {cpu['arch']}   AVX-512 VPOPCNTDQ: "
          f"{'yes' if cpu['avx512_vpopcntdq'] else 'no'}\n")

    verify_correctness(rng)

    out = {
        "benchmark": "Batch QPS (per-query vs cache-blocked batched scan)",
        "backend": backend_info(),
        "cpu": cpu,
        "n_bits": args.n_bits,
        "queries": args.queries,
        "n_candidates": args.n_candidates,
        "batched_correct_vs_truth": True,
        "note": ("scan-only throughput (rerank excluded, identical on all paths); "
                 "bandwidth win appears once DB > last-level cache"),
        "scales": {},
    }
    for tag in args.scales.split(","):
        tag = tag.strip()
        if tag not in SCALE_MAP:
            continue
        out["scales"][tag] = run_scale(
            SCALE_MAP[tag], args.n_bits, args.queries, args.n_candidates, rng)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bench_batch_qps.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved: results/bench_batch_qps.json")


if __name__ == "__main__":
    main()
