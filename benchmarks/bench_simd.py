"""
SIMD Hamming benchmark — raw binary-scan speed of the Rust kernel.

Measures the AVX2 (x86) / NEON (ARM) Hamming kernel on synthetic packed codes,
single-threaded and parallel, across DB sizes. A NumPy correctness gate runs
first — speed numbers are only reported once the kernel is proven byte-exact
(this self-validates the AVX2 path, which never runs on the ARM dev machine).

Writes results/bench_simd.json so it checks into GitHub alongside the others.

Usage:
    python benchmarks/bench_simd.py [--scales 100K,500K] [--n-bits 512]
"""
import sys
import time
import json
import argparse
import platform
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


def verify_correctness(rng):
    """Assert the SIMD kernel is byte-exact vs NumPy before timing anything."""
    print("  Verifying SIMD output vs NumPy...")
    for nb in (64, 32, 40, 17):  # multiples of 32/8 and awkward remainders
        sdb = rng.integers(0, 256, (1000, nb), dtype=np.uint8)
        sq = rng.integers(0, 256, nb, dtype=np.uint8)
        gt = np.unpackbits(np.bitwise_xor(sdb, sq.reshape(1, -1)), axis=1).sum(axis=1)
        got = np.asarray(core.hamming_distances(sq, sdb))
        assert np.array_equal(gt.astype(np.uint32), got), f"SIMD MISMATCH at nbytes={nb}!"
        print(f"    nbytes={nb:3d}: exact match ✓")
    print("  ✓ SIMD kernel is byte-exact vs NumPy\n")


def _time(fn, db, query, reps=50):
    for _ in range(10):  # warmup
        fn(query, db, 100)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(query, db, 100)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(float(np.mean(times)), 4),
        "p50_ms": round(float(np.median(times)), 4),
        "p95_ms": round(float(np.percentile(times, 95)), 4),
        "qps": round(1000 / float(np.mean(times)), 1),
    }


def run_scale(n_db, n_bits, rng):
    db = rng.integers(0, 256, (n_db, n_bits // 8), dtype=np.uint8)
    query = rng.integers(0, 256, n_bits // 8, dtype=np.uint8)
    single = _time(core.hamming_topk, db, query)
    parallel = _time(core.hamming_topk_parallel, db, query)
    parallel["speedup_vs_single"] = round(single["mean_ms"] / parallel["mean_ms"], 2)
    print(f"  {n_db:>8,} x {n_bits}b  single={single['mean_ms']:.3f}ms "
          f"({single['qps']:.0f} qps)   parallel={parallel['mean_ms']:.3f}ms")
    return {"n_database": n_db, "n_bits": n_bits,
            "single_threaded": single, "parallel": parallel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="100K,500K")
    ap.add_argument("--n-bits", type=int, default=512)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    backend = backend_info()
    print(f"Backend: {backend}")
    print(f"Arch:    {platform.machine()}\n")

    verify_correctness(rng)

    out = {
        "benchmark": "SIMD Hamming kernel (raw binary scan)",
        "backend": backend,
        "arch": platform.machine(),
        "n_bits": args.n_bits,
        "simd_correct_vs_numpy": True,
        "note": "AVX2 on x86_64, NEON on aarch64, scalar fallback otherwise",
        "scales": {},
    }
    for tag in args.scales.split(","):
        tag = tag.strip()
        if tag not in SCALE_MAP:
            continue
        out["scales"][tag] = run_scale(SCALE_MAP[tag], args.n_bits, rng)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bench_simd.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  ✓ Saved: results/bench_simd.json")


if __name__ == "__main__":
    main()
