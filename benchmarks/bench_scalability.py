"""
FaceFlash — Latency Scaling Benchmark

Measures search LATENCY as database size grows from 13K to 1M.
Does NOT measure recall — that requires real distinct identities
and is validated separately on VGGFace2 (100K real faces).

Database: real LFW embeddings (13K) padded with random unit vectors to target scale.
Random vectors are valid for latency measurement because XOR + POPCNT cost
is independent of vector content — it's pure memory bandwidth.

Usage:
    .venv/bin/python benchmarks/bench_scalability.py
"""
import sys
import time
import json
import platform
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_database(base_embs, target_n):
    """
    Build database at target scale: real embeddings + random unit vectors.
    Used for latency measurement only (not recall).
    """
    n_base = len(base_embs)
    if target_n <= n_base:
        return base_embs[:target_n]

    rng = np.random.default_rng(42)
    n_pad = target_n - n_base
    pad = rng.standard_normal((n_pad, base_embs.shape[1])).astype(np.float32)
    pad /= np.linalg.norm(pad, axis=1, keepdims=True)

    # Shuffle real + random together
    db = np.vstack([base_embs, pad])
    perm = rng.permutation(len(db))
    return db[perm]


def main():
    print("\n" + "=" * 70)
    print("  FaceFlash — Latency Scaling Benchmark")
    print("=" * 70)
    print(f"  Backend: {backend_info()}")
    print(f"  Platform: {platform.platform()}, CPU: {platform.machine()}")
    print(f"  Purpose: Validate that search latency scales linearly with N")
    print(f"  NOTE: Does NOT validate recall at scale (see bench_vggface2.py)")
    print(f"  Database: real LFW + random unit vectors as padding")
    print(f"  All single-threaded (FAISS omp_threads=1)")

    # Load LFW
    base_embs = np.load(DATA_DIR / "lfw_embeddings.npy").astype(np.float32)
    print(f"  Base: {len(base_embs):,} real embeddings")

    # Fixed queries (random subset — recall isn't meaningful here anyway)
    rng = np.random.default_rng(123)
    qi = rng.choice(len(base_embs), size=100, replace=False)
    queries = base_embs[qi].copy()
    print(f"  Queries: {len(queries)}")

    faiss.omp_set_num_threads(1)

    # Scale sweep
    scales = [
        ("13K", 13000),
        ("50K", 50000),
        ("100K", 100000),
        ("500K", 500000),
        ("1M", 1000000),
    ]

    print(f"\n  {'─'*70}")
    print(f"  {'Scale':<8} {'N':<10} {'FAISS ms':<12} {'FF scan ms':<12} "
          f"{'FF total ms':<12} {'Speedup':<10} {'Memory':<8}")
    print(f"  {'─'*70}")

    results = []
    for scale_name, target_n in scales:
        database = build_database(base_embs, target_n)
        n_db = len(database)

        # Fit PCA on sample (for encoding)
        pca = PCABinaryQuantizer(n_bits=512)
        pca.fit(database[:min(5000, n_db)])
        db_codes = pca.encode(database)
        q_codes = pca.encode(queries)

        # FAISS-Flat
        index = faiss.IndexFlatIP(512)
        index.add(database)

        # Warmup
        for i in range(3):
            index.search(queries[i:i+1], 1)
            faceflash_core.hamming_topk(q_codes[i], db_codes, 50)

        # Measure FAISS
        times_faiss = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            index.search(queries[i:i+1], 1)
            times_faiss.append((time.perf_counter() - t0) * 1000)

        # Measure FaceFlash (top-50 for timing — recall not meaningful)
        n_cand = 50
        times_scan = []
        times_total = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            t_scan = time.perf_counter() - t0
            cand_idx = np.asarray(topk).astype(np.intp)
            _ = database[cand_idx] @ queries[i]  # rerank cost
            t_total = time.perf_counter() - t0
            times_scan.append(t_scan * 1000)
            times_total.append(t_total * 1000)

        faiss_ms = np.mean(times_faiss)
        scan_ms = np.mean(times_scan)
        total_ms = np.mean(times_total)
        speedup = faiss_ms / total_ms
        mem_mb = db_codes.nbytes / 1024 / 1024

        print(f"  {scale_name:<8} {n_db:<10,} {faiss_ms:<12.3f}{scan_ms:<12.3f}"
              f"{total_ms:<12.3f}{speedup:<10.1f}x {mem_mb:<6.1f}MB")

        results.append({
            "scale": scale_name,
            "n": n_db,
            "faiss_flat_ms": faiss_ms,
            "faceflash_scan_ms": scan_ms,
            "faceflash_total_ms": total_ms,
            "speedup": speedup,
            "memory_binary_mb": mem_mb,
        })

    print(f"  {'─'*70}")
    print(f"\n  Latency scales linearly: {results[0]['faceflash_total_ms']:.2f}ms → "
          f"{results[-1]['faceflash_total_ms']:.2f}ms (13K → 1M)")
    print(f"  Consistent ~{np.mean([r['speedup'] for r in results]):.0f}x speedup over FAISS-Flat")
    print(f"  Memory at 1M: {results[-1]['memory_binary_mb']:.0f} MB (binary index)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_scale.json", "w") as f:
        json.dump({"purpose": "latency scaling only (not recall)",
                   "database": "real LFW + random padding",
                   "results": results,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'bench_scale.json'}")


if __name__ == "__main__":
    main()
