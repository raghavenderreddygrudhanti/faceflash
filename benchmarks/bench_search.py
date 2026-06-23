"""
FaceFlash Benchmark — Compare search speed and accuracy vs brute force.

Generates synthetic face embeddings (same distribution as ArcFace)
and benchmarks:
  1. Brute force cosine similarity
  2. FaceFlash binary search (Hamming + rerank)
  3. FaceFlash binary-only (fastest, no rerank)

Run:
    cd ~/lang-chain/faceflash
    .venv/bin/python benchmarks/bench_search.py
"""

import numpy as np
import time
import sys

sys.path.insert(0, ".")
from faceflash.index import FaceIndex
from faceflash.quantize import float_to_binary, pack_binary, hamming_distance_batch


def generate_embeddings(n: int, dim: int = 512) -> np.ndarray:
    """Generate random normalized embeddings (simulates ArcFace output)."""
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # L2 normalize (like real ArcFace embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def brute_force_search(query: np.ndarray, database: np.ndarray, k: int = 1):
    """Brute force cosine similarity search."""
    similarities = np.dot(database, query)
    top_k = np.argsort(-similarities)[:k]
    return top_k, similarities[top_k]


def benchmark_scale(n_faces: int, n_queries: int = 100, dim: int = 512):
    """Run benchmark at a given scale."""
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {n_faces:,} faces, {n_queries} queries, {dim}-dim")
    print(f"{'='*60}")

    # Generate data
    print(f"\n  Generating {n_faces:,} embeddings...")
    database = generate_embeddings(n_faces, dim)
    queries = generate_embeddings(n_queries, dim)

    # Measure memory
    mem_float = database.nbytes / 1024 / 1024
    print(f"  Float memory: {mem_float:.1f} MB")

    # Build FaceFlash index
    print(f"  Building FaceFlash index...")
    index = FaceIndex()
    labels = [f"person_{i}" for i in range(n_faces)]
    start = time.perf_counter()
    for i in range(n_faces):
        index.add(database[i], labels[i])
    build_time = time.perf_counter() - start
    print(f"  Index built in {build_time:.2f}s")

    stats = index.stats()
    print(f"  Binary memory: {stats['binary_memory_mb']:.1f} MB")
    print(f"  Compression: {stats['compression_ratio']:.0f}x")

    # --- Benchmark 1: Brute Force ---
    print(f"\n  [1] Brute Force (cosine similarity)")
    times_bf = []
    for q in queries:
        t0 = time.perf_counter()
        brute_force_search(q, database, k=1)
        times_bf.append(time.perf_counter() - t0)
    avg_bf = np.mean(times_bf) * 1000
    p99_bf = np.percentile(times_bf, 99) * 1000
    print(f"      Avg: {avg_bf:.2f} ms | P99: {p99_bf:.2f} ms")

    # --- Benchmark 2: FaceFlash (binary + rerank) ---
    print(f"\n  [2] FaceFlash (binary + cosine rerank)")
    times_ff = []
    correct_ff = 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        results = index.search(q, k=1)
        times_ff.append(time.perf_counter() - t0)

        # Check accuracy: does FaceFlash find the same top-1 as brute force?
        bf_top, _ = brute_force_search(q, database, k=1)
        if results and results[0][2] == bf_top[0]:
            correct_ff += 1

    avg_ff = np.mean(times_ff) * 1000
    p99_ff = np.percentile(times_ff, 99) * 1000
    recall_ff = correct_ff / n_queries * 100
    print(f"      Avg: {avg_ff:.2f} ms | P99: {p99_ff:.2f} ms | Recall@1: {recall_ff:.1f}%")

    # --- Benchmark 3: FaceFlash binary-only ---
    print(f"\n  [3] FaceFlash (binary-only, no rerank)")
    times_bo = []
    correct_bo = 0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        results = index.search_binary_only(q, k=1)
        times_bo.append(time.perf_counter() - t0)

        bf_top, _ = brute_force_search(q, database, k=1)
        if results and results[0][2] == bf_top[0]:
            correct_bo += 1

    avg_bo = np.mean(times_bo) * 1000
    p99_bo = np.percentile(times_bo, 99) * 1000
    recall_bo = correct_bo / n_queries * 100
    print(f"      Avg: {avg_bo:.2f} ms | P99: {p99_bo:.2f} ms | Recall@1: {recall_bo:.1f}%")

    # --- Summary ---
    print(f"\n  {'─'*56}")
    print(f"  SUMMARY ({n_faces:,} faces)")
    print(f"  {'─'*56}")
    print(f"  {'Method':<30} {'Avg ms':<10} {'Memory':<12} {'Recall@1':<10}")
    print(f"  {'─'*56}")
    print(f"  {'Brute Force':<30} {avg_bf:<10.2f} {mem_float:<10.1f}MB {'100.0%':<10}")
    print(f"  {'FaceFlash (binary+rerank)':<30} {avg_ff:<10.2f} {stats['binary_memory_mb']:<10.1f}MB {recall_ff:<8.1f}%")
    print(f"  {'FaceFlash (binary-only)':<30} {avg_bo:<10.2f} {stats['binary_memory_mb']:<10.1f}MB {recall_bo:<8.1f}%")
    print(f"  {'─'*56}")
    speedup = avg_bf / avg_ff if avg_ff > 0 else 0
    print(f"  Speedup vs brute force: {speedup:.1f}x")
    print(f"  Memory savings: {stats['compression_ratio']:.0f}x smaller")

    return {
        "n_faces": n_faces,
        "brute_force_ms": avg_bf,
        "faceflash_ms": avg_ff,
        "faceflash_binary_ms": avg_bo,
        "recall_rerank": recall_ff,
        "recall_binary": recall_bo,
        "memory_float_mb": mem_float,
        "memory_binary_mb": stats["binary_memory_mb"],
        "speedup": speedup,
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FaceFlash Benchmark Suite")
    print("  Sub-linear face retrieval via binary vector search")
    print("="*60)

    results = []

    # Run at different scales
    for n in [1000, 10000, 100000]:
        r = benchmark_scale(n)
        results.append(r)

    # Final comparison table
    print(f"\n\n{'='*70}")
    print(f"  FINAL RESULTS — FaceFlash vs Brute Force")
    print(f"{'='*70}")
    print(f"  {'Scale':<12} {'BF (ms)':<10} {'FF (ms)':<10} {'Speedup':<10} {'Recall':<10} {'Memory':<10}")
    print(f"  {'─'*65}")
    for r in results:
        print(f"  {r['n_faces']:<12,} {r['brute_force_ms']:<10.2f} {r['faceflash_ms']:<10.2f} {r['speedup']:<10.1f}x {r['recall_rerank']:<8.1f}% {r['memory_binary_mb']:<8.1f}MB")
    print(f"  {'─'*65}")
    print(f"\n  TurboVec-powered binary search. MIT Licensed.")
    print(f"  https://github.com/***REMOVED***rudhanti/faceflash\n")
