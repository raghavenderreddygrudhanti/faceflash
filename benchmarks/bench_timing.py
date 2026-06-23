"""
FaceFlash — Rigorous LFW Benchmark (Paper-Quality)

Measures:
  1. Timing breakdown: binary scan, candidate selection, cosine rerank, total
  2. FAISS comparison: single-query AND batch mode (fair to both)
  3. Recall at multiple candidate counts
  4. All on real ArcFace embeddings from LFW

Methodology:
  - Single-threaded, same process, same machine
  - Warmup runs excluded from measurement
  - 300 queries, each timed independently
  - Reported: mean, median, P95, P99
  - FAISS: IndexFlatIP (inner product, equivalent to cosine for normalized vectors)

Usage:
    cd ~/lang-chain/faceflash
    .venv/bin/python benchmarks/bench_lfw_rigorous.py
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


def load_lfw():
    """Load real LFW embeddings and split into database + queries."""
    embs = np.load(DATA_DIR / "lfw_embeddings.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "lfw_labels.npy", allow_pickle=True)

    # Split: last image per identity (with >=2 images) becomes a query
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(l, []).append(i)

    qi = [indices[-1] for l, indices in label_to_idx.items() if len(indices) >= 2]
    qi = np.array(qi[:300])

    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False

    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    db_labels = labels[np.where(mask)[0]]

    # Ground truth (exact cosine nearest neighbor)
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])

    return database, queries, db_labels, gt


def report_latency(times_ms, label):
    """Print latency statistics."""
    arr = np.array(times_ms)
    print(f"    {label:<24} mean={arr.mean():.4f}ms  "
          f"median={np.median(arr):.4f}ms  "
          f"P95={np.percentile(arr, 95):.4f}ms  "
          f"P99={np.percentile(arr, 99):.4f}ms")
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99))}


def bench_faceflash_breakdown(database, queries, gt, db_codes, q_codes, n_candidates=50):
    """
    FaceFlash with explicit timing breakdown.

    Measures each phase separately:
      Phase 1: Binary Hamming scan (Rust POPCNT)
      Phase 2: Candidate selection (argpartition)
      Phase 3: Cosine rerank (float dot product on candidates)
    """
    n_q = len(queries)

    # Warmup (5 queries, excluded)
    for i in range(min(5, n_q)):
        faceflash_core.hamming_topk(q_codes[i], db_codes, n_candidates)

    times_binary = []
    times_rerank = []
    times_total = []
    correct = 0

    for i in range(n_q):
        # Total timer
        t_total_start = time.perf_counter()

        # Phase 1+2: Binary scan + top-K selection (Rust does both)
        t0 = time.perf_counter()
        topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_candidates)
        cand_idx = np.asarray(topk).astype(np.intp)
        t_binary = time.perf_counter() - t0

        # Phase 3: Cosine rerank
        t0 = time.perf_counter()
        cosine_sims = database[cand_idx] @ queries[i]
        best_local = np.argmax(cosine_sims)
        best_idx = cand_idx[best_local]
        t_rerank = time.perf_counter() - t0

        t_total = time.perf_counter() - t_total_start

        times_binary.append(t_binary * 1000)
        times_rerank.append(t_rerank * 1000)
        times_total.append(t_total * 1000)

        if best_idx == gt[i]:
            correct += 1

    recall = correct / n_q

    print(f"\n  FaceFlash (PCA binary, top-{n_candidates} rerank)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"    Recall@1: {recall*100:.1f}% ({correct}/{n_q})")
    stats_binary = report_latency(times_binary, "Binary scan + topK")
    stats_rerank = report_latency(times_rerank, "Cosine rerank")
    stats_total = report_latency(times_total, "Total (scan+rerank)")
    print(f"    Memory: {db_codes.nbytes/1024:.1f} KB (binary) + "
          f"{database.nbytes/1024/1024:.1f} MB (float, for rerank)")

    return {
        "method": f"FaceFlash (top-{n_candidates})",
        "recall_1": recall,
        "n_candidates": n_candidates,
        "latency_binary_ms": stats_binary,
        "latency_rerank_ms": stats_rerank,
        "latency_total_ms": stats_total,
        "memory_binary_kb": db_codes.nbytes / 1024,
        "memory_float_mb": database.nbytes / 1024 / 1024,
    }


def bench_faiss_flat(database, queries, gt):
    """
    FAISS-Flat baseline — measured both single-query and batch.

    Single-query: per-query latency (fair comparison to FaceFlash per-query)
    Batch: total throughput for all queries at once (FAISS advantage: BLAS)
    """
    n_q = len(queries)
    dim = database.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(database)

    # Warmup
    for i in range(5):
        index.search(queries[i:i+1], 1)

    # Single-query measurement
    times_single = []
    for i in range(n_q):
        t0 = time.perf_counter()
        index.search(queries[i:i+1], 1)
        times_single.append((time.perf_counter() - t0) * 1000)

    # Batch measurement
    t0 = time.perf_counter()
    D_batch, I_batch = index.search(queries, 1)
    batch_total_ms = (time.perf_counter() - t0) * 1000
    batch_per_query_ms = batch_total_ms / n_q

    recall_single = float(np.mean(I_batch[:, 0] == gt))

    print(f"\n  FAISS-Flat (IndexFlatIP, exact)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"    Recall@1: {recall_single*100:.1f}% (exact by definition)")
    stats_single = report_latency(times_single, "Per-query (single)")
    print(f"    {'Batch (all queries)':<24} total={batch_total_ms:.2f}ms  "
          f"per_query={batch_per_query_ms:.4f}ms  "
          f"({n_q} queries)")
    print(f"    Memory: {database.nbytes/1024/1024:.1f} MB")
    print(f"    Note: Batch uses BLAS parallelism (Apple Accelerate)")

    return {
        "method": "FAISS-Flat (IP, exact)",
        "recall_1": recall_single,
        "latency_single_ms": stats_single,
        "latency_batch_per_query_ms": batch_per_query_ms,
        "latency_batch_total_ms": batch_total_ms,
        "memory_mb": database.nbytes / 1024 / 1024,
    }


def main():
    print("\n" + "=" * 70)
    print("  FaceFlash — Rigorous LFW Benchmark")
    print("=" * 70)

    # Environment
    print(f"\n  Environment:")
    print(f"    Platform: {platform.platform()}")
    print(f"    CPU: {platform.processor() or platform.machine()}")
    print(f"    Python: {platform.python_version()}")
    print(f"    NumPy: {np.__version__}")
    print(f"    FAISS: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")
    print(f"    FaceFlash backend: {backend_info()}")
    print(f"    FAISS threads: {faiss.omp_get_max_threads()}")

    # Force single-threaded for fair comparison
    faiss.omp_set_num_threads(1)
    print(f"    FAISS threads (set to): 1 (single-threaded for fair per-query comparison)")

    # Load data
    print(f"\n  Loading LFW embeddings...")
    database, queries, db_labels, gt = load_lfw()
    n_db = len(database)
    n_q = len(queries)
    print(f"    Database: {n_db:,} vectors × 512 dims")
    print(f"    Queries: {n_q} vectors")
    print(f"    Identities: {len(np.unique(db_labels)):,}")

    # Fit PCA quantizer
    print(f"\n  Fitting PCA quantizer (512 bits)...")
    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(database[:5000])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)
    print(f"    Binary index: {db_codes.shape} = {db_codes.nbytes/1024:.1f} KB")

    # ─── Benchmarks ─────────────────────────────────────────────────────
    results = {}

    # FAISS baseline
    results["faiss_flat"] = bench_faiss_flat(database, queries, gt)

    # FaceFlash at multiple candidate counts
    for n_cand in [10, 30, 50, 100, 200]:
        key = f"faceflash_{n_cand}"
        results[key] = bench_faceflash_breakdown(database, queries, gt, db_codes, q_codes, n_cand)

    # ─── Summary Table ──────────────────────────────────────────────────
    print(f"\n\n  {'='*70}")
    print(f"  SUMMARY — LFW Real Data ({n_db:,} faces)")
    print(f"  {'='*70}")
    print(f"  {'Method':<30} {'Recall@1':<10} {'Latency (mean)':<16} {'Memory':<12}")
    print(f"  {'─'*65}")

    # FAISS
    f = results["faiss_flat"]
    print(f"  {'FAISS-Flat (single-query)':<30} {f['recall_1']*100:<8.1f}%  "
          f"{f['latency_single_ms']['mean']:<14.4f}ms  {f['memory_mb']:<10.1f}MB")
    print(f"  {'FAISS-Flat (batch, BLAS)':<30} {f['recall_1']*100:<8.1f}%  "
          f"{f['latency_batch_per_query_ms']:<14.4f}ms  {f['memory_mb']:<10.1f}MB")

    # FaceFlash variants
    for n_cand in [10, 30, 50, 100, 200]:
        r = results[f"faceflash_{n_cand}"]
        mem_str = f"{r['memory_binary_kb']/1024:.2f}MB"
        print(f"  {'FaceFlash (top-' + str(n_cand) + ')':<30} "
              f"{r['recall_1']*100:<8.1f}%  "
              f"{r['latency_total_ms']['mean']:<14.4f}ms  {mem_str:<10}")

    print(f"  {'─'*65}")
    print(f"\n  Notes:")
    print(f"  • All measurements: single-threaded, same process, same hardware")
    print(f"  • FAISS batch uses BLAS (Apple Accelerate) — inherent parallelism")
    print(f"  • FaceFlash binary memory excludes float vectors needed for reranking")
    print(f"  • 'Latency' = binary Hamming scan + candidate selection + cosine rerank")
    print(f"  • Warmup: 5 queries excluded before measurement")

    # ─── Save ───────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "benchmark": "LFW Rigorous",
        "dataset": "LFW (Labeled Faces in the Wild)",
        "n_database": n_db,
        "n_queries": n_q,
        "n_identities": int(len(np.unique(db_labels))),
        "environment": {
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "faiss_threads": 1,
            "faceflash_backend": backend_info(),
        },
        "methodology": {
            "metric": "inner product (cosine for L2-normalized vectors)",
            "faiss_index": "IndexFlatIP",
            "faiss_mode": "single-threaded (omp_set_num_threads=1)",
            "warmup_queries": 5,
            "timing": "time.perf_counter() per query",
            "ground_truth": "exact brute-force cosine argmax",
        },
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = RESULTS_DIR / "bench_lfw_rigorous.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
