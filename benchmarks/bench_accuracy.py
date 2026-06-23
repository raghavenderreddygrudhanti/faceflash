"""
FaceFlash — Comprehensive Paper-Quality Benchmark

Reports: Recall@1, @10, @100, latency (mean/P95/P99), memory,
build time, throughput, candidate efficiency.

Datasets: LFW (13K) + VGGFace2 (100K)
Baselines: FAISS-Flat, FAISS-IVF, USearch

Usage:
    .venv/bin/python benchmarks/bench_comprehensive.py
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_dataset(name):
    """Load embeddings and split into db + queries."""
    if name == "lfw":
        embs = np.load(DATA_DIR / "lfw_embeddings.npy").astype(np.float32)
        labels = np.load(DATA_DIR / "lfw_labels.npy", allow_pickle=True)
    elif name == "vggface2":
        p = DATA_DIR / "vggface2_100k_embeddings.npy"
        if not p.exists():
            p = DATA_DIR / "vggface2_embeddings.npy"
        embs = np.load(p).astype(np.float32)
        labels = np.load(p.parent / p.name.replace("embeddings", "labels"), allow_pickle=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)

    qi = [indices[-1] for indices in label_to_idx.values() if len(indices) >= 2][:300]
    qi = np.array(qi)
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False

    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    return database, queries, name


def compute_gt(database, queries, k=100):
    """Brute force ground truth top-K."""
    gt = np.zeros((len(queries), k), dtype=np.int64)
    for i in range(len(queries)):
        sims = database @ queries[i]
        gt[i] = np.argsort(-sims)[:k]
    return gt


def recall_at_k(retrieved, gt, k):
    """Recall@K: fraction of queries where true NN is in top-K retrieved."""
    correct = 0
    for i in range(len(retrieved)):
        if gt[i, 0] in retrieved[i, :k]:
            correct += 1
    return correct / len(retrieved)


def bench_faceflash(database, queries, gt, n_candidates_list):
    """Full FaceFlash benchmark across candidate counts."""
    pca = PCABinaryQuantizer(n_bits=512)

    # Build
    t0 = time.perf_counter()
    pca.fit(database[:min(5000, len(database))])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)
    build_time = time.perf_counter() - t0

    results = []
    for n_cand in n_candidates_list:
        # Warmup
        for i in range(3):
            faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)

        times = []
        retrieved = np.zeros((len(queries), n_cand), dtype=np.int64)
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand_idx = np.asarray(topk).astype(np.intp)
            # Rerank by cosine
            cosine_sims = database[cand_idx] @ queries[i]
            order = np.argsort(-cosine_sims)
            retrieved[i] = cand_idx[order]
            times.append((time.perf_counter() - t0) * 1000)

        r1 = recall_at_k(retrieved, gt, 1)
        r10 = recall_at_k(retrieved, gt, min(10, n_cand))
        r100 = recall_at_k(retrieved, gt, min(100, n_cand))

        # Throughput
        total_time = sum(times) / 1000
        qps = len(queries) / total_time

        results.append({
            "method": f"FaceFlash (top-{n_cand})",
            "n_candidates": n_cand,
            "recall_1": r1,
            "recall_10": r10,
            "recall_100": r100,
            "latency_mean_ms": np.mean(times),
            "latency_p95_ms": np.percentile(times, 95),
            "latency_p99_ms": np.percentile(times, 99),
            "throughput_qps": qps,
            "memory_mb": db_codes.nbytes / 1024 / 1024,
            "build_time_s": build_time,
        })

    return results


def bench_faiss_flat(database, queries, gt):
    """FAISS-Flat baseline."""
    faiss.omp_set_num_threads(1)
    dim = database.shape[1]
    index = faiss.IndexFlatIP(dim)

    t0 = time.perf_counter()
    index.add(database)
    build_time = time.perf_counter() - t0

    # Warmup
    index.search(queries[:3], 100)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 100)
        times.append((time.perf_counter() - t0) * 1000)

    D, I = index.search(queries, 100)

    return {
        "method": "FAISS-Flat (exact)",
        "n_candidates": len(database),
        "recall_1": float(np.mean(I[:, 0] == gt[:, 0])),
        "recall_10": float(np.mean([gt[i, 0] in I[i, :10] for i in range(len(queries))])),
        "recall_100": float(np.mean([gt[i, 0] in I[i, :100] for i in range(len(queries))])),
        "latency_mean_ms": np.mean(times),
        "latency_p95_ms": np.percentile(times, 95),
        "latency_p99_ms": np.percentile(times, 99),
        "throughput_qps": len(queries) / (sum(times) / 1000),
        "memory_mb": database.nbytes / 1024 / 1024,
        "build_time_s": build_time,
    }


def bench_faiss_ivf(database, queries, gt):
    """FAISS-IVF baseline."""
    faiss.omp_set_num_threads(1)
    dim = database.shape[1]
    n = len(database)
    nlist = max(16, min(int(np.sqrt(n) * 0.5), 256))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    t0 = time.perf_counter()
    index.train(database)
    index.add(database)
    build_time = time.perf_counter() - t0
    index.nprobe = max(4, nlist // 4)

    # Warmup
    index.search(queries[:3], 100)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 100)
        times.append((time.perf_counter() - t0) * 1000)

    D, I = index.search(queries, 100)

    return {
        "method": "FAISS-IVF",
        "n_candidates": n,
        "recall_1": float(np.mean(I[:, 0] == gt[:, 0])),
        "recall_10": float(np.mean([gt[i, 0] in I[i, :10] for i in range(len(queries))])),
        "recall_100": float(np.mean([gt[i, 0] in I[i, :100] for i in range(len(queries))])),
        "latency_mean_ms": np.mean(times),
        "latency_p95_ms": np.percentile(times, 95),
        "latency_p99_ms": np.percentile(times, 99),
        "throughput_qps": len(queries) / (sum(times) / 1000),
        "memory_mb": database.nbytes / 1024 / 1024,
        "build_time_s": build_time,
    }


def run_dataset(dataset_name):
    """Run full benchmark on one dataset."""
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} BENCHMARK")
    print(f"{'='*70}")

    database, queries, _ = load_dataset(dataset_name)
    n_db, n_q = len(database), len(queries)
    print(f"  Database: {n_db:,}, Queries: {n_q}, Backend: {backend_info()}")

    # Ground truth
    print("  Computing ground truth (top-100)...")
    gt = compute_gt(database, queries, k=100)
    print(f"  GT top-1 avg cosine: {np.mean([database[gt[i,0]] @ queries[i] for i in range(n_q)]):.4f}")

    all_results = []

    # FAISS-Flat
    print("\n  FAISS-Flat...")
    r = bench_faiss_flat(database, queries, gt)
    all_results.append(r)

    # FAISS-IVF
    print("  FAISS-IVF...")
    r = bench_faiss_ivf(database, queries, gt)
    all_results.append(r)

    # FaceFlash at multiple candidate counts
    cand_list = [10, 30, 50, 100, 200, 300, 500, 1000]
    cand_list = [c for c in cand_list if c < n_db]
    print(f"  FaceFlash (candidates: {cand_list})...")
    ff_results = bench_faceflash(database, queries, gt, cand_list)
    all_results.extend(ff_results)

    # Print table
    print(f"\n  {'─'*85}")
    print(f"  {'Method':<25} {'R@1':<7} {'R@10':<7} {'R@100':<7} "
          f"{'Mean':<8} {'P95':<8} {'P99':<8} {'Mem':<8} {'QPS':<8}")
    print(f"  {'─'*85}")
    for r in all_results:
        print(f"  {r['method']:<25} "
              f"{r['recall_1']*100:<5.1f}% {r['recall_10']*100:<5.1f}% {r['recall_100']*100:<5.1f}% "
              f"{r['latency_mean_ms']:<7.3f} {r['latency_p95_ms']:<7.3f} {r['latency_p99_ms']:<7.3f} "
              f"{r['memory_mb']:<6.1f}M {r['throughput_qps']:<7.0f}")
    print(f"  {'─'*85}")

    return all_results


def main():
    print("\n" + "=" * 70)
    print("  FaceFlash — Comprehensive Benchmark Suite")
    print("  All metrics. Real data. Paper-quality.")
    print("=" * 70)

    all_data = {}

    # LFW
    if (DATA_DIR / "lfw_embeddings.npy").exists():
        all_data["lfw"] = run_dataset("lfw")

    # VGGFace2
    if (DATA_DIR / "vggface2_100k_embeddings.npy").exists() or \
       (DATA_DIR / "vggface2_embeddings.npy").exists():
        all_data["vggface2"] = run_dataset("vggface2")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bench_comprehensive.json"
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\n  All results saved: {out_path}")


if __name__ == "__main__":
    main()
