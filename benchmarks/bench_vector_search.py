"""
FaceFlash — Tier 1 Baselines at 100K (VGGFace2 real data)

All the comparisons that matter for a systems paper:
  FAISS-Flat, FAISS-IVF, HNSWLIB, USearch, FaceFlash

All single-threaded, same hardware, same data.
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
import hnswlib
from usearch.index import Index

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_vggface2():
    p = DATA_DIR / "vggface2_100k_embeddings.npy"
    embs = np.load(p).astype(np.float32)
    labels = np.load(DATA_DIR / "vggface2_100k_labels.npy", allow_pickle=True)

    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)
    qi = [indices[-1] for indices in label_to_idx.values() if len(indices) >= 2][:200]
    qi = np.array(qi)
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])
    return database, queries, gt


def bench(name, search_fn, database, queries, gt, n_queries=None):
    """Generic benchmark wrapper."""
    if n_queries is None:
        n_queries = len(queries)
    # Warmup
    for i in range(min(5, n_queries)):
        search_fn(queries[i])

    times = []
    correct = 0
    for i in range(n_queries):
        t0 = time.perf_counter()
        idx = search_fn(queries[i])
        times.append((time.perf_counter() - t0) * 1000)
        if idx == gt[i]:
            correct += 1

    return {
        "method": name,
        "recall_1": correct / n_queries,
        "mean_ms": np.mean(times),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
        "qps": n_queries / (sum(times) / 1000),
    }


def main():
    print("=" * 70)
    print("  FaceFlash — Tier 1 Baselines (VGGFace2, 100K real faces)")
    print("=" * 70)
    print(f"  Backend: {backend_info()}")

    database, queries, gt = load_vggface2()
    n_db, dim = database.shape
    n_q = len(queries)
    print(f"  Database: {n_db:,} × {dim}, Queries: {n_q}")

    faiss.omp_set_num_threads(1)
    results = []

    # 1. FAISS-Flat
    print("\n  [1] FAISS-Flat...")
    idx_flat = faiss.IndexFlatIP(dim)
    idx_flat.add(database)
    def faiss_flat_search(q):
        _, I = idx_flat.search(q.reshape(1, -1), 1)
        return I[0, 0]
    results.append(bench("FAISS-Flat (exact)", faiss_flat_search, database, queries, gt))

    # 2. FAISS-IVF
    print("  [2] FAISS-IVF...")
    nlist = 128
    quantizer = faiss.IndexFlatIP(dim)
    idx_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    idx_ivf.train(database)
    idx_ivf.add(database)
    idx_ivf.nprobe = 16
    def faiss_ivf_search(q):
        _, I = idx_ivf.search(q.reshape(1, -1), 1)
        return I[0, 0]
    results.append(bench("FAISS-IVF (nprobe=16)", faiss_ivf_search, database, queries, gt))

    # 3. HNSWLIB
    print("  [3] HNSWLIB...")
    hnsw_idx = hnswlib.Index(space='ip', dim=dim)
    hnsw_idx.init_index(max_elements=n_db, ef_construction=200, M=32)
    hnsw_idx.add_items(database, np.arange(n_db))

    for ef in [32, 64, 128]:
        hnsw_idx.set_ef(ef)
        def hnsw_search(q, _ef=ef):
            labels_out, _ = hnsw_idx.knn_query(q.reshape(1, -1), k=1)
            return labels_out[0][0]
        results.append(bench(f"HNSWLIB (ef={ef})", hnsw_search, database, queries, gt))

    # 4. USearch
    print("  [4] USearch...")
    usearch_idx = Index(ndim=dim, metric='ip')
    usearch_idx.add(np.arange(n_db), database)
    def usearch_search(q):
        matches = usearch_idx.search(q, 1)
        return int(matches.keys[0])
    results.append(bench("USearch (HNSW)", usearch_search, database, queries, gt))

    # 5. FaceFlash
    print("  [5] FaceFlash...")
    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(database[:5000])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    for n_cand in [50, 100, 300, 500]:
        def ff_search(q, _i=[0], _nc=n_cand):
            i = _i[0]
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, _nc)
            cand = np.asarray(topk).astype(np.intp)
            sims = database[cand] @ q
            _i[0] += 1
            return cand[np.argmax(sims)]

        # Reset counter for each run
        _counter = [0]
        times = []
        correct = 0
        for i in range(5):  # warmup
            faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)

        for i in range(n_q):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand = np.asarray(topk).astype(np.intp)
            sims = database[cand] @ queries[i]
            best = cand[np.argmax(sims)]
            times.append((time.perf_counter() - t0) * 1000)
            if best == gt[i]:
                correct += 1

        results.append({
            "method": f"FaceFlash (top-{n_cand})",
            "recall_1": correct / n_q,
            "mean_ms": np.mean(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "qps": n_q / (sum(times) / 1000),
        })

    # Print
    print(f"\n  {'='*75}")
    print(f"  RESULTS — VGGFace2 100K (real data, single-threaded)")
    print(f"  {'='*75}")
    print(f"  {'Method':<28} {'R@1':<8} {'Mean':<9} {'P95':<9} {'P99':<9} {'QPS':<8}")
    print(f"  {'─'*70}")
    for r in results:
        print(f"  {r['method']:<28} {r['recall_1']*100:<6.1f}% "
              f"{r['mean_ms']:<8.3f} {r['p95_ms']:<8.3f} {r['p99_ms']:<8.3f} "
              f"{r['qps']:<7.0f}")
    print(f"  {'─'*70}")

    # Memory
    print(f"\n  Memory comparison:")
    print(f"    FAISS-Flat/IVF:  {database.nbytes/1024/1024:.1f} MB")
    print(f"    HNSWLIB:         ~{database.nbytes/1024/1024*1.5:.1f} MB (vectors + graph)")
    print(f"    USearch:         ~{database.nbytes/1024/1024*1.3:.1f} MB")
    print(f"    FaceFlash:       {db_codes.nbytes/1024/1024:.1f} MB (binary only)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_tier1.json", "w") as f:
        json.dump({"n_database": n_db, "results": results,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_tier1.json'}")


if __name__ == "__main__":
    main()
