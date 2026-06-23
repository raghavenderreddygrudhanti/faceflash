"""
Tier 2 baselines: Annoy, NMSLIB at 100K VGGFace2.
DiskANN and ScaNN require Linux — noted as future work.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core
import faiss

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_data():
    embs = np.load(DATA_DIR / "vggface2_100k_embeddings.npy").astype(np.float32)
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


def main():
    print("=" * 70)
    print("  Tier 2 Baselines (VGGFace2 100K)")
    print("=" * 70)

    database, queries, gt = load_data()
    n_db, dim = database.shape
    n_q = len(queries)
    print(f"  Database: {n_db:,}, Queries: {n_q}")

    results = []

    # Annoy
    print("\n  [1] Annoy...")
    from annoy import AnnoyIndex
    for n_trees in [50, 100]:
        annoy_idx = AnnoyIndex(dim, 'dot')
        t0 = time.perf_counter()
        for i in range(n_db):
            annoy_idx.add_item(i, database[i])
        annoy_idx.build(n_trees)
        build_time = time.perf_counter() - t0

        # Warmup
        for i in range(5):
            annoy_idx.get_nns_by_vector(queries[i], 1)

        times = []
        correct = 0
        for i in range(n_q):
            t0 = time.perf_counter()
            nns = annoy_idx.get_nns_by_vector(queries[i], 1)
            times.append((time.perf_counter() - t0) * 1000)
            if nns[0] == gt[i]:
                correct += 1

        results.append({
            "method": f"Annoy ({n_trees} trees)",
            "recall_1": correct / n_q,
            "mean_ms": np.mean(times),
            "p99_ms": np.percentile(times, 99),
            "build_s": build_time,
        })
        print(f"    {n_trees} trees: R@1={correct/n_q*100:.1f}%, {np.mean(times):.3f}ms, build={build_time:.1f}s")

    # NMSLIB (careful with memory)
    print("\n  [2] NMSLIB...")
    try:
        import nmslib
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(database)
        t0 = time.perf_counter()
        index.createIndex({'M': 16, 'efConstruction': 100, 'post': 0}, print_progress=False)
        build_time = time.perf_counter() - t0

        for ef in [32, 64]:
            index.setQueryTimeParams({'efSearch': ef})
            times = []
            correct = 0
            for i in range(n_q):
                t0 = time.perf_counter()
                ids, _ = index.knnQuery(queries[i], k=1)
                times.append((time.perf_counter() - t0) * 1000)
                if ids[0] == gt[i]:
                    correct += 1

            results.append({
                "method": f"NMSLIB (ef={ef})",
                "recall_1": correct / n_q,
                "mean_ms": np.mean(times),
                "p99_ms": np.percentile(times, 99),
                "build_s": build_time,
            })
            print(f"    ef={ef}: R@1={correct/n_q*100:.1f}%, {np.mean(times):.3f}ms")
    except Exception as e:
        print(f"    FAILED: {e}")

    # FaceFlash for reference
    print("\n  [3] FaceFlash (reference)...")
    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(database[:5000])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    for n_cand in [100, 300]:
        times = []
        correct = 0
        for i in range(n_q):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand = np.asarray(topk).astype(np.intp)
            best = cand[np.argmax(database[cand] @ queries[i])]
            times.append((time.perf_counter() - t0) * 1000)
            if best == gt[i]:
                correct += 1
        results.append({
            "method": f"FaceFlash (top-{n_cand})",
            "recall_1": correct / n_q,
            "mean_ms": np.mean(times),
            "p99_ms": np.percentile(times, 99),
        })
        print(f"    top-{n_cand}: R@1={correct/n_q*100:.1f}%, {np.mean(times):.3f}ms")

    # Summary
    print(f"\n  {'─'*60}")
    print(f"  {'Method':<25} {'R@1':<8} {'Mean ms':<10} {'P99 ms':<10}")
    print(f"  {'─'*60}")
    for r in results:
        print(f"  {r['method']:<25} {r['recall_1']*100:<6.1f}% "
              f"{r['mean_ms']:<10.3f}{r['p99_ms']:<10.3f}")
    print(f"  {'─'*60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_tier2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_tier2.json'}")


if __name__ == "__main__":
    main()
