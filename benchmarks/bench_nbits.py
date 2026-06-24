"""
FaceFlash — n_bits Sweep

Tests different binary code lengths to find the optimal tradeoff:
  128 bits (16 bytes) — smallest, fastest scan, lowest recall
  256 bits (32 bytes) — half memory, good speed
  384 bits (48 bytes) — balanced
  512 bits (64 bytes) — current default, best recall

Question: Can we use fewer bits and still maintain high recall?
If yes, that halves memory AND doubles scan speed.

Usage:
    .venv/bin/python benchmarks/bench_nbits.py
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


def load_vggface2():
    embs = np.load(DATA_DIR / "vggface2_100k_embeddings.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "vggface2_100k_labels.npy", allow_pickle=True)
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)
    # Hold out last 3 images per identity (instead of 1) for more queries
    qi = []
    for indices in label_to_idx.values():
        if len(indices) >= 4:  # need at least 4 (3 queries + 1 in DB)
            qi.extend(indices[-3:])
        elif len(indices) >= 2:
            qi.append(indices[-1])
    qi = np.array(qi)
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])
    print(f"  Queries: {len(queries)} (multi-holdout for tighter CIs)")
    return database, queries, gt


def bench_nbits(database, queries, gt, n_bits, n_candidates_list):
    """Benchmark a specific n_bits setting."""
    pca = PCABinaryQuantizer(n_bits=n_bits)
    pca.fit(database[:min(5000, len(database))])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    bytes_per_vec = db_codes.shape[1]
    memory_mb = db_codes.nbytes / 1024 / 1024

    results = []
    for n_cand in n_candidates_list:
        # Warmup
        for i in range(3):
            faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)

        times = []
        correct = 0
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand_idx = np.asarray(topk).astype(np.intp)
            cosine_sims = database[cand_idx] @ queries[i]
            best_idx = cand_idx[np.argmax(cosine_sims)]
            times.append((time.perf_counter() - t0) * 1000)
            if best_idx == gt[i]:
                correct += 1

        results.append({
            "n_candidates": n_cand,
            "recall_1": correct / len(queries),
            "mean_ms": np.mean(times),
            "p99_ms": np.percentile(times, 99),
        })

    return {
        "n_bits": n_bits,
        "bytes_per_vec": bytes_per_vec,
        "memory_mb": memory_mb,
        "results": results,
    }


def main():
    print("=" * 70)
    print("  FaceFlash — n_bits Sweep (VGGFace2, 100K real faces)")
    print("=" * 70)

    database, queries, gt = load_vggface2()
    print(f"  Database: {len(database):,}, Queries: {len(queries)}")

    faiss.omp_set_num_threads(1)

    # FAISS baseline for reference
    index = faiss.IndexFlatIP(512)
    index.add(database)
    times_faiss = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 1)
        times_faiss.append((time.perf_counter() - t0) * 1000)
    faiss_ms = np.mean(times_faiss)
    print(f"  FAISS-Flat: {faiss_ms:.2f}ms (reference)")

    # Sweep
    bits_to_test = [128, 256, 384, 512]
    n_candidates_list = [10, 30, 50, 100, 200, 300]

    all_results = []

    for n_bits in bits_to_test:
        print(f"\n  Testing {n_bits} bits ({n_bits // 8} bytes/vector)...")
        r = bench_nbits(database, queries, gt, n_bits, n_candidates_list)
        all_results.append(r)

        # Print summary for this n_bits
        for cr in r["results"]:
            print(f"    cands={cr['n_candidates']:<4} R@1={cr['recall_1']*100:<6.1f}% "
                  f"latency={cr['mean_ms']:.3f}ms")

    # Summary table
    print(f"\n  {'='*75}")
    print(f"  RESULTS — n_bits sweep at 100 candidates")
    print(f"  {'='*75}")
    print(f"  {'Bits':<8} {'Bytes':<8} {'Memory':<10} {'R@1':<8} {'Latency':<10} "
          f"{'Speedup':<10} {'vs 512-bit':<12}")
    print(f"  {'─'*70}")

    baseline_recall = None
    baseline_ms = None
    for r in all_results:
        # Get the 100-candidate result
        cr = next(x for x in r["results"] if x["n_candidates"] == 100)
        if r["n_bits"] == 512:
            baseline_recall = cr["recall_1"]
            baseline_ms = cr["mean_ms"]
        speedup_faiss = faiss_ms / cr["mean_ms"]
        recall_delta = f"{(cr['recall_1'] - (baseline_recall or cr['recall_1']))*100:+.1f}%" if baseline_recall else "—"

        print(f"  {r['n_bits']:<8} {r['bytes_per_vec']:<8} {r['memory_mb']:<8.1f}MB "
              f"{cr['recall_1']*100:<6.1f}% {cr['mean_ms']:<10.3f}"
              f"{speedup_faiss:<10.1f}x {recall_delta:<12}")

    print(f"  {'─'*70}")
    print(f"  FAISS-Flat: {faiss_ms:.2f}ms, 194.7 MB")

    # Find sweet spot
    print(f"\n  ANALYSIS:")
    for r in all_results:
        cr100 = next(x for x in r["results"] if x["n_candidates"] == 100)
        cr200 = next(x for x in r["results"] if x["n_candidates"] == 200)
        print(f"    {r['n_bits']} bits: {cr100['recall_1']*100:.1f}% @100 cands, "
              f"{cr200['recall_1']*100:.1f}% @200 cands, "
              f"{r['memory_mb']:.1f} MB")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_nbits.json", "w") as f:
        json.dump({"faiss_ms": faiss_ms, "results": all_results,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_nbits.json'}")


if __name__ == "__main__":
    main()
