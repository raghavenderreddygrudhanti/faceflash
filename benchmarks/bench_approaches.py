"""
FaceFlash — Compare Different Quantization Approaches

Tests multiple binary code generation methods at 256 bits (the sweet spot):
  1. Random Projection (SimHash baseline)
  2. PCA only (no rotation)
  3. PCA + ITQ (current default)
  4. Sign of raw embedding (simplest possible)
  5. Top-256 PCA components (dimensional reduction)

Also compares against FAISS IndexBinaryFlat (their binary search implementation)
to validate our Rust backend is actually fast.

Usage:
    .venv/bin/python benchmarks/bench_approaches.py
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

N_BITS = 256  # Sweet spot from nbits sweep
N_CANDS = 100


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


def bench_method(name, db_codes, q_codes, database, queries, gt):
    """Benchmark a quantization method using Rust Hamming + cosine rerank."""
    # Warmup
    for i in range(3):
        faceflash_core.hamming_topk(q_codes[i], db_codes, N_CANDS)

    times = []
    correct = 0
    for i in range(len(queries)):
        t0 = time.perf_counter()
        topk = faceflash_core.hamming_topk(q_codes[i], db_codes, N_CANDS)
        cand_idx = np.asarray(topk).astype(np.intp)
        cosine_sims = database[cand_idx] @ queries[i]
        best_idx = cand_idx[np.argmax(cosine_sims)]
        times.append((time.perf_counter() - t0) * 1000)
        if best_idx == gt[i]:
            correct += 1

    return {
        "method": name,
        "recall_1": correct / len(queries),
        "mean_ms": np.mean(times),
        "p99_ms": np.percentile(times, 99),
        "memory_mb": db_codes.nbytes / 1024 / 1024,
    }


def main():
    print("=" * 70)
    print(f"  FaceFlash — Quantization Approaches ({N_BITS} bits, VGGFace2 100K)")
    print("=" * 70)

    database, queries, gt = load_data()
    n_db, dim = database.shape
    print(f"  Database: {n_db:,} × {dim}, Queries: {len(queries)}")
    print(f"  n_bits={N_BITS}, n_candidates={N_CANDS}")

    rng = np.random.default_rng(42)
    results = []

    # ─── Method 1: Random Projection (SimHash) ─────────────────────────
    print("\n  [1] Random Projection...")
    W_rand = rng.standard_normal((N_BITS, dim)).astype(np.float32)
    W_rand /= np.linalg.norm(W_rand, axis=1, keepdims=True)
    db_rand = np.packbits((database @ W_rand.T > 0).astype(np.uint8), axis=1)
    q_rand = np.packbits((queries @ W_rand.T > 0).astype(np.uint8), axis=1)
    results.append(bench_method("Random Projection", db_rand, q_rand, database, queries, gt))

    # ─── Method 2: PCA only (no ITQ) ───────────────────────────────────
    print("  [2] PCA only...")
    X = database[:5000].astype(np.float64)
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    W_pca = Vt[:N_BITS].astype(np.float32)
    projections = database[:5000] @ W_pca.T
    threshold_pca = np.median(projections, axis=0).astype(np.float32)
    db_pca = np.packbits((database @ W_pca.T - threshold_pca > 0).astype(np.uint8), axis=1)
    q_pca = np.packbits((queries @ W_pca.T - threshold_pca > 0).astype(np.uint8), axis=1)
    results.append(bench_method("PCA only", db_pca, q_pca, database, queries, gt))

    # ─── Method 3: PCA + ITQ (current default) ─────────────────────────
    print("  [3] PCA + ITQ...")
    pca_itq = PCABinaryQuantizer(n_bits=N_BITS)
    pca_itq.fit(database[:5000])
    db_itq = pca_itq.encode(database)
    q_itq = pca_itq.encode(queries)
    results.append(bench_method("PCA + ITQ", db_itq, q_itq, database, queries, gt))

    # ─── Method 4: Sign of raw embedding (simplest) ────────────────────
    print("  [4] Sign of embedding (first 256 dims)...")
    db_sign = np.packbits((database[:, :N_BITS] > 0).astype(np.uint8), axis=1)
    q_sign = np.packbits((queries[:, :N_BITS] > 0).astype(np.uint8), axis=1)
    results.append(bench_method("Sign (first 256 dims)", db_sign, q_sign, database, queries, gt))

    # ─── Method 5: FAISS IndexBinaryFlat (their binary search) ─────────
    print("  [5] FAISS Binary (IndexBinaryFlat)...")
    faiss_binary = faiss.IndexBinaryFlat(N_BITS)
    faiss_binary.add(db_itq)  # Use ITQ codes

    times_fb = []
    correct_fb = 0
    for i in range(len(queries)):
        t0 = time.perf_counter()
        D, I = faiss_binary.search(q_itq[i:i+1], N_CANDS)
        cand_idx = I[0].astype(np.intp)
        cosine_sims = database[cand_idx] @ queries[i]
        best_idx = cand_idx[np.argmax(cosine_sims)]
        times_fb.append((time.perf_counter() - t0) * 1000)
        if best_idx == gt[i]:
            correct_fb += 1

    results.append({
        "method": "FAISS Binary + rerank",
        "recall_1": correct_fb / len(queries),
        "mean_ms": np.mean(times_fb),
        "p99_ms": np.percentile(times_fb, 99),
        "memory_mb": db_itq.nbytes / 1024 / 1024,
    })

    # ─── Summary ────────────────────────────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  RESULTS — {N_BITS} bits, {N_CANDS} candidates, VGGFace2 100K")
    print(f"  {'='*65}")
    print(f"  {'Method':<28} {'R@1':<8} {'Latency':<10} {'Memory':<8}")
    print(f"  {'─'*60}")
    for r in results:
        print(f"  {r['method']:<28} {r['recall_1']*100:<6.1f}% "
              f"{r['mean_ms']:<10.3f}{r['memory_mb']:<6.1f}MB")
    print(f"  {'─'*60}")

    # Compare Rust vs FAISS binary backend
    rust_r = next(r for r in results if r["method"] == "PCA + ITQ")
    faiss_r = next(r for r in results if r["method"] == "FAISS Binary + rerank")
    print(f"\n  Rust vs FAISS Binary (same codes, same rerank):")
    print(f"    Rust:  {rust_r['mean_ms']:.3f}ms")
    print(f"    FAISS: {faiss_r['mean_ms']:.3f}ms")
    print(f"    Ratio: {faiss_r['mean_ms']/rust_r['mean_ms']:.2f}x")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_approaches.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_approaches.json'}")


if __name__ == "__main__":
    main()
