"""
Compression Isolation Benchmark — PCA+ITQ codes searched via FaceFlash vs FAISS.

Addresses the question: "Is the value in the compression or the search?"
Answer: the compression. Both search methods (FaceFlash Hamming kernel and
FAISS IndexBinaryFlat) perform the same brute-force Hamming scan. The recall
comes from PCA+ITQ preserving NN ordering on face embeddings.

This benchmark:
  1. Generates PCA+ITQ binary codes from real embeddings
  2. Searches the SAME codes via FaceFlash Rust kernel
  3. Searches the SAME codes via FAISS IndexBinaryFlat
  4. Compares recall and latency — they should be nearly identical,
     proving the value is in the quantization, not the scan.

Usage:
    python benchmarks/bench_compression_isolation.py
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from faceflash.pca_quantize import PCABinaryQuantizer, _POPCOUNT_TABLE

try:
    from faceflash import _core as _rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("FAISS not installed — run: pip install faiss-cpu")

# ---------------------------------------------------------------------------

DATA_PATH = Path("data/ms1m_embeddings.npy")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_QUERIES = 500
K = 1
N_CANDIDATES = 100  # FaceFlash rerank shortlist
N_BITS = 512


def load_data(n_db=100_000):
    """Load MS1MV2 embeddings or generate synthetic ones if unavailable."""
    if DATA_PATH.exists():
        print(f"Loading real MS1MV2 embeddings...")
        all_emb = np.load(DATA_PATH)
        # Normalize
        norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        all_emb = (all_emb / norms).astype(np.float32)
        n_db = min(n_db, len(all_emb) - N_QUERIES)
        db = all_emb[:n_db]
        queries = all_emb[n_db:n_db + N_QUERIES]
        print(f"  Database: {db.shape[0]}, Queries: {queries.shape[0]}, Dim: {db.shape[1]}")
        return db, queries
    else:
        print("MS1MV2 not found — using synthetic data (results won't reflect real faces)")
        rng = np.random.default_rng(42)
        db = rng.standard_normal((n_db, 512)).astype(np.float32)
        db /= np.linalg.norm(db, axis=1, keepdims=True)
        queries = rng.standard_normal((N_QUERIES, 512)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        return db, queries


def ground_truth(db, queries, k=1):
    """Exact brute-force cosine NN."""
    gt = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        sims = db @ q
        gt[i] = np.argsort(-sims)[:k]
    return gt


def recall_at_k(predicted, gt, k=1):
    hits = 0
    for i in range(len(gt)):
        if gt[i, 0] in predicted[i, :k]:
            hits += 1
    return hits / len(gt)


def bench_faceflash(db, queries, quantizer, db_codes, k=1):
    """FaceFlash: Hamming shortlist + cosine rerank."""
    n_cand = min(N_CANDIDATES, len(db))
    predicted = np.empty((len(queries), k), dtype=np.int64)

    latencies = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        q_code = quantizer.encode(q).ravel()

        # Hamming phase
        if HAS_RUST:
            cand_idx = np.asarray(_rust.hamming_topk(q_code, db_codes, n_cand)).astype(np.intp)
        else:
            xor = np.bitwise_xor(db_codes, q_code.reshape(1, -1))
            dists = _POPCOUNT_TABLE[xor].sum(axis=1)
            cand_idx = np.argpartition(dists, n_cand)[:n_cand]

        # Cosine rerank
        sims = db[cand_idx] @ q
        top = np.argsort(-sims)[:k]
        predicted[i] = cand_idx[top]
        latencies.append(time.perf_counter() - t0)

    return predicted, latencies


def bench_faiss_binary(db, queries, quantizer, db_codes, k=1):
    """FAISS IndexBinaryFlat on the SAME PCA+ITQ codes, then cosine rerank."""
    n_bits = db_codes.shape[1] * 8
    n_cand = min(N_CANDIDATES, len(db))

    # Build FAISS binary index
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(db_codes)

    predicted = np.empty((len(queries), k), dtype=np.int64)
    latencies = []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        q_code = quantizer.encode(q).reshape(1, -1)

        # FAISS Hamming search
        D, I = index.search(q_code, n_cand)
        cand_idx = I[0].astype(np.intp)
        # Remove -1 padding
        cand_idx = cand_idx[cand_idx >= 0]

        # Cosine rerank (same as FaceFlash)
        if len(cand_idx) > 0:
            sims = db[cand_idx] @ q
            top = np.argsort(-sims)[:k]
            predicted[i] = cand_idx[top]
        else:
            predicted[i] = 0

        latencies.append(time.perf_counter() - t0)

    return predicted, latencies


def bench_faiss_binary_only(db, queries, quantizer, db_codes, k=1):
    """FAISS IndexBinaryFlat — Hamming only, NO cosine rerank. Shows raw binary recall."""
    n_bits = db_codes.shape[1] * 8
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(db_codes)

    predicted = np.empty((len(queries), k), dtype=np.int64)
    latencies = []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        q_code = quantizer.encode(q).reshape(1, -1)
        D, I = index.search(q_code, k)
        predicted[i] = I[0][:k]
        latencies.append(time.perf_counter() - t0)

    return predicted, latencies


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Compression Isolation Benchmark")
    print("PCA+ITQ codes → searched via FaceFlash kernel vs FAISS IndexBinaryFlat")
    print("=" * 70)
    print(f"FaceFlash Rust backend: {'yes' if HAS_RUST else 'no (NumPy fallback)'}")
    print(f"FAISS: {'yes' if HAS_FAISS else 'NOT INSTALLED'}")
    print()

    db, queries = load_data(100_000)
    dim = db.shape[1]

    # Fit PCA+ITQ quantizer
    print(f"Fitting PCA+ITQ quantizer (n_bits={N_BITS}) on {min(5000, len(db))} samples...")
    quantizer = PCABinaryQuantizer(n_bits=N_BITS)
    quantizer.fit(db[:5000])

    # Encode database
    print("Encoding database to binary codes...")
    db_codes = quantizer.encode(db)
    print(f"  Codes shape: {db_codes.shape} ({db_codes.nbytes / 1024:.1f} KB)")
    print()

    # Ground truth
    print("Computing ground truth (exact cosine)...")
    gt = ground_truth(db, queries, k=K)
    print()

    # --- Benchmark 1: FaceFlash (Hamming + cosine rerank) ---
    print("Running FaceFlash (Hamming shortlist + cosine rerank)...")
    ff_pred, ff_lat = bench_faceflash(db, queries, quantizer, db_codes, k=K)
    ff_recall = recall_at_k(ff_pred, gt, k=K)
    print(f"  Recall@{K}: {ff_recall*100:.1f}%")
    print(f"  Avg latency: {np.mean(ff_lat)*1000:.3f}ms")
    print(f"  P99 latency: {np.percentile(ff_lat, 99)*1000:.3f}ms")
    print()

    if HAS_FAISS:
        # --- Benchmark 2: FAISS Binary + cosine rerank (same codes) ---
        print("Running FAISS IndexBinaryFlat + cosine rerank (SAME codes)...")
        faiss_pred, faiss_lat = bench_faiss_binary(db, queries, quantizer, db_codes, k=K)
        faiss_recall = recall_at_k(faiss_pred, gt, k=K)
        print(f"  Recall@{K}: {faiss_recall*100:.1f}%")
        print(f"  Avg latency: {np.mean(faiss_lat)*1000:.3f}ms")
        print(f"  P99 latency: {np.percentile(faiss_lat, 99)*1000:.3f}ms")
        print()

        # --- Benchmark 3: FAISS Binary ONLY (no rerank) ---
        print("Running FAISS IndexBinaryFlat — Hamming only, NO rerank...")
        faiss_raw_pred, faiss_raw_lat = bench_faiss_binary_only(db, queries, quantizer, db_codes, k=K)
        faiss_raw_recall = recall_at_k(faiss_raw_pred, gt, k=K)
        print(f"  Recall@{K}: {faiss_raw_recall*100:.1f}%")
        print(f"  Avg latency: {np.mean(faiss_raw_lat)*1000:.3f}ms")
        print()

    # --- Summary ---
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Database: {len(db)} vectors, {dim}-dim, {N_BITS}-bit codes")
    print(f"  Queries: {N_QUERIES}, k={K}, n_candidates={N_CANDIDATES}")
    print()
    print(f"  {'Method':<45} {'Recall@1':<12} {'Avg Latency':<15}")
    print(f"  {'─'*70}")
    print(f"  {'FaceFlash (PCA+ITQ + Rust Hamming + rerank)':<45} {ff_recall*100:.1f}%{'':<7} {np.mean(ff_lat)*1000:.3f}ms")
    if HAS_FAISS:
        print(f"  {'FAISS IndexBinaryFlat + rerank (SAME codes)':<45} {faiss_recall*100:.1f}%{'':<7} {np.mean(faiss_lat)*1000:.3f}ms")
        print(f"  {'FAISS IndexBinaryFlat only (no rerank)':<45} {faiss_raw_recall*100:.1f}%{'':<7} {np.mean(faiss_raw_lat)*1000:.3f}ms")
    print()
    print("  INTERPRETATION:")
    print("  - FaceFlash vs FAISS+rerank should have IDENTICAL recall")
    print("    (same codes, same rerank — only the Hamming kernel differs)")
    print("  - The Hamming-only row shows raw binary code quality without rerank")
    print("  - Any latency difference = kernel speed, not algorithmic advantage")
    print("  - The REAL contribution is the PCA+ITQ compression achieving high")
    print("    recall on face embeddings, not the search algorithm")
    print("=" * 70)

    # Save results
    results = {
        "config": {"n_db": len(db), "n_queries": N_QUERIES, "k": K,
                   "n_bits": N_BITS, "n_candidates": N_CANDIDATES, "dim": dim},
        "faceflash": {"recall_at_1": ff_recall,
                      "avg_latency_ms": round(np.mean(ff_lat) * 1000, 3),
                      "p99_latency_ms": round(np.percentile(ff_lat, 99) * 1000, 3)},
    }
    if HAS_FAISS:
        results["faiss_binary_rerank"] = {
            "recall_at_1": faiss_recall,
            "avg_latency_ms": round(np.mean(faiss_lat) * 1000, 3),
            "p99_latency_ms": round(np.percentile(faiss_lat, 99) * 1000, 3),
        }
        results["faiss_binary_only"] = {
            "recall_at_1": faiss_raw_recall,
            "avg_latency_ms": round(np.mean(faiss_raw_lat) * 1000, 3),
        }

    out_path = RESULTS_DIR / "bench_compression_isolation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
