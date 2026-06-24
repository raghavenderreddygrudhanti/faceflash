"""
n_bits x candidates grid benchmark with bootstrap confidence intervals.

Characterizes the memory <-> recall trade for FaceFlash on real faces.
For every (n_bits, n_candidates) pair, measures Recall@1 vs the exact
cosine nearest neighbor, with a 95% bootstrap CI over queries, plus
stable per-query latency. Writes results/bench_nbits_grid.json so the
operating-point claims are reproducible and statistically defensible.

Recall is defined against exact search: ground truth = argmax of exact
cosine over the database (standard ANN recall, not identity accuracy).

Usage:
    cd ~/lang-chain/faceflash
    .venv/bin/python benchmarks/bench_nbits_grid.py
    .venv/bin/python benchmarks/bench_nbits_grid.py --bits 256,384,512 --cands 50,100,300
"""
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS = Path(__file__).parent.parent / "results" / "bench_nbits_grid.json"

# Timing knobs
TIMING_REPEATS = 5      # per-query repeats; take the min (best-of) to cut OS noise
N_BOOTSTRAP = 2000      # bootstrap resamples for the recall CI
RNG = np.random.default_rng(0)


def load_data():
    """Load the largest available embeddings + labels (DATA_TAG first, then biggest scale)."""
    import os
    candidates = []
    tag = os.environ.get("DATA_TAG")
    if tag:  # e.g. DATA_TAG=ms1m → picks up an MS1MV2 extraction
        candidates.append((DATA_DIR / f"{tag}_embeddings.npy", DATA_DIR / f"{tag}_labels.npy"))
    candidates += [
        (DATA_DIR / "vggface2_1m_embeddings.npy", DATA_DIR / "vggface2_1m_labels.npy"),
        (DATA_DIR / "vggface2_500k_embeddings.npy", DATA_DIR / "vggface2_500k_labels.npy"),
        (DATA_DIR / "vggface2_100k_embeddings.npy", DATA_DIR / "vggface2_100k_labels.npy"),
        (DATA_DIR / "vggface2_embeddings.npy", DATA_DIR / "vggface2_labels.npy"),
    ]
    for emb_path, lbl_path in candidates:
        if emb_path.exists() and lbl_path.exists():
            embs = np.load(emb_path)
            if embs.dtype != np.float32:  # avoid a 2 GB copy at 1M scale
                embs = embs.astype(np.float32)
            labels = np.load(lbl_path, allow_pickle=True)
            return embs, labels, emb_path.name
    raise FileNotFoundError("No VGGFace2 embeddings found in data/. Run extraction first.")


def split_query_database(embs, labels, max_queries=1000, per_identity=3):
    """Hold out the last `per_identity` images of each multi-image identity as queries.

    Holding out up to 3 per identity (not just 1) yields enough queries for tight
    confidence intervals at any scale (~800+ at 100K, more at 1M). Always keeps at
    least one image of each held-out identity in the database, so the query has a
    true match to find.
    """
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)
    qi = []
    for idxs in label_to_idx.values():
        if len(idxs) >= per_identity + 1:
            qi.extend(idxs[-per_identity:])
        elif len(idxs) >= 2:
            qi.append(idxs[-1])
    qi = np.array(qi[:max_queries])
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = np.ascontiguousarray(embs[mask])
    queries = np.ascontiguousarray(embs[qi])
    return database, queries


def bootstrap_ci(correct_vec, n_boot=N_BOOTSTRAP, alpha=0.05):
    """95% bootstrap CI for a mean (recall) over the per-query correctness vector."""
    n = len(correct_vec)
    if n == 0:
        return (0.0, 0.0)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boot_means = correct_vec[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (lo, hi)


def time_search(q_code, db_codes, database, query_vec, n_cand):
    """Run one query (hamming filter + cosine rerank); return (best_idx, best_ms)."""
    best_ms = float("inf")
    best_idx = -1
    for _ in range(TIMING_REPEATS):
        t0 = time.perf_counter()
        topk = faceflash_core.hamming_topk(q_code, db_codes, n_cand)
        cand_idx = np.asarray(topk).astype(np.intp)
        cosine_sims = database[cand_idx] @ query_vec
        best_idx = int(cand_idx[np.argmax(cosine_sims)])
        ms = (time.perf_counter() - t0) * 1000
        best_ms = min(best_ms, ms)
    return best_idx, best_ms


def faiss_baseline_ms(database, queries):
    """Single-thread FAISS-Flat exact search latency, for the speedup column."""
    if not _HAS_FAISS:
        return None
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    times = []
    for q in queries:  # warmup + measure
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 1)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="128,256,384,512",
                    help="comma-separated bit widths")
    ap.add_argument("--cands", default="10,30,50,100,200,300,500",
                    help="comma-separated candidate counts")
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()

    bit_widths = [int(b) for b in args.bits.split(",")]
    cand_counts = [int(c) for c in args.cands.split(",")]

    print(f"Backend: {backend_info()}")
    embs, labels, src = load_data()
    database, queries = split_query_database(embs, labels, args.queries)
    n_db, dim = database.shape
    n_q = len(queries)
    print(f"Source: {src}  |  database={n_db:,}  queries={n_q}  dim={dim}")
    print(f"Timing: best-of-{TIMING_REPEATS} per query   Bootstrap: {N_BOOTSTRAP} resamples")

    # Ground truth = exact cosine NN, chunked over queries to bound memory at 1M
    # (a full (n_db x n_q) matrix would be ~4 GB at 1M; chunks keep it small).
    gt = np.empty(n_q, dtype=np.int64)
    for s in range(0, n_q, 200):
        gt[s:s + 200] = (database @ queries[s:s + 200].T).argmax(axis=0)

    faiss_ms = faiss_baseline_ms(database, queries)
    if faiss_ms:
        print(f"FAISS-Flat (1 thread): {faiss_ms:.3f}ms median\n")
    else:
        print("FAISS not available — speedup column omitted\n")

    grid = []
    hdr = f"{'bits':>5} {'cand':>5} {'R@1':>7} {'95% CI':>16} {'lat(ms)':>9} {'p95':>7} {'vsFAISS':>8}"
    print(hdr)
    print("-" * len(hdr))

    for n_bits in bit_widths:
        quant = PCABinaryQuantizer(n_bits=n_bits)
        quant.fit(database[:min(5000, n_db)])
        db_codes = np.ascontiguousarray(quant.encode(database))
        q_codes = np.ascontiguousarray(quant.encode(queries))
        index_mb = db_codes.nbytes / 1024 / 1024

        # binary-only recall@1 (no rerank): top-1 hamming == exact NN
        binary_correct = np.zeros(n_q)
        for i in range(n_q):
            top1 = faceflash_core.hamming_topk(q_codes[i], db_codes, 1)
            binary_correct[i] = (int(np.asarray(top1)[0]) == gt[i])
        binary_recall = float(binary_correct.mean())

        bit_entry = {
            "n_bits": n_bits,
            "bytes_per_face": n_bits // 8,
            "index_mb": round(index_mb, 2),
            "binary_recall_at1": round(binary_recall, 4),
            "candidates": [],
        }

        for n_cand in cand_counts:
            correct = np.zeros(n_q)
            lat = np.zeros(n_q)
            # warmup
            time_search(q_codes[0], db_codes, database, queries[0], n_cand)
            for i in range(n_q):
                best_idx, ms = time_search(q_codes[i], db_codes, database, queries[i], n_cand)
                correct[i] = (best_idx == gt[i])
                lat[i] = ms
            recall = float(correct.mean())
            lo, hi = bootstrap_ci(correct)
            lat_mean = float(lat.mean())
            lat_med = float(np.median(lat))
            lat_p95 = float(np.percentile(lat, 95))
            speedup = round(faiss_ms / lat_mean, 1) if faiss_ms else None

            bit_entry["candidates"].append({
                "n_cand": n_cand,
                "recall_at1": round(recall, 4),
                "recall_ci95": [round(lo, 4), round(hi, 4)],
                "latency_mean_ms": round(lat_mean, 4),
                "latency_median_ms": round(lat_med, 4),
                "latency_p95_ms": round(lat_p95, 4),
                "speedup_vs_faiss": speedup,
            })
            sp = f"{speedup:.1f}x" if speedup else "-"
            print(f"{n_bits:>5} {n_cand:>5} {recall*100:>6.1f}% "
                  f"[{lo*100:>5.1f},{hi*100:>5.1f}] {lat_mean:>8.3f} {lat_p95:>6.3f} {sp:>8}")

        grid.append(bit_entry)
        print(f"      -> {n_bits} bits: {bit_entry['bytes_per_face']} B/face, "
              f"{index_mb:.2f} MB, binary-only R@1={binary_recall*100:.1f}%\n")

    out = {
        "benchmark": "n_bits x candidates grid (memory vs recall)",
        "dataset": src,
        "n_database": int(n_db),
        "n_queries": int(n_q),
        "recall_definition": "Recall@1 vs exact cosine nearest neighbor",
        "timing": f"best-of-{TIMING_REPEATS} per query, single-threaded",
        "n_bootstrap": N_BOOTSTRAP,
        "faiss_baseline_ms": round(faiss_ms, 4) if faiss_ms else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grid": grid,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {RESULTS.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
