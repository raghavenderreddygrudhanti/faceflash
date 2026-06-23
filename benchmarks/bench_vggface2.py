"""
Run FaceFlash benchmark on VGGFace2 embeddings (after extraction completes).

Usage:
    cd ~/lang-chain/faceflash
    .venv/bin/python benchmarks/bench_vggface2.py
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss

DATA_DIR = Path(__file__).parent.parent / "data"
# Auto-detect best available VGGFace2 embeddings
EMB_100K = DATA_DIR / "vggface2_100k_embeddings.npy"
LBL_100K = DATA_DIR / "vggface2_100k_labels.npy"
EMB_20K = DATA_DIR / "vggface2_embeddings.npy"
LBL_20K = DATA_DIR / "vggface2_labels.npy"

if EMB_100K.exists():
    EMB_PATH = EMB_100K
    LBL_PATH = LBL_100K
else:
    EMB_PATH = EMB_20K
    LBL_PATH = LBL_20K


def main():
    if not EMB_PATH.exists():
        print("VGGFace2 embeddings not found. Run scripts/extract_vggface2.py first.")
        return

    print(f"Backend: {backend_info()}")
    embs = np.load(EMB_PATH).astype(np.float32)
    labels = np.load(LBL_PATH, allow_pickle=True)
    n_ids = len(np.unique(labels))
    print(f"VGGFace2: {embs.shape[0]:,} embeddings, {n_ids:,} identities")

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

    print(f"Database: {len(database):,}, Queries: {len(queries)}")

    # Ground truth
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])

    # PCA
    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(database[:min(5000, len(database))])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    # FAISS baseline (single-thread)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(512)
    index.add(database)
    times_faiss = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), 1)
        times_faiss.append((time.perf_counter() - t0) * 1000)

    # FaceFlash sweep
    print(f"\n{'Cands':<8} {'Recall@1':<10} {'Latency':<12} {'vs FAISS'}")
    print("-" * 45)
    faiss_ms = np.mean(times_faiss)
    print(f"{'FAISS':<8} {'100.0%':<10} {faiss_ms:.3f}ms      1.0x")

    for n_cand in [10, 30, 50, 100, 200, 300, 500, 1000]:
        correct = 0
        times = []
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand_idx = np.asarray(topk).astype(np.intp)
            cosine_sims = database[cand_idx] @ queries[i]
            best_idx = cand_idx[np.argmax(cosine_sims)]
            times.append((time.perf_counter() - t0) * 1000)
            if best_idx == gt[i]:
                correct += 1
        recall = correct / len(queries)
        avg_ms = np.mean(times)
        speedup = faiss_ms / avg_ms
        print(f"{n_cand:<8} {recall*100:.1f}%     {avg_ms:.3f}ms      {speedup:.1f}x")

    print(f"\nMemory: {db_codes.nbytes/1024:.0f}KB binary vs {database.nbytes/1024/1024:.1f}MB float")


if __name__ == "__main__":
    main()
