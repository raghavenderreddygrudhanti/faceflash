"""
1:N face identification benchmark — the realistic test.

Enroll ONE image per identity (a gallery of DISTINCT people), then probe with a
DIFFERENT held-out image, and report rank-1 accuracy: "did it find the right
person among N strangers?" This is the metric that actually matters, and it
needs a high-identity dataset (MS1MV2, ~85K people) to mean anything.

FaceFlash is compared against the FAISS-Flat exact ceiling — if FaceFlash ties
it, the binary compression loses nothing and any accuracy gap is the
embeddings' fault, not the search.

Picks the dataset via DATA_TAG (e.g. DATA_TAG=ms1m), else largest VGGFace2 file.

Usage:
    DATA_TAG=ms1m python benchmarks/bench_identification.py
    python benchmarks/bench_identification.py --max-probes 5000
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_data():
    tag = os.environ.get("DATA_TAG")
    names = ([tag] if tag else []) + ["ms1m", "vggface2_1m", "vggface2_500k",
                                      "vggface2_100k", "vggface2"]
    for name in names:
        emb = DATA_DIR / f"{name}_embeddings.npy"
        lbl = DATA_DIR / f"{name}_labels.npy"
        if emb.exists() and lbl.exists():
            print(f"  Loaded: {emb.name}")
            return np.load(emb).astype(np.float32), np.load(lbl, allow_pickle=True), emb.name
    print("  ERROR: no embeddings found in data/")
    sys.exit(1)


def build_sparse_gallery(embs, labels, max_probes):
    """One image per identity in the gallery; a different image as the probe."""
    by_id = {}
    for i, l in enumerate(labels):
        by_id.setdefault(str(l), []).append(i)
    gallery_idx, probe_idx = [], []
    for idxs in by_id.values():
        if len(idxs) >= 2:
            gallery_idx.append(idxs[0])      # enroll first
            probe_idx.append(idxs[-1])       # probe with a different image
    gallery = np.ascontiguousarray(embs[np.array(gallery_idx)])
    probe_idx = np.array(probe_idx[:max_probes])
    probes = np.ascontiguousarray(embs[probe_idx])
    truth = np.arange(len(probe_idx))        # probe j's answer is gallery slot j
    return gallery, probes, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-probes", type=int, default=5000)
    ap.add_argument("--configs", default="256/100,512/100,512/300",
                    help="comma-separated n_bits/n_candidates")
    args = ap.parse_args()

    embs, labels, src = load_data()
    gallery, probes, truth = build_sparse_gallery(embs, labels, args.max_probes)
    n_id = len(gallery)
    print(f"  Gallery: {n_id:,} DISTINCT identities (1 image each) | "
          f"Probes: {len(probes):,}")
    if n_id < 1000:
        print(f"  ⚠ only {n_id} identities — use MS1MV2 (DATA_TAG=ms1m) for a real test")

    out = {"dataset": src, "n_gallery_identities": int(n_id),
           "n_probes": int(len(probes)),
           "note": "rank-1 identification: 1 image/identity gallery, different probe",
           "results": []}

    # Exact ceiling (FAISS-Flat)
    if HAS_FAISS:
        faiss.omp_set_num_threads(1)
        idx = faiss.IndexFlatIP(gallery.shape[1]); idx.add(gallery)
        correct = sum(int(idx.search(probes[j:j+1], 1)[1][0][0]) == truth[j]
                      for j in range(len(probes)))
        exact = correct / len(probes)
        out["results"].append({"method": "FAISS-Flat (exact)", "rank1": exact})
        print(f"  Exact rank-1 (ceiling): {exact*100:.1f}%")

    # FaceFlash at each operating point
    for cfg in args.configs.split(","):
        nb, nc = (int(x) for x in cfg.split("/"))
        pca = PCABinaryQuantizer(n_bits=nb); pca.fit(gallery[:min(5000, n_id)])
        gc = pca.encode(gallery); pc = pca.encode(probes)
        correct = 0
        for j in range(len(probes)):
            topk = faceflash_core.hamming_topk(pc[j], gc, min(nc, n_id))
            cand = np.asarray(topk).astype(np.intp)
            if int(cand[np.argmax(gallery[cand] @ probes[j])]) == truth[j]:
                correct += 1
        r1 = correct / len(probes)
        out["results"].append({"method": f"FaceFlash({nb}/{nc})", "rank1": r1,
                               "memory_mb": round(gc.nbytes / 1024 / 1024, 3)})
        print(f"  FaceFlash {nb}/{nc}: rank-1 {r1*100:.1f}%")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = os.environ.get("DATA_TAG", "data")
    for p in (RESULTS_DIR / f"bench_identification_{tag}.json",):
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
    print(f"  ✓ Saved: results/bench_identification_{tag}.json")


if __name__ == "__main__":
    main()
