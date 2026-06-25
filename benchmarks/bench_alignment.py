"""
Alignment benchmark — Haar center-crop vs RetinaFace 5-point alignment.

Runs the standard LFW verification protocol from RAW images through each
alignment path, then the same ArcFace embedder. Reports 10-fold accuracy and
TAR@FAR. This is the benchmark that measures the alignment upgrade — the
embedding benchmarks use pre-aligned data and bypass it.

LFW source (auto): local data/lfw_funneled/pairs.txt if present, else
scikit-learn's fetch_lfw_pairs (downloads + caches reliably — no manual fetch).

Usage:
    python benchmarks/bench_alignment.py
    python benchmarks/bench_alignment.py --max-pairs 1200   # quick subset
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.embed import FaceEmbedder
from faceflash.detect import _detect_haar_fallback
from faceflash import align as align_mod

LFW_DIR = Path(__file__).parent.parent / "data" / "lfw_funneled"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_lfw(max_pairs=None):
    """Return (images {key: rgb_uint8}, pairs [(key_a, key_b, same)])."""
    import cv2
    images, pairs = {}, []

    if (LFW_DIR / "pairs.txt").exists():
        print("  LFW source: local data/lfw_funneled/")
        def _p(name, idx):
            return LFW_DIR / name / f"{name}_{int(idx):04d}.jpg"
        for ln in (LFW_DIR / "pairs.txt").read_text().splitlines()[1:]:
            f = ln.split()
            if len(f) == 3:
                a, b, same = _p(f[0], f[1]), _p(f[0], f[2]), True
            elif len(f) == 4:
                a, b, same = _p(f[0], f[1]), _p(f[2], f[3]), False
            else:
                continue
            if a.exists() and b.exists():
                ka, kb = str(a), str(b)
                images[ka] = None
                images[kb] = None
                pairs.append((ka, kb, same))
        for k in images:
            images[k] = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)
    else:
        print("  LFW source: scikit-learn fetch_lfw_pairs (download + cache)")
        from sklearn.datasets import fetch_lfw_pairs
        # slice_=None → full funneled image (250x250). The default tight crop
        # (~125x94) leaves no context for the face detector, so both methods
        # would fall back to center-crop and the comparison would be meaningless.
        d = fetch_lfw_pairs(subset="10_folds", funneled=True, color=True,
                            resize=1.0, slice_=None)
        arr = d.pairs
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
        for i, (pair, lbl) in enumerate(zip(arr, d.target)):
            ka, kb = f"{i}a", f"{i}b"
            images[ka], images[kb] = pair[0], pair[1]
            pairs.append((ka, kb, bool(lbl)))

    if max_pairs:
        pairs = pairs[:max_pairs // 2] + pairs[-max_pairs // 2:]
        keep = {k for a, b, _ in pairs for k in (a, b)}
        images = {k: v for k, v in images.items() if k in keep}
    return images, pairs


def embed_method(images, embedder, method):
    """Embed each unique image once via the given alignment method."""
    cache = {}
    for k, img in images.items():
        if method == "haar":
            face = _detect_haar_fallback(img)
        else:  # retinaface (falls back to haar if no face found)
            face = align_mod.detect_and_align(img)
            if face is None:
                face = _detect_haar_fallback(img)
        cache[k] = embedder.embed(face)
    return cache


def tenfold_accuracy(sims, labels):
    """LFW 10-fold accuracy: threshold tuned on 9 folds, tested on the 10th."""
    sims, labels = np.asarray(sims), np.asarray(labels)
    folds = np.array_split(np.arange(len(sims)), 10)
    thresholds = np.linspace(-1, 1, 400)
    accs = []
    for i in range(10):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(10) if j != i])
        best_t = max(thresholds, key=lambda t:
                     ((sims[train] >= t) == labels[train]).mean())
        accs.append(float(((sims[test] >= best_t) == labels[test]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def tar_at_far(sims, labels, far_target=0.001):
    sims, labels = np.asarray(sims), np.asarray(labels)
    thr = np.quantile(sims[labels == 0], 1 - far_target)
    return float((sims[labels == 1] >= thr).mean())


def evaluate(method, images, pairs, embedder):
    cache = embed_method(images, embedder, method)
    sims = [float(cache[a] @ cache[b]) for a, b, _ in pairs]
    labels = [int(s) for _, _, s in pairs]
    acc, std = tenfold_accuracy(sims, labels)
    return {
        "method": method,
        "accuracy": round(acc, 4),
        "accuracy_std": round(std, 4),
        "tar_at_far_1e-3": round(tar_at_far(sims, labels), 4),
        "mean_genuine_cos": round(float(np.mean([s for s, l in zip(sims, labels) if l])), 4),
        "mean_impostor_cos": round(float(np.mean([s for s, l in zip(sims, labels) if not l])), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    images, pairs = load_lfw(args.max_pairs)
    print(f"  LFW pairs: {len(pairs)} ({sum(p[2] for p in pairs)} same), "
          f"{len(images)} unique images")
    embedder = FaceEmbedder()

    out = {"benchmark": "alignment (LFW verification)", "n_pairs": len(pairs), "results": []}
    print(f"\n  {'method':<12} {'accuracy':<16} {'TAR@FAR=1e-3':<14} {'genuine':<9} {'impostor'}")
    print("  " + "-" * 62)
    for method in ("haar", "retinaface"):
        r = evaluate(method, images, pairs, embedder)
        out["results"].append(r)
        print(f"  {method:<12} {r['accuracy']*100:>5.2f}% ± {r['accuracy_std']*100:.2f}    "
              f"{r['tar_at_far_1e-3']*100:>6.2f}%       "
              f"{r['mean_genuine_cos']:<9} {r['mean_impostor_cos']}")

    if len(out["results"]) == 2:
        lift = (out["results"][1]["accuracy"] - out["results"][0]["accuracy"]) * 100
        print(f"\n  → RetinaFace alignment lifts LFW accuracy by {lift:+.2f} points")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "bench_alignment.json", "w") as f:
        json.dump(out, f, indent=2)
    print("  ✓ Saved: results/bench_alignment.json")


if __name__ == "__main__":
    main()
