"""
Alignment benchmark — Haar center-crop vs RetinaFace 5-point alignment.

Runs the standard LFW verification protocol (pairs.txt, 10 folds) from RAW
images through each alignment path, then the same ArcFace embedder. Reports
10-fold accuracy and TAR@FAR. This is the benchmark that actually measures the
alignment upgrade — the embedding benchmarks use pre-aligned data and bypass it.

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


def _img_path(name, idx):
    return LFW_DIR / name / f"{name}_{int(idx):04d}.jpg"


def parse_pairs(max_pairs=None):
    """Parse LFW pairs.txt → list of (pathA, pathB, same)."""
    lines = (LFW_DIR / "pairs.txt").read_text().splitlines()[1:]  # skip header
    pairs = []
    for ln in lines:
        f = ln.split()
        if len(f) == 3:      # same person
            a, b, same = _img_path(f[0], f[1]), _img_path(f[0], f[2]), True
        elif len(f) == 4:    # different people
            a, b, same = _img_path(f[0], f[1]), _img_path(f[2], f[3]), False
        else:
            continue
        if a.exists() and b.exists():   # skip pairs missing from a partial dataset
            pairs.append((a, b, same))
    if max_pairs:
        # keep balance: interleave from both ends (same pairs first half, diff second)
        pairs = pairs[:max_pairs // 2] + pairs[-max_pairs // 2:]
    return pairs


def embed_all(paths, embedder, method):
    """Embed each unique image once via the given alignment method."""
    import cv2
    cache = {}
    for p in paths:
        if p in cache:
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if method == "haar":
            face = _detect_haar_fallback(img)
        else:  # retinaface (falls back to haar if no face found)
            face = align_mod.detect_and_align(img)
            if face is None:
                face = _detect_haar_fallback(img)
        cache[p] = embedder.embed(face)
    return cache


def tenfold_accuracy(sims, labels):
    """Standard LFW 10-fold accuracy: threshold tuned on 9 folds, tested on the 10th."""
    sims, labels = np.asarray(sims), np.asarray(labels)
    n = len(sims)
    folds = np.array_split(np.arange(n), 10)
    accs = []
    thresholds = np.linspace(-1, 1, 400)
    for i in range(10):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(10) if j != i])
        best_t = max(thresholds, key=lambda t:
                     ((sims[train] >= t) == labels[train]).mean())
        accs.append(float(((sims[test] >= best_t) == labels[test]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def tar_at_far(sims, labels, far_target=0.001):
    sims, labels = np.asarray(sims), np.asarray(labels)
    impostor = sims[labels == 0]
    genuine = sims[labels == 1]
    thr = np.quantile(impostor, 1 - far_target)   # threshold giving the target FAR
    return float((genuine >= thr).mean())


def evaluate(method, pairs, embedder):
    paths = [p for pair in pairs for p in pair[:2]]
    cache = embed_all(paths, embedder, method)
    sims = [float(cache[a] @ cache[b]) for a, b, _ in pairs]
    labels = [int(same) for _, _, same in pairs]
    acc, std = tenfold_accuracy(sims, labels)
    return {
        "method": method,
        "accuracy": round(acc, 4),
        "accuracy_std": round(std, 4),
        "tar_at_far_1e-3": round(tar_at_far(sims, labels, 1e-3), 4),
        "mean_genuine_cos": round(float(np.mean([s for s, l in zip(sims, labels) if l])), 4),
        "mean_impostor_cos": round(float(np.mean([s for s, l in zip(sims, labels) if not l])), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=None, help="subset for a quick run")
    args = ap.parse_args()

    pairs = parse_pairs(args.max_pairs)
    print(f"  LFW pairs: {len(pairs)} ({sum(p[2] for p in pairs)} same)")
    embedder = FaceEmbedder()

    out = {"benchmark": "alignment (LFW verification)", "n_pairs": len(pairs), "results": []}
    print(f"\n  {'method':<12} {'accuracy':<16} {'TAR@FAR=1e-3':<14} {'genuine':<9} {'impostor'}")
    print("  " + "-" * 62)
    for method in ("haar", "retinaface"):
        r = evaluate(method, pairs, embedder)
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
    print(f"  ✓ Saved: results/bench_alignment.json")


if __name__ == "__main__":
    main()
