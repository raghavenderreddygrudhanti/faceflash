"""
FaceFlash — FAR / FRR at decision thresholds.

The "confidence" FaceFlash returns is raw cosine similarity, not a
probability. This benchmark translates thresholds into the error rates that
actually matter for deployment:

  FAR (false accept rate)  — fraction of impostor pairs scored >= threshold
  FRR (false reject rate)  — fraction of genuine pairs scored <  threshold

Pairs are built from a labeled embedding set (LFW by default): genuine =
two photos of the same person, impostor = photos of two different people.
Reports the full threshold sweep, the equal-error rate (EER), and the
thresholds needed for FAR = 1%, 0.1%, 0.01%.

Usage:
    python benchmarks/bench_far_frr.py [--data lfw] [--max-impostors 200000]
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load(tag):
    embs = np.load(DATA_DIR / f"{tag}_embeddings.npy").astype(np.float32)
    labels = np.load(DATA_DIR / f"{tag}_labels.npy", allow_pickle=True)
    labels = np.array([str(l) for l in labels])
    # L2-normalize so dot product == cosine
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs, labels


def build_pairs(embs, labels, max_genuine=100_000, max_impostors=200_000, seed=0):
    """Genuine sims (same label) and impostor sims (different labels)."""
    rng = np.random.default_rng(seed)

    by_label = {}
    for i, l in enumerate(labels):
        by_label.setdefault(l, []).append(i)

    genuine = []
    for idxs in by_label.values():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        # All within-identity pairs, capped per identity to avoid a few
        # heavily-photographed people dominating the distribution.
        n_pairs = min(len(idxs) * (len(idxs) - 1) // 2, 50)
        seen = set()
        while len(seen) < n_pairs:
            a, b = rng.choice(idxs, 2, replace=False)
            if a > b:
                a, b = b, a
            seen.add((int(a), int(b)))
        for a, b in seen:
            genuine.append(float(embs[a] @ embs[b]))
        if len(genuine) >= max_genuine:
            break

    n = len(embs)
    impostor = []
    while len(impostor) < max_impostors:
        a = rng.integers(0, n, size=4096)
        b = rng.integers(0, n, size=4096)
        keep = labels[a] != labels[b]
        sims = np.einsum('ij,ij->i', embs[a[keep]], embs[b[keep]])
        impostor.extend(sims.tolist())
    impostor = impostor[:max_impostors]

    return np.array(genuine), np.array(impostor)


def far_frr_table(genuine, impostor, thresholds):
    rows = []
    for t in thresholds:
        far = float(np.mean(impostor >= t))
        frr = float(np.mean(genuine < t))
        rows.append({"threshold": round(float(t), 3), "far": far, "frr": frr})
    return rows


def threshold_at_far(genuine, impostor, target_far):
    """Smallest threshold with FAR <= target, and the FRR paid for it."""
    t = float(np.quantile(impostor, 1.0 - target_far))
    return t, float(np.mean(genuine < t))


def main():
    ap = argparse.ArgumentParser(description="FAR/FRR threshold benchmark")
    ap.add_argument("--data", default="lfw", help="dataset tag (default: lfw)")
    ap.add_argument("--max-impostors", type=int, default=200_000)
    args = ap.parse_args()

    embs, labels = load(args.data)
    n_ids = len(set(labels))
    print(f"  {args.data}: {len(embs):,} embeddings, {n_ids:,} identities")

    genuine, impostor = build_pairs(embs, labels, max_impostors=args.max_impostors)
    print(f"  Pairs: {len(genuine):,} genuine, {len(impostor):,} impostor")

    thresholds = np.arange(0.20, 0.71, 0.05)
    rows = far_frr_table(genuine, impostor, thresholds)

    print(f"\n  {'Threshold':<11} {'FAR':<12} {'FRR':<10}")
    print(f"  {'─'*33}")
    for r in rows:
        print(f"  {r['threshold']:<11} {r['far']:<12.5f} {r['frr']:<10.5f}")

    # EER: threshold where FAR == FRR
    ts = np.linspace(0.0, 1.0, 2001)
    fars = np.array([np.mean(impostor >= t) for t in ts])
    frrs = np.array([np.mean(genuine < t) for t in ts])
    i_eer = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[i_eer] + frrs[i_eer]) / 2)

    operating_points = {}
    print(f"\n  EER: {eer*100:.2f}% at threshold {ts[i_eer]:.3f}")
    for target in (0.01, 0.001, 0.0001):
        t, frr = threshold_at_far(genuine, impostor, target)
        operating_points[f"far_{target}"] = {"threshold": round(t, 4), "frr": frr}
        print(f"  FAR {target*100:>5.2f}%: threshold {t:.3f}  (FRR {frr*100:.2f}%)")

    out = {
        "benchmark": "FAR/FRR at cosine thresholds",
        "dataset": args.data,
        "n_embeddings": len(embs),
        "n_identities": n_ids,
        "n_genuine_pairs": len(genuine),
        "n_impostor_pairs": len(impostor),
        "note": "confidence returned by FaceFlash is raw cosine similarity; "
                "these are the error rates a given threshold buys",
        "table": rows,
        "eer": eer,
        "eer_threshold": float(ts[i_eer]),
        "operating_points": operating_points,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"bench_far_frr_{args.data}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == "__main__":
    main()
