"""
Pre-flight check — run this BEFORE renting a GPU.

Streams a few hundred samples from the dataset and verifies the two things the
1M run depends on:
  1. There is an 'image' field and a usable label field.
  2. The same identity shows up more than once (otherwise recall can't be tested).

Costs nothing but a minute of streaming. If this fails, fix the field names in
scripts/runpod_full.sh before you pay for a GPU.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --dataset logasja/VGGFace2 --n 300
"""
import sys
import argparse
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="logasja/VGGFace2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300, help="samples to probe")
    args = ap.parse_args()

    print(f"Pre-flight: {args.dataset} (split={args.split}, probing {args.n} samples)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("FAIL: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    try:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
    except Exception as e:
        print(f"FAIL: could not open dataset: {e}")
        sys.exit(1)

    probe = next(iter(ds))
    keys = list(probe.keys())
    print(f"  Fields available: {keys}")

    # 1. image field
    if "image" not in keys:
        print(f"\nFAIL: no 'image' field. Edit runpod_full.sh to use the right key.")
        sys.exit(1)
    print("  ✓ 'image' field present")

    # 2. label field
    label_field = next((c for c in ("class_id", "label", "identity", "person_id", "id")
                        if c in keys), None)
    if label_field is None:
        print(f"\nFAIL: no known label field in {keys}.")
        print("  Pick the identity column above and set it in runpod_full.sh.")
        sys.exit(1)
    print(f"  ✓ label field: '{label_field}'")

    # 3. repeats per identity (needed for recall)
    counts = Counter()
    for i, s in enumerate(ds):
        if i >= args.n:
            break
        counts[str(s.get(label_field))] += 1
    repeats = sum(1 for v in counts.values() if v >= 2)
    print(f"\n  {len(counts)} identities in {args.n} samples; "
          f"{repeats} have 2+ images.")

    if repeats == 0:
        print("\nFAIL: no identity repeats in the sample. Recall can't be measured.")
        print("  The stream may be shuffled or one-image-per-identity here.")
        sys.exit(1)

    print("\nPASS — dataset looks good. Safe to launch the GPU run.")


if __name__ == "__main__":
    main()
