"""
Face Dedup — Find duplicate faces in a dataset.

Usage:
    python examples/face_dedup.py --input photos/ --threshold 0.6

Finds all pairs of photos that contain the same person,
useful for cleaning datasets or finding duplicates in a gallery.
"""
import argparse
from pathlib import Path
from collections import defaultdict

from faceflash import FaceFlash


def main():
    parser = argparse.ArgumentParser(description="Find duplicate faces")
    parser.add_argument("--input", required=True, help="Folder with photos")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Similarity threshold for 'same person' (default: 0.6)")
    parser.add_argument("--output", default=None, help="Output file (default: print)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in extensions])
    print(f"Scanning {len(images)} images for duplicate faces...")

    ff = FaceFlash(n_bits=512, n_candidates=100)

    # Register all images (each as its own "person" = filename)
    registered = []
    for i, img_path in enumerate(images):
        try:
            ff.register(str(img_path), str(img_path))
            registered.append(img_path)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  Registered {i+1}/{len(images)}")

    print(f"  Registered: {len(registered)} faces")
    print(f"  Searching for duplicates...\n")

    # Search each image against the index (skip self-match)
    duplicates = []
    seen_pairs = set()

    for i, img_path in enumerate(registered):
        try:
            result = ff.search(str(img_path), k=5, threshold=args.threshold)
            for match in result["matches"]:
                other_path = match["name"]
                if other_path == str(img_path):
                    continue  # skip self
                # Avoid duplicate pairs
                pair = tuple(sorted([str(img_path), other_path]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    duplicates.append({
                        "file_a": str(img_path),
                        "file_b": other_path,
                        "confidence": match["confidence"],
                    })
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            print(f"  Checked {i+1}/{len(registered)} — {len(duplicates)} duplicates found")

    # Output
    print(f"\nFound {len(duplicates)} duplicate pairs:")
    for d in duplicates[:20]:
        a = Path(d["file_a"]).name
        b = Path(d["file_b"]).name
        print(f"  {a} <-> {b} (similarity: {d['confidence']:.2f})")

    if len(duplicates) > 20:
        print(f"  ... and {len(duplicates) - 20} more")

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(duplicates, f, indent=2)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
