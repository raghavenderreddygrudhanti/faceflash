"""
Photo Organizer — Group photos by person (like Google Photos).

Usage:
    python examples/photo_organizer.py --input ~/Photos/ --output ~/Organized/

It scans all photos, detects faces, clusters them by identity,
and copies each photo into a folder named after the person.
"""
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

from faceflash import FaceFlash

def main():
    parser = argparse.ArgumentParser(description="Organize photos by person")
    parser.add_argument("--input", required=True, help="Folder with unsorted photos")
    parser.add_argument("--output", default="organized/", help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Similarity threshold for same person (default: 0.45)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]
    print(f"Found {len(images)} images in {input_dir}")

    ff = FaceFlash(n_bits=512, n_candidates=100)
    person_count = 0
    assignments = defaultdict(list)  # person_name -> [image_paths]
    failed = []

    for i, img_path in enumerate(images):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(images)}... ({person_count} people found)")

        try:
            if len(ff) == 0:
                # First face — register as Person_1
                person_count += 1
                name = f"Person_{person_count:03d}"
                ff.register(name, str(img_path))
                assignments[name].append(img_path)
            else:
                # Search for this face
                result = ff.search(str(img_path), k=1, threshold=args.threshold)
                if result["matches"]:
                    # Known person
                    name = result["matches"][0]["name"]
                    assignments[name].append(img_path)
                    # Also register this photo (improves future matching)
                    ff.register(name, str(img_path))
                else:
                    # New person
                    person_count += 1
                    name = f"Person_{person_count:03d}"
                    ff.register(name, str(img_path))
                    assignments[name].append(img_path)
        except Exception as e:
            failed.append((img_path, str(e)))

    # Copy organized photos
    print(f"\nOrganizing into {output_dir}/...")
    for name, paths in assignments.items():
        person_dir = output_dir / name
        person_dir.mkdir(exist_ok=True)
        for p in paths:
            shutil.copy2(p, person_dir / p.name)

    # Summary
    print(f"\nDone!")
    print(f"  People found: {person_count}")
    print(f"  Photos organized: {sum(len(v) for v in assignments.values())}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output: {output_dir}/")

    if failed:
        print(f"\n  Failed photos (no face detected):")
        for path, err in failed[:10]:
            print(f"    {path.name}: {err}")


if __name__ == "__main__":
    main()
