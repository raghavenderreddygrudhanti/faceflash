"""
TEMPORARY — pull MS1M images from YOUR HuggingFace dataset (not Kaggle).

Downloads the image dataset from a HF repo and lays it out as
    data/ms1m/ms1m-arcface/<identity>/<image>.jpg
so scripts/extract_ms1m.py can consume it unchanged.

Auto-detects the storage format:
  • zip / tar archives        → downloads + extracts them
  • raw folder tree of images → snapshot-downloads as-is
  • parquet / webdataset      → decodes via `datasets` into identity folders

Usage:
    export HF_TOKEN=hf_xxx                       # only if the repo is private
    python scripts/hf_images.py --repo <user>/<dataset>
    # then:  python scripts/extract_ms1m.py --ms1m-dir data/ms1m --max-per-identity 12

Env fallbacks: HF_IMG_REPO (repo id), HF_TOKEN (auth).
"""
import os
import sys
import glob
import time
import tarfile
import zipfile
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_DEFAULT = ROOT / "data" / "ms1m"


def _extract_archive(path: Path, dest: Path):
    print(f"  extracting {path.name} ...")
    if path.suffix == ".zip" or zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(dest)
    else:
        print(f"  ! unknown archive type: {path.name}")


def main():
    ap = argparse.ArgumentParser()
    # Repo id: --repo, else HF_IMG_REPO, else fall back to the embeddings repo
    # (HF_EMB_REPO) in case images live alongside the embeddings.
    ap.add_argument("--repo", default=(os.environ.get("HF_IMG_REPO")
                                       or os.environ.get("HF_EMB_REPO")),
                    help="HF dataset repo id, e.g. user/ms1m-images")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    args = ap.parse_args()

    if not args.repo:
        sys.exit("  ✗ Set --repo <user>/<dataset> (or export HF_IMG_REPO / HF_EMB_REPO)")

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    print(f"  Downloading {args.repo} from HuggingFace (this can be large)...")
    t0 = time.time()
    local = Path(snapshot_download(repo_id=args.repo, repo_type=args.repo_type,
                                   token=token, local_dir=str(out / "_hf_raw")))
    print(f"  ✓ Downloaded to {local}  ({time.time()-t0:.0f}s)")

    # ── Case 1: archives present → extract into the expected layout ───────────
    archives = [Path(p) for p in glob.glob(str(local / "**" / "*"), recursive=True)
                if Path(p).suffix.lower() in (".zip", ".tar", ".gz", ".tgz")]
    if archives:
        print(f"  Found {len(archives)} archive(s) → extracting...")
        for a in archives:
            _extract_archive(a, out)
    else:
        print("  No archives found — assuming raw image folders or parquet.")

    # ── Case 2: parquet / webdataset → decode via datasets ───────────────────
    parquets = glob.glob(str(local / "**" / "*.parquet"), recursive=True)
    if parquets and not archives:
        print(f"  Found {len(parquets)} parquet file(s) → decoding with `datasets`...")
        from datasets import load_dataset
        ds = load_dataset(args.repo, split="train", token=token)
        # expects an image column + a label/identity column
        img_col = next((c for c in ds.column_names
                        if c.lower() in ("image", "img", "jpg", "png")), None)
        lbl_col = next((c for c in ds.column_names
                        if c.lower() in ("label", "identity", "id", "person")), None)
        if not img_col or not lbl_col:
            sys.exit(f"  ✗ Couldn't find image/label columns in {ds.column_names}")
        base = out / "ms1m-arcface"
        for i, row in enumerate(ds):
            d = base / str(row[lbl_col])
            d.mkdir(parents=True, exist_ok=True)
            row[img_col].save(d / f"{i}.jpg")
            if i % 50000 == 0:
                print(f"    wrote {i:,} images...")
        print(f"  ✓ Decoded {len(ds):,} images → {base}")

    # ── Report what we ended up with ─────────────────────────────────────────
    # Find the deepest dir holding per-identity folders of images
    candidates = [out / "ms1m-arcface", out]
    for c in candidates:
        if c.exists():
            folders = [d for d in c.iterdir() if d.is_dir() and d.name != "_hf_raw"]
            if len(folders) > 1000:
                print(f"\n  ✓ Ready: {len(folders):,} identity folders under {c}")
                print(f"    Next: python scripts/extract_ms1m.py --ms1m-dir {out} "
                      f"--max-per-identity 12")
                return
    print("\n  ⚠ Could not auto-locate identity folders. Inspect:", out)
    print("    Tell me the layout and I'll adjust the extractor path.")


if __name__ == "__main__":
    main()
