"""
Sync embeddings with the HuggingFace dataset (public, ungated).

  download  — pull precomputed embeddings into data/ (skip Kaggle + extraction)
  upload    — push newly extracted embeddings back to HF for future runs

Repo: raghavenderreddy1212/faceflash-embeddings (dataset).
Download needs no token (the dataset is public). Upload needs HF_TOKEN
with write access.

    python scripts/hf_sync.py download --files ms1m_embeddings.npy ms1m_labels.npy
    python scripts/hf_sync.py upload   --files ms1m_embeddings.npy ms1m_labels.npy
"""
import os
import sys
import argparse
from pathlib import Path

REPO = os.environ.get("HF_EMB_REPO", "raghavenderreddy1212/faceflash-embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "upload"])
    ap.add_argument("--files", nargs="+",
                    default=["ms1m_embeddings.npy", "ms1m_labels.npy"])
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("  huggingface_hub not installed"); sys.exit(1)

    if args.action == "download":
        all_ok = True
        for f in args.files:
            try:
                hf_hub_download(REPO, f, repo_type="dataset", token=token,
                                local_dir=str(data))
                print(f"  ✓ pulled {f} from HF")
            except Exception as e:
                print(f"  · {f} not on HF ({type(e).__name__})")
                all_ok = False
        sys.exit(0 if all_ok else 1)

    else:  # upload
        if not token:
            print("  HF_TOKEN not set — skipping upload"); sys.exit(1)
        api = HfApi()
        for f in args.files:
            fp = data / f
            if not fp.exists():
                print(f"  · {f} missing locally — skip"); continue
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=f,
                            repo_id=REPO, repo_type="dataset", token=token)
            print(f"  ✓ uploaded {f} to HF")


if __name__ == "__main__":
    main()
