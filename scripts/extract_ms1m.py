"""
Extract face embeddings from MS1MV2 (ms1m-arcface) dataset.

MS1MV2 is available on Kaggle as individual JPEGs organized in folders:
  ms1m-arcface/<identity_id>/<image>.jpg
  
Images are already aligned to 112x112 by InsightFace.
85,742 identities, ~5.8M images total.

Usage:
    python scripts/extract_ms1m.py [--target 1000000] [--data-dir data]
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_from_folders(ms1m_dir, target_n, embedder, max_per_identity=None):
    """
    Extract embeddings from folder-of-JPEGs format:
      ms1m_dir/<identity_id>/<image>.jpg
    
    If max_per_identity is set, extracts at most N images per identity
    but covers ALL identities (for maximum identity diversity).
    """
    ms1m_path = Path(ms1m_dir)
    
    # Find all identity folders
    identity_dirs = sorted([d for d in ms1m_path.iterdir() if d.is_dir()])
    print(f"  Found {len(identity_dirs):,} identity folders")
    if max_per_identity:
        print(f"  Mode: ALL identities, max {max_per_identity} images each")
    else:
        print(f"  Mode: sequential until {target_n:,} embeddings")

    embeddings = []
    labels = []
    failed = 0
    t_start = time.perf_counter()

    # Label = folder position (works for ANY folder naming: m.0abc, 00001, etc.)
    for label_idx, identity_dir in enumerate(tqdm(identity_dirs, desc="  Identities")):
        if not max_per_identity and len(embeddings) >= target_n:
            break

        image_files = sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.png"))
        
        # Limit images per identity if specified
        if max_per_identity:
            image_files = image_files[:max_per_identity]

        for img_path in image_files:
            if not max_per_identity and len(embeddings) >= target_n:
                break
            try:
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                emb = embedder.embed(img_np)
                embeddings.append(emb)
                labels.append(label_idx)
            except Exception:
                failed += 1
        
        # Checkpoint every 50K embeddings
        if len(embeddings) % 50000 < len(image_files) and len(embeddings) > 0:
            elapsed = time.perf_counter() - t_start
            rate = len(embeddings) / elapsed
            remaining_ids = len(identity_dirs) - label_idx
            est_remaining = remaining_ids * (elapsed / max(label_idx, 1)) / 60
            print(f"\n  Progress: {len(embeddings):,} embeddings, "
                  f"{label_idx+1:,}/{len(identity_dirs):,} identities | "
                  f"{rate:.0f} img/s | ETA {est_remaining:.0f} min | Failed: {failed}")
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)


def extract_from_rec(rec_path, idx_path, target_n, embedder):
    """Extract from MXNet RecordIO .rec file (alternative format)."""
    try:
        import mxnet as mx
    except ImportError:
        print("  MXNet not available, cannot read .rec files")
        return None, None
    
    print(f"  Using MXNet RecordIO reader")
    imgrec = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), 'r')
    
    embeddings = []
    labels = []
    failed = 0
    t_start = time.perf_counter()
    
    keys = sorted(list(imgrec.keys))
    n_to_process = min(target_n + 1, len(keys))
    
    for key in tqdm(keys[1:n_to_process], desc="  Extracting"):
        if len(embeddings) >= target_n:
            break
        try:
            record = imgrec.read_idx(key)
            header, img_bytes = mx.recordio.unpack(record)
            label = int(header.label) if isinstance(header.label, float) else int(header.label[0])
            
            img = mx.image.imdecode(img_bytes).asnumpy()[:, :, ::-1]  # BGR→RGB
            emb = embedder.embed(img)
            embeddings.append(emb)
            labels.append(label)
        except Exception:
            failed += 1
        
        if len(embeddings) % 50000 == 0 and len(embeddings) > 0:
            elapsed = time.perf_counter() - t_start
            rate = len(embeddings) / elapsed
            remaining = (target_n - len(embeddings)) / max(rate, 1) / 60
            print(f"\n  Progress: {len(embeddings):,} done | "
                  f"{rate:.0f} img/s | ETA {remaining:.0f} min | Failed: {failed}")
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=1000000)
    parser.add_argument('--max-per-identity', type=int, default=None,
                        help='Max images per identity. If set, processes ALL identities '
                             '(ignores --target). Use 10-20 for full 85K identity coverage.')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--ms1m-dir', default='data/ms1m')
    args = parser.parse_args()
    
    DATA_DIR = Path(args.data_dir)
    MS1M_DIR = Path(args.ms1m_dir)
    OUT = DATA_DIR / 'ms1m_embeddings.npy'
    OUT_L = DATA_DIR / 'ms1m_labels.npy'
    
    if OUT.exists():
        embs = np.load(OUT)
        print(f"  ✓ Already extracted: {embs.shape[0]:,} MS1MV2 embeddings")
        return
    
    # Detect format: folder-of-JPEGs or .rec file
    rec_path = MS1M_DIR / 'train.rec'
    
    # Check for folder-of-JPEGs (Kaggle format: ms1m-arcface/<id>/<img>.jpg)
    jpeg_dir = None
    for candidate in [MS1M_DIR / 'ms1m-arcface', MS1M_DIR, MS1M_DIR / 'faces_emore']:
        if candidate.exists():
            subdirs = sorted([d for d in candidate.iterdir() if d.is_dir()])
            # look in the first few subfolders for any jpg/png (robust to naming)
            if subdirs and any(any(d.glob(ext) for ext in ("*.jpg", "*.png"))
                               for d in subdirs[:5]):
                jpeg_dir = candidate
                break
    
    if jpeg_dir is None and not rec_path.exists():
        print(f"  ERROR: No MS1MV2 data found in {MS1M_DIR}")
        print(f"  Expected either:")
        print(f"    - Folder format: {MS1M_DIR}/ms1m-arcface/<id>/*.jpg")
        print(f"    - RecordIO format: {MS1M_DIR}/train.rec + train.idx")
        sys.exit(1)
    
    from faceflash.embed import FaceEmbedder
    embedder = FaceEmbedder()
    
    import onnxruntime as ort
    provider = embedder.session.get_providers()[0]
    print(f"  ONNX provider: {provider}")
    if 'CUDA' in provider:
        print(f"  ★ GPU acceleration active")
    
    if jpeg_dir:
        print(f"  MS1MV2 format: folder-of-JPEGs ({jpeg_dir})")
        embeddings, labels = extract_from_folders(jpeg_dir, args.target, embedder,
                                                   max_per_identity=args.max_per_identity)
    else:
        print(f"  MS1MV2 format: MXNet RecordIO ({rec_path})")
        idx_path = MS1M_DIR / 'train.idx'
        embeddings, labels = extract_from_rec(rec_path, idx_path, args.target, embedder)
    
    if embeddings is None or len(embeddings) == 0:
        print("  ERROR: No embeddings extracted!")
        sys.exit(1)
    
    # Filter zero-norm
    valid = np.linalg.norm(embeddings, axis=1) > 0.5
    embeddings = embeddings[valid]
    labels = labels[valid]
    
    n_ids = len(np.unique(labels))
    print(f"\n  ═══════════════════════════════════════════════════")
    print(f"  MS1MV2 EXTRACTION COMPLETE")
    print(f"  Embeddings: {len(embeddings):,}")
    print(f"  Identities: {n_ids:,}")
    print(f"  ═══════════════════════════════════════════════════")
    
    np.save(OUT, embeddings)
    np.save(OUT_L, labels)
    print(f"  ✓ Saved: {OUT} ({embeddings.nbytes/1024/1024:.0f} MB)")
    print(f"  ✓ Saved: {OUT_L}")


if __name__ == '__main__':
    main()
