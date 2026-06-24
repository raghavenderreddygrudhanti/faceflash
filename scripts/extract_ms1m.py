"""
Extract face embeddings from MS1MV2 (faces_emore) dataset.

MS1MV2 is distributed as MXNet RecordIO (.rec/.idx) files with images
already aligned to 112x112. We decode directly and run ArcFace.

Data source: InsightFace's cleaned MS-Celeb-1M (85K identities, 5.8M images)
Format: train.rec + train.idx (MXNet RecordIO)

Usage:
    python scripts/extract_ms1m.py [--target 1000000] [--data-dir data]
"""

import sys
import os
import time
import struct
import argparse
import numpy as np
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))


def read_rec_file(rec_path, idx_path, target_n, embedder):
    """
    Read MXNet RecordIO .rec file and extract embeddings.
    
    RecordIO format:
      - Each record: 4 bytes (magic) + 4 bytes (length/flag) + data
      - Image records contain a header (label info) + JPEG bytes
    """
    import struct
    from PIL import Image

    # Read index file to get offsets
    print(f"  Reading index file: {idx_path}")
    offsets = []
    with open(idx_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                offsets.append((int(parts[0]), int(parts[1])))
    
    # Sort by index
    offsets.sort(key=lambda x: x[0])
    print(f"  Total records in index: {len(offsets):,}")
    
    # The first record (index 0) is typically the header with identity ranges
    # Records 1..N are the actual face images
    embeddings = []
    labels = []
    failed = 0
    
    t_start = time.perf_counter()
    
    with open(rec_path, 'rb') as f:
        # Skip header record (index 0)
        # Parse records starting from index 1
        n_to_process = min(target_n, len(offsets) - 1)
        
        for rec_idx in range(1, n_to_process + 1):
            if len(embeddings) >= target_n:
                break
                
            try:
                idx_key, offset = offsets[rec_idx]
                f.seek(offset)
                
                # Read record header: magic(4) + lrec(4)
                header = f.read(8)
                if len(header) < 8:
                    break
                magic_lflag = struct.unpack('<I', header[0:4])[0]
                # cflag is the higher bits
                cflag = (magic_lflag >> 29) & 7
                lrec = magic_lflag & ((1 << 29) - 1)
                
                # Read record data
                data = f.read(lrec)
                if len(data) < lrec:
                    break
                
                # Parse MXNet image record header
                # Header: flag(4) + label(4 or 8 float) + id(8)
                # IRHeader: flag(I), label(f), id(Q), id2(Q)
                # flag=0: single label, flag>0: seq label
                header_flag = struct.unpack('<I', data[0:4])[0]
                label_val = struct.unpack('<f', data[4:8])[0]
                
                # Image data starts after the 24-byte header
                img_data = data[24:]
                
                # Decode image
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img_np = np.array(img)
                
                # Already 112x112 aligned — embed directly
                emb = embedder.embed(img_np)
                embeddings.append(emb)
                labels.append(int(label_val))
                
            except Exception:
                failed += 1
                continue
            
            # Progress
            if len(embeddings) % 50000 == 0 and len(embeddings) > 0:
                elapsed = time.perf_counter() - t_start
                rate = len(embeddings) / elapsed
                remaining = (target_n - len(embeddings)) / max(rate, 1) / 60
                print(f"  Checkpoint: {len(embeddings):,} done | "
                      f"{rate:.0f} img/s | ETA {remaining:.0f} min | Failed: {failed}")
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)


def read_rec_mxnet(rec_path, idx_path, target_n, embedder):
    """Read using MXNet recordio (if available) — more reliable."""
    try:
        import mxnet as mx
    except ImportError:
        return None, None
    
    from PIL import Image
    
    print(f"  Using MXNet RecordIO reader")
    imgrec = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), 'r')
    
    # Read header (record 0) to get identity info
    header0 = mx.recordio.unpack(imgrec.read_idx(0))
    # header0[0] contains (start_idx, end_idx) for identities
    
    embeddings = []
    labels = []
    failed = 0
    t_start = time.perf_counter()
    
    keys = list(imgrec.keys)
    keys.sort()
    n_to_process = min(target_n + 1, len(keys))
    
    for key in keys[1:n_to_process]:  # skip header at index 0
        if len(embeddings) >= target_n:
            break
        try:
            record = imgrec.read_idx(key)
            header, img_bytes = mx.recordio.unpack(record)
            label = int(header.label) if isinstance(header.label, float) else int(header.label[0])
            
            img = mx.image.imdecode(img_bytes).asnumpy()  # BGR
            img_rgb = img[:, :, ::-1]  # to RGB
            
            emb = embedder.embed(img_rgb)
            embeddings.append(emb)
            labels.append(label)
        except Exception:
            failed += 1
        
        if len(embeddings) % 50000 == 0 and len(embeddings) > 0:
            elapsed = time.perf_counter() - t_start
            rate = len(embeddings) / elapsed
            remaining = (target_n - len(embeddings)) / max(rate, 1) / 60
            print(f"  Checkpoint: {len(embeddings):,} done | "
                  f"{rate:.0f} img/s | ETA {remaining:.0f} min | Failed: {failed}")
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=1000000)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--rec-dir', default='data/ms1m')
    args = parser.parse_args()
    
    DATA_DIR = Path(args.data_dir)
    REC_DIR = Path(args.rec_dir)
    OUT = DATA_DIR / 'ms1m_embeddings.npy'
    OUT_L = DATA_DIR / 'ms1m_labels.npy'
    
    if OUT.exists():
        embs = np.load(OUT)
        print(f"  ✓ Already extracted: {embs.shape[0]:,} MS1MV2 embeddings")
        return
    
    # Find .rec files
    rec_path = REC_DIR / 'train.rec'
    idx_path = REC_DIR / 'train.idx'
    
    if not rec_path.exists():
        print(f"  ERROR: {rec_path} not found!")
        print(f"  Download MS1MV2 (faces_emore) first. See instructions in runpod_full.sh")
        sys.exit(1)
    
    print(f"  MS1MV2 (faces_emore) extraction")
    print(f"  rec: {rec_path} ({rec_path.stat().st_size / 1024/1024/1024:.1f} GB)")
    print(f"  Target: {args.target:,} embeddings")
    
    from faceflash.embed import FaceEmbedder
    embedder = FaceEmbedder()
    
    import onnxruntime as ort
    provider = embedder.session.get_providers()[0]
    print(f"  ONNX provider: {provider}")
    if 'CUDA' in provider:
        print(f"  ★ GPU acceleration active")
    
    # Try MXNet reader first (more reliable), fall back to manual parsing
    embeddings, labels = read_rec_mxnet(rec_path, idx_path, args.target, embedder)
    
    if embeddings is None:
        print(f"  MXNet not available, using manual RecordIO parser")
        embeddings, labels = read_rec_file(rec_path, idx_path, args.target, embedder)
    
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
