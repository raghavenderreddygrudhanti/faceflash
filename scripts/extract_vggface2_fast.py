"""
Fast VGGFace2 extraction — batched ArcFace inference.
Target: 100K+ embeddings from VGGFace2 test split.

Uses batch inference (32 images at a time) for ~4x speedup over single-image.
Saves incrementally every 5000 images so you can stop and resume.

Usage:
    cd ~/lang-chain/faceflash
    .venv/bin/python scripts/extract_vggface2_fast.py
"""
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_EMBS = DATA_DIR / "vggface2_100k_embeddings.npy"
OUT_LABELS = DATA_DIR / "vggface2_100k_labels.npy"
CHECKPOINT = DATA_DIR / "vggface2_checkpoint.npz"

TARGET_N = 100000
BATCH_SIZE = 32


class BatchEmbedder:
    """ArcFace with batch inference support."""

    def __init__(self):
        import onnxruntime as ort
        model_path = str(Path.home() / ".faceflash" / "models" / "arcface_r100.onnx")
        # Try CoreML first (Apple Silicon), fall back to CPU
        providers = ["CPUExecutionProvider"]
        try:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"  Model loaded. Provider: {self.session.get_providers()[0]}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize to 112x112, normalize, CHW format."""
        pil_img = Image.fromarray(img).resize((112, 112), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32)
        arr = (arr - 127.5) / 127.5
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def embed_batch(self, images: list) -> np.ndarray:
        """Embed a batch of face images. Returns (N, 512) normalized."""
        batch = np.stack([self.preprocess(img) for img in images], axis=0)
        outputs = self.session.run(None, {self.input_name: batch})
        embeddings = outputs[0]
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return (embeddings / norms).astype(np.float32)


def main():
    # Check for existing checkpoint
    start_idx = 0
    existing_embs = []
    existing_labels = []

    if OUT_EMBS.exists():
        embs = np.load(OUT_EMBS)
        print(f"Already have {embs.shape[0]} embeddings. Done!")
        return

    if CHECKPOINT.exists():
        data = np.load(CHECKPOINT, allow_pickle=True)
        existing_embs = list(data["embeddings"])
        existing_labels = list(data["labels"])
        start_idx = len(existing_embs)
        print(f"Resuming from checkpoint: {start_idx} embeddings")

    from datasets import load_dataset

    print("Loading VGGFace2 test split (streaming)...")
    ds = load_dataset("logasja/VGGFace2", split="test", streaming=True)

    print("Loading ArcFace model...")
    embedder = BatchEmbedder()

    embeddings = existing_embs
    labels = existing_labels
    batch_imgs = []
    batch_labels = []
    failed = 0
    t_start = time.perf_counter()

    for i, sample in enumerate(tqdm(ds, total=TARGET_N + start_idx, initial=start_idx,
                                     desc="Embedding")):
        if i < start_idx:
            continue
        if len(embeddings) >= TARGET_N:
            break

        try:
            img = sample["image"]
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            img_np = np.array(img)
            # Mild center crop
            h, w = img_np.shape[:2]
            mh, mw = int(h * 0.05), int(w * 0.05)
            face = img_np[mh:h-mh, mw:w-mw]

            batch_imgs.append(face)
            batch_labels.append(sample.get("class_id", i))

            # Process batch
            if len(batch_imgs) >= BATCH_SIZE:
                embs = embedder.embed_batch(batch_imgs)
                embeddings.extend(embs)
                labels.extend(batch_labels)
                batch_imgs = []
                batch_labels = []
        except Exception:
            failed += 1
            batch_imgs = []
            batch_labels = []

        # Checkpoint every 5000
        if len(embeddings) > 0 and len(embeddings) % 5000 == 0 and len(embeddings) > start_idx:
            elapsed = time.perf_counter() - t_start
            rate = (len(embeddings) - start_idx) / elapsed
            eta = (TARGET_N - len(embeddings)) / rate / 60
            print(f"\n  Checkpoint: {len(embeddings)} embeddings, "
                  f"{rate:.1f} img/s, ETA {eta:.0f} min")
            np.savez(CHECKPOINT,
                     embeddings=np.array(embeddings, dtype=np.float32),
                     labels=np.array(labels))

    # Process remaining batch
    if batch_imgs:
        embs = embedder.embed_batch(batch_imgs)
        embeddings.extend(embs)
        labels.extend(batch_labels)

    # Save final
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels)

    # Remove zero-norm
    valid = np.linalg.norm(embeddings, axis=1) > 0.5
    embeddings = embeddings[valid]
    labels = labels[valid]

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {len(embeddings)} embeddings, {len(np.unique(labels))} identities")
    print(f"Failed: {failed}, Time: {elapsed/60:.1f} min ({len(embeddings)/elapsed:.1f} img/s)")

    np.save(OUT_EMBS, embeddings)
    np.save(OUT_LABELS, labels)
    print(f"Saved: {OUT_EMBS}")

    # Clean checkpoint
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()


if __name__ == "__main__":
    main()
