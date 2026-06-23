"""Extract ArcFace embeddings from LFW dataset (already downloaded)."""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.embed import FaceEmbedder

DATA_DIR = Path(__file__).parent.parent / "data"
LFW_DIR = DATA_DIR / "lfw_funneled"
EMBEDDINGS_PATH = DATA_DIR / "lfw_embeddings.npy"
LABELS_PATH = DATA_DIR / "lfw_labels.npy"


def main():
    if EMBEDDINGS_PATH.exists():
        embs = np.load(EMBEDDINGS_PATH)
        print(f"Already extracted: {embs.shape}")
        return

    print("Loading ArcFace model...")
    embedder = FaceEmbedder()

    # Collect all images
    persons = sorted([d for d in LFW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(persons)} identities")

    all_images = []
    all_labels = []
    for person_dir in persons:
        name = person_dir.name
        for img_path in sorted(person_dir.glob("*.jpg")):
            all_images.append(img_path)
            all_labels.append(name)

    print(f"Total images: {len(all_images)}")

    # Extract embeddings
    embeddings = []
    failed = 0
    for img_path in tqdm(all_images, desc="Extracting"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            # LFW funneled is pre-aligned — center crop to face
            h, w = img_np.shape[:2]
            margin_h = int(h * 0.15)
            margin_w = int(w * 0.15)
            face = img_np[margin_h:h-margin_h, margin_w:w-margin_w]
            emb = embedder.embed(face)
            embeddings.append(emb)
        except Exception as e:
            embeddings.append(np.zeros(512, dtype=np.float32))
            failed += 1

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(all_labels)

    # Remove failed
    valid = np.linalg.norm(embeddings, axis=1) > 0.5
    embeddings = embeddings[valid]
    labels = labels[valid]

    print(f"\nValid: {len(embeddings)} / {len(all_images)} (failed: {failed})")
    print(f"Saving to {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)

    # Quick stats
    unique = np.unique(labels)
    print(f"Identities: {len(unique)}")
    print(f"Embedding norm range: [{np.linalg.norm(embeddings, axis=1).min():.3f}, "
          f"{np.linalg.norm(embeddings, axis=1).max():.3f}]")


if __name__ == "__main__":
    main()
