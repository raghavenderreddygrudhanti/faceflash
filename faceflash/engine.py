"""
FaceFlash Engine — the main high-level API.
Combines detection + embedding + quantization + search.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import time

from faceflash.detect import load_image, detect_and_align
from faceflash.embed import FaceEmbedder
from faceflash.index import FaceIndex


class FaceFlash:
    """
    FaceFlash — memory-efficient face retrieval via PCA+ITQ binary quantization.

    100% Recall@1 (matching exact search) at 48-96x less memory than HNSW,
    measured on MS1MV2 from 100K to 1M faces. CPU-only, SIMD-accelerated.

    Usage:
        ff = FaceFlash()
        ff.register("person_name", "photo.jpg")
        result = ff.search("query.jpg")
    """

    def __init__(self, index_path: Optional[str] = None,
                 n_bits: int = 512, n_candidates: int = 100):
        """
        Args:
            n_bits: binary code length. Memory = n_bits/8 bytes per face.
                    512 is exact-capable; 256 halves memory at ~99% recall.
                    See the operating-point table in the README.
            n_candidates: rerank shortlist size — the search-effort knob,
                          independent of k. More candidates = higher recall
                          and more float reranks (= more mmap reads on edge).
        """
        self._embedder: Optional[FaceEmbedder] = None
        self.n_candidates = n_candidates
        self.index = FaceIndex(n_bits=n_bits)
        if index_path and Path(index_path).exists():
            self.index.load(index_path)

    @property
    def embedder(self) -> FaceEmbedder:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            self._embedder = FaceEmbedder()
        return self._embedder

    def register(self, name: str, image_path: str) -> dict:
        """
        Register a person's face in the index.

        Args:
            name: Person identifier
            image_path: Path to face image

        Returns:
            dict with embedding info
        """
        img = load_image(image_path)
        face = detect_and_align(img)
        embedding = self.embedder.embed(face)
        self.index.add(embedding, name, image_path)
        return {"name": name, "registered": True, "index_size": self.index.count}

    def register_folder(self, folder_path: str, progress: bool = True) -> dict:
        """
        Register all faces in a folder.
        Folder structure: folder/person_name/photo1.jpg, photo2.jpg, ...

        Or flat: folder/person_name_001.jpg (name derived from filename)
        """
        folder = Path(folder_path)
        registered = 0
        errors = 0

        items = list(folder.rglob("*.jpg")) + list(folder.rglob("*.png")) + list(folder.rglob("*.jpeg"))

        if progress:
            from tqdm import tqdm
            items = tqdm(items, desc="Indexing faces")

        for img_path in items:
            # Derive name from parent folder or filename
            if img_path.parent != folder:
                name = img_path.parent.name
            else:
                name = img_path.stem.rsplit("_", 1)[0]

            try:
                self.register(name, str(img_path))
                registered += 1
            except Exception as e:
                errors += 1
                if not progress:
                    print(f"  Warning: failed to register {img_path}: {e}")

        return {"registered": registered, "errors": errors, "total": self.index.count}

    def search(self, image_path: str, k: int = 1, threshold: float = 0.4,
               n_candidates: Optional[int] = None) -> dict:
        """
        Search for a face in the index.

        Args:
            image_path: Path to query image
            k: Number of results
            threshold: Minimum cosine similarity to consider a match
            n_candidates: override the instance default search-effort knob

        Returns:
            List of match dicts with name, confidence, time_ms
        """
        start = time.perf_counter()

        img = load_image(image_path)
        face = detect_and_align(img)
        embedding = self.embedder.embed(face)

        embed_time = time.perf_counter() - start

        search_start = time.perf_counter()
        results = self.index.search(embedding, k=k, n_candidates=n_candidates or self.n_candidates)
        search_time = time.perf_counter() - search_start

        total_time = time.perf_counter() - start

        matches = []
        for label, similarity, idx in results:
            if similarity >= threshold:
                matches.append({
                    "name": label,
                    "confidence": round(float(similarity), 4),
                    "index": idx,
                })

        return {
            "matches": matches,
            "time_ms": round(total_time * 1000, 2),
            "embed_time_ms": round(embed_time * 1000, 2),
            "search_time_ms": round(search_time * 1000, 2),
            "index_size": self.index.count,
        }

    def verify(self, image1: str, image2: str) -> dict:
        """
        Verify if two images are the same person.

        Returns:
            dict with match (bool), confidence (float), time_ms
        """
        start = time.perf_counter()

        img1 = load_image(image1)
        face1 = detect_and_align(img1)
        emb1 = self.embedder.embed(face1)

        img2 = load_image(image2)
        face2 = detect_and_align(img2)
        emb2 = self.embedder.embed(face2)

        similarity = float(np.dot(emb1, emb2))
        total_time = time.perf_counter() - start

        return {
            "match": similarity > 0.4,
            "confidence": round(similarity, 4),
            "time_ms": round(total_time * 1000, 2),
        }

    def save(self, path: str):
        """Save the index to disk."""
        self.index.save(path)

    def load(self, path: str):
        """Load an index from disk."""
        self.index.load(path)

    def stats(self) -> dict:
        """Return index statistics."""
        return self.index.stats()

    def names(self) -> list:
        """Return the sorted list of registered names."""
        return self.index.names()

    def remove(self, name: str) -> int:
        """Unregister a person — removes all their faces. Returns the count removed."""
        return self.index.remove(name)

    def __len__(self) -> int:
        """Number of registered faces."""
        return len(self.index)

    def __contains__(self, name: str) -> bool:
        """`name in ff` — whether this person is registered."""
        return name in self.index

    def __repr__(self) -> str:
        return (f"FaceFlash(registered={len(self.index)}, "
                f"people={len(self.index.names())})")
