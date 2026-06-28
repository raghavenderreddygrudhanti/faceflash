"""High-level FaceFlash API. Wraps detection, embedding, and search."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
import time

from faceflash.detect import load_image, detect_and_align
from faceflash.embed import FaceEmbedder
from faceflash.index import FaceIndex
from faceflash.attributes import FaceAttributes

logger = logging.getLogger(__name__)


class FaceFlash:
    """Face search engine. Register faces, search by image, verify pairs."""

    def __init__(self, index_path: Optional[str] = None,
                 n_bits: int = 512, n_candidates: int = 100):
        """
        n_bits: code length. 512 = full recall, 256 = half memory / ~99% recall.
        n_candidates: how many to shortlist before float rerank. Tradeoff knob.
        """
        self._embedder: Optional[FaceEmbedder] = None
        self._attributes: Optional[FaceAttributes] = None
        self.n_candidates = n_candidates
        self.index = FaceIndex(n_bits=n_bits)
        if index_path and Path(index_path).exists():
            self.index.load(index_path)

    @property
    def embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = FaceEmbedder()
        return self._embedder

    @property
    def attributes(self) -> FaceAttributes:
        if self._attributes is None:
            self._attributes = FaceAttributes()
        return self._attributes

    def register(self, name: str, image_path: str) -> dict:
        """Add a face to the gallery."""
        img = load_image(image_path)
        face = detect_and_align(img)
        embedding = self.embedder.embed(face)
        self.index.add(embedding, name, image_path)
        return {"name": name, "registered": True, "index_size": self.index.count}

    def register_folder(self, folder_path: str, progress: bool = True) -> dict:
        """Bulk register. Expects folder/person_name/photo.jpg structure."""
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
                logger.warning("Failed to register %s: %s", img_path, e)

        return {"registered": registered, "errors": errors, "total": self.index.count}

    def analyze(self, image_path: str) -> dict:
        """Detect a face and return age, gender, emotion — no search needed."""
        img = load_image(image_path)
        face = detect_and_align(img)
        attrs = self.attributes.analyze(face)
        return attrs

    def search(self, image_path: str, k: int = 1, threshold: float = 0.4,
               n_candidates: Optional[int] = None,
               with_attributes: bool = False,
               age_filter: Optional[str] = None,
               min_age: Optional[int] = None,
               max_age: Optional[int] = None) -> dict:
        """Identify a face. Returns matches above threshold with timing info."""
        start = time.perf_counter()

        img = load_image(image_path)
        face = detect_and_align(img)
        embedding = self.embedder.embed(face)

        # Age/gender filtering on the query face
        query_attrs = None
        if with_attributes or age_filter or min_age or max_age:
            query_attrs = self.attributes.analyze(face)
            # Apply age filters — skip search entirely if query doesn't pass
            if age_filter == "adult" and query_attrs["age"] < 18:
                return {"matches": [], "filtered": "query is child", "attributes": query_attrs,
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if age_filter == "child" and query_attrs["age"] >= 18:
                return {"matches": [], "filtered": "query is adult", "attributes": query_attrs,
                        "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if min_age and query_attrs["age"] < min_age:
                return {"matches": [], "filtered": f"age {query_attrs['age']} below min_age {min_age}",
                        "attributes": query_attrs, "time_ms": round((time.perf_counter() - start) * 1000, 2)}
            if max_age and query_attrs["age"] > max_age:
                return {"matches": [], "filtered": f"age {query_attrs['age']} above max_age {max_age}",
                        "attributes": query_attrs, "time_ms": round((time.perf_counter() - start) * 1000, 2)}

        embed_time = time.perf_counter() - start

        search_start = time.perf_counter()
        results = self.index.search(embedding, k=k, n_candidates=n_candidates or self.n_candidates)
        search_time = time.perf_counter() - search_start

        total_time = time.perf_counter() - start

        matches = []
        for label, similarity, idx in results:
            if similarity >= threshold:
                match_info = {
                    "name": label,
                    "confidence": round(float(similarity), 4),
                    "index": idx,
                }
                matches.append(match_info)

        result = {
            "matches": matches,
            "time_ms": round(total_time * 1000, 2),
            "embed_time_ms": round(embed_time * 1000, 2),
            "search_time_ms": round(search_time * 1000, 2),
            "index_size": self.index.count,
        }
        if query_attrs:
            result["attributes"] = query_attrs
        return result

    def verify(self, image1: str, image2: str) -> dict:
        """Check if two images are the same person."""
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
        self.index.save(path)

    def load(self, path: str):
        self.index.load(path)

    def stats(self) -> dict:
        return self.index.stats()

    def names(self) -> list:
        return self.index.names()

    def remove(self, name: str) -> int:
        """Remove all faces for a person. Returns count removed."""
        return self.index.remove(name)

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, name: str) -> bool:
        return name in self.index

    def __repr__(self) -> str:
        return f"FaceFlash(faces={len(self.index)}, people={len(self.index.names())})"
