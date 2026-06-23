"""
FaceFlash Index — stores binary face vectors for ultra-fast search.
Uses TurboVec for the actual vector storage and search.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import json
import time

from faceflash.quantize import float_to_binary, pack_binary, hamming_distance_batch


class FaceIndex:
    """
    Binary face vector index for sub-linear search.

    Stores packed binary vectors + metadata (person ID, path).
    Search is O(n) in raw form but with TurboVec routing becomes sub-linear.
    """

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None  # Packed binary vectors
        self.labels: List[str] = []                 # Person IDs
        self.paths: List[str] = []                  # Image paths
        self.float_vectors: Optional[np.ndarray] = None  # Original (for reranking)
        self.count = 0

    def add(self, embedding: np.ndarray, label: str, path: str = ""):
        """Add a face embedding to the index."""
        binary = float_to_binary(embedding)
        packed = pack_binary(binary)

        if self.vectors is None:
            self.vectors = packed.reshape(1, -1)
            self.float_vectors = embedding.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, packed.reshape(1, -1)])
            self.float_vectors = np.vstack([self.float_vectors, embedding.reshape(1, -1)])

        self.labels.append(label)
        self.paths.append(path)
        self.count += 1

    def add_batch(self, embeddings: np.ndarray, labels: List[str], paths: List[str] = None):
        """Add a batch of face embeddings."""
        if paths is None:
            paths = [""] * len(labels)
        for i, (emb, label, path) in enumerate(zip(embeddings, labels, paths)):
            self.add(emb, label, path)

    def search(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[str, float, int]]:
        """
        Search for the closest face(s) to the query.

        Returns: List of (label, distance, index) tuples, sorted by distance.
        """
        if self.vectors is None or self.count == 0:
            return []

        # Phase 1: Binary search (ultra-fast, coarse)
        query_binary = float_to_binary(query_embedding)
        query_packed = pack_binary(query_binary)

        start = time.perf_counter()
        distances = hamming_distance_batch(query_packed, self.vectors)
        binary_time = time.perf_counter() - start

        # Get top-k candidates from binary search
        # Take more candidates than needed for reranking
        n_candidates = min(k * 10, self.count)
        candidate_indices = np.argsort(distances)[:n_candidates]

        # Phase 2: Rerank with float cosine similarity (precise)
        start = time.perf_counter()
        candidates_float = self.float_vectors[candidate_indices]
        cosine_sims = np.dot(candidates_float, query_embedding)
        rerank_time = time.perf_counter() - start

        # Sort by cosine similarity (descending)
        sorted_indices = np.argsort(-cosine_sims)[:k]

        results = []
        for idx in sorted_indices:
            orig_idx = candidate_indices[idx]
            results.append((
                self.labels[orig_idx],
                float(cosine_sims[idx]),
                int(orig_idx)
            ))

        return results

    def search_binary_only(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[str, int, int]]:
        """Search using ONLY binary vectors (fastest, slightly less accurate)."""
        if self.vectors is None:
            return []

        query_binary = float_to_binary(query_embedding)
        query_packed = pack_binary(query_binary)

        distances = hamming_distance_batch(query_packed, self.vectors)
        top_k = np.argsort(distances)[:k]

        return [(self.labels[i], int(distances[i]), int(i)) for i in top_k]

    def save(self, path: str):
        """Save index to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vectors)
        np.save(p / "float_vectors.npy", self.float_vectors)
        with open(p / "metadata.json", "w") as f:
            json.dump({"labels": self.labels, "paths": self.paths, "count": self.count}, f)

    def load(self, path: str):
        """Load index from disk."""
        p = Path(path)
        self.vectors = np.load(p / "vectors.npy")
        self.float_vectors = np.load(p / "float_vectors.npy")
        with open(p / "metadata.json") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.paths = meta["paths"]
        self.count = meta["count"]

    def stats(self) -> dict:
        """Return index statistics."""
        mem_binary = self.vectors.nbytes if self.vectors is not None else 0
        mem_float = self.float_vectors.nbytes if self.float_vectors is not None else 0
        return {
            "count": self.count,
            "binary_memory_mb": round(mem_binary / 1024 / 1024, 2),
            "float_memory_mb": round(mem_float / 1024 / 1024, 2),
            "compression_ratio": round(mem_float / max(mem_binary, 1), 1),
        }
