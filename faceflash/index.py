"""
FaceFlash Index — stores binary face vectors for fast search.

Uses PCA+ITQ binary quantization for high-quality codes,
with Hamming distance search (Rust POPCNT) and cosine reranking.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import json

from faceflash.pca_quantize import PCABinaryQuantizer, _POPCOUNT_TABLE

# Try Rust backend
try:
    import faceflash_core as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


class FaceIndex:
    """
    Binary face vector index with two-phase search.

    Stores packed binary vectors + float vectors for reranking.
    Uses PCA+ITQ quantization (fitted after accumulating enough data).
    Falls back to random projection before fit threshold is reached.
    """

    def __init__(self, dim: int = 512, n_bits: int = 512):
        self.vectors: Optional[np.ndarray] = None  # Packed binary (N, n_bits//8)
        self.labels: List[str] = []
        self.paths: List[str] = []
        self.float_vectors: Optional[np.ndarray] = None  # Original for reranking
        self.count = 0
        self.dim = dim
        self.n_bits = n_bits

        # PCA+ITQ quantizer (fitted once we have enough data)
        self.quantizer = PCABinaryQuantizer(n_bits=n_bits)
        self._pca_fitted = False
        self._pca_fit_threshold = 1024  # Need > n_bits samples for full-rank SVD

        # Fallback: random projection (used before PCA is fitted)
        rng = np.random.default_rng(42)
        self.projection = rng.standard_normal((n_bits, dim)).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

    def add(self, embedding: np.ndarray, label: str, path: str = ""):
        """Add a single face embedding to the index."""
        if self._pca_fitted:
            packed = self.quantizer.encode(embedding.reshape(1, -1))[0]
        else:
            binary = ((embedding @ self.projection.T) > 0).astype(np.uint8)
            packed = np.packbits(binary)

        if self.vectors is None:
            self.vectors = packed.reshape(1, -1)
            self.float_vectors = embedding.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, packed.reshape(1, -1)])
            self.float_vectors = np.vstack([self.float_vectors, embedding.reshape(1, -1)])

        self.labels.append(label)
        self.paths.append(path)
        self.count += 1

        # Auto-fit PCA+ITQ once we have enough data
        if not self._pca_fitted and self.count >= self._pca_fit_threshold:
            self._fit_pca()

    def add_batch(self, embeddings: np.ndarray, labels: List[str], paths: List[str] = None):
        """Add a batch of face embeddings (more efficient than repeated add())."""
        if paths is None:
            paths = [""] * len(labels)

        # Store float vectors
        if self.float_vectors is None:
            self.float_vectors = embeddings.astype(np.float32)
        else:
            self.float_vectors = np.vstack([self.float_vectors, embeddings.astype(np.float32)])

        self.labels.extend(labels)
        self.paths.extend(paths)
        self.count += len(labels)

        # Fit PCA if we now have enough data
        if not self._pca_fitted and self.count >= self._pca_fit_threshold:
            self._fit_pca()
        elif self._pca_fitted:
            # Encode new batch with fitted quantizer
            new_codes = self.quantizer.encode(embeddings)
            if self.vectors is None:
                self.vectors = new_codes
            else:
                self.vectors = np.vstack([self.vectors, new_codes])
        else:
            # Fallback: random projection
            binary = (embeddings @ self.projection.T > 0).astype(np.uint8)
            new_codes = np.packbits(binary, axis=1)
            if self.vectors is None:
                self.vectors = new_codes
            else:
                self.vectors = np.vstack([self.vectors, new_codes])

    def _fit_pca(self):
        """Fit PCA+ITQ quantizer on accumulated vectors and re-encode all."""
        self.quantizer.fit(self.float_vectors[:min(5000, self.count)])
        self._pca_fitted = True
        # Re-encode all existing vectors with the fitted quantizer
        self.vectors = self.quantizer.encode(self.float_vectors)

    def search(self, query_embedding: np.ndarray, k: int = 1,
               n_candidates: Optional[int] = None) -> List[Tuple[str, float, int]]:
        """
        Two-phase search: Hamming filter → cosine rerank.
        Uses Rust backend when available (50x faster).

        n_candidates is the search-effort knob, independent of k (the result
        count). More candidates → higher recall + more float reranks. Defaults
        to max(100, k*10), which reaches ≥99% recall in benchmarks.
        """
        if self.vectors is None or self.count == 0:
            return []

        # Encode query
        if self._pca_fitted:
            query_packed = self.quantizer.encode(query_embedding.reshape(1, -1)).ravel()
        else:
            query_binary = ((query_embedding @ self.projection.T) > 0).astype(np.uint8)
            query_packed = np.packbits(query_binary)

        # Phase 1: Hamming filter (Rust or NumPy)
        if n_candidates is None:
            n_candidates = max(100, k * 10)
        n_candidates = min(n_candidates, self.count - 1) if self.count > 1 else 1

        if _HAS_RUST:
            topk = _rust.hamming_topk(query_packed, self.vectors, n_candidates)
            candidate_indices = np.asarray(topk).astype(np.intp)
        else:
            xor = np.bitwise_xor(self.vectors, query_packed.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            candidate_indices = np.argpartition(distances, n_candidates)[:n_candidates]

        # Phase 2: Cosine rerank
        candidates_float = self.float_vectors[candidate_indices]
        cosine_sims = np.dot(candidates_float, query_embedding)

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
        """Search using only binary codes (fastest, lower accuracy)."""
        if self.vectors is None:
            return []

        if self._pca_fitted:
            query_packed = self.quantizer.encode(query_embedding.reshape(1, -1)).ravel()
        else:
            query_binary = ((query_embedding @ self.projection.T) > 0).astype(np.uint8)
            query_packed = np.packbits(query_binary)

        if _HAS_RUST:
            topk = _rust.hamming_topk(query_packed, self.vectors, k)
            indices = np.asarray(topk).astype(np.intp)
            # Get distances for the top-k
            xor = np.bitwise_xor(self.vectors[indices], query_packed.reshape(1, -1))
            dists = _POPCOUNT_TABLE[xor].sum(axis=1)
            return [(self.labels[i], int(d), int(i)) for i, d in zip(indices, dists)]
        else:
            xor = np.bitwise_xor(self.vectors, query_packed.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            top_k = np.argsort(distances)[:k]
            return [(self.labels[i], int(distances[i]), int(i)) for i in top_k]

    def save(self, path: str):
        """Save index + quantizer to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self.vectors)
        np.save(p / "float_vectors.npy", self.float_vectors)
        with open(p / "metadata.json", "w") as f:
            json.dump({
                "labels": self.labels,
                "paths": self.paths,
                "count": self.count,
                "pca_fitted": self._pca_fitted,
            }, f)
        # Persist quantizer if fitted
        if self._pca_fitted:
            self.quantizer.save(str(p / "quantizer"))

    def load(self, path: str):
        """Load index + quantizer from disk."""
        p = Path(path)
        self.vectors = np.load(p / "vectors.npy")
        self.float_vectors = np.load(p / "float_vectors.npy", mmap_mode='r')
        with open(p / "metadata.json") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.paths = meta["paths"]
        self.count = meta["count"]
        self._pca_fitted = meta.get("pca_fitted", False)
        # Load quantizer if it was persisted
        quantizer_path = p / "quantizer"
        if quantizer_path.exists() and self._pca_fitted:
            self.quantizer.load(str(quantizer_path))

    def stats(self) -> dict:
        """Return index statistics."""
        mem_binary = self.vectors.nbytes if self.vectors is not None else 0
        # Float vectors may be mmap'd (on-disk, paged on demand) or resident
        float_is_mmap = hasattr(self.float_vectors, 'filename') and self.float_vectors.filename is not None
        float_bytes = self.float_vectors.nbytes if self.float_vectors is not None else 0
        return {
            "count": self.count,
            "pca_fitted": self._pca_fitted,
            "rust_backend": _HAS_RUST,
            "binary_memory_mb": round(mem_binary / 1024 / 1024, 2),
            "float_storage_mb": round(float_bytes / 1024 / 1024, 2),
            "float_mode": "mmap (on-disk, paged on demand)" if float_is_mmap else "resident (in RAM)",
            "resident_memory_mb": round(mem_binary / 1024 / 1024, 2) if float_is_mmap else round((mem_binary + float_bytes) / 1024 / 1024, 2),
            "compression_ratio": round(float_bytes / max(mem_binary, 1), 1),
        }
