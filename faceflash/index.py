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
from faceflash.cluster import CoarseCluster

# Try Rust backend — packaged as faceflash._core, with a fallback to a
# top-level faceflash_core build (legacy / `maturin develop` dev workflow).
try:
    from faceflash import _core as _rust
    _HAS_RUST = True
except ImportError:
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

        # Pending buffer — avoids O(n²) vstack on every add()
        self._pending_embeddings: List[np.ndarray] = []
        self._pending_labels: List[str] = []
        self._pending_paths: List[str] = []

        # PCA+ITQ quantizer (fitted once we have enough data)
        self.quantizer = PCABinaryQuantizer(n_bits=n_bits)
        self._pca_fitted = False
        self._pca_fit_threshold = 1024  # Need > n_bits samples for full-rank SVD

        # Fallback: random projection (used before PCA is fitted)
        rng = np.random.default_rng(42)
        self.projection = rng.standard_normal((n_bits, dim)).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)

        # Optional coarse clustering (IVF) for sub-linear scan. None = full scan.
        self.cluster: Optional[CoarseCluster] = None

    def _flush_pending(self):
        """Flush buffered embeddings into the main arrays."""
        if not self._pending_embeddings:
            return
        batch = np.array(self._pending_embeddings, dtype=np.float32)
        labels = self._pending_labels[:]
        paths = self._pending_paths[:]
        self._pending_embeddings.clear()
        self._pending_labels.clear()
        self._pending_paths.clear()
        # Use internal _add_batch_raw to avoid double-counting
        self._add_batch_raw(batch, labels, paths)

    def add(self, embedding: np.ndarray, label: str, path: str = ""):
        """Add a single face embedding to the index.

        Buffers internally and flushes in batches for O(n) performance.
        The index is always searchable — pending items are flushed
        automatically before search or when the PCA fit threshold is reached.
        """
        if embedding.ndim != 1 or embedding.shape[0] != self.dim:
            raise ValueError(f"Expected ({self.dim},) embedding, got {embedding.shape}")

        self._pending_embeddings.append(embedding)
        self._pending_labels.append(label)
        self._pending_paths.append(path)
        self.count += 1

        # Flush when buffer is large enough or PCA threshold reached
        if len(self._pending_embeddings) >= 256 or \
           (not self._pca_fitted and self.count >= self._pca_fit_threshold):
            self._flush_pending()

    def add_batch(self, embeddings: np.ndarray, labels: List[str], paths: Optional[List[str]] = None):
        """Add a batch of face embeddings (more efficient than repeated add())."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (N, {self.dim}) embeddings, got {embeddings.shape}")
        if len(labels) != len(embeddings):
            raise ValueError(f"labels length ({len(labels)}) != embeddings length ({len(embeddings)})")
        if paths is None:
            paths = [""] * len(labels)
        self.count += len(labels)
        self._add_batch_raw(embeddings, labels, paths)

    def _add_batch_raw(self, embeddings: np.ndarray, labels: List[str], paths: List[str]):
        """Internal: add batch without incrementing count (used by _flush_pending)."""

        # Store float vectors
        if self.float_vectors is None:
            self.float_vectors = embeddings.astype(np.float32)
        else:
            self.float_vectors = np.vstack([self.float_vectors, embeddings.astype(np.float32)])

        self.labels.extend(labels)
        self.paths.extend(paths)

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

    def build_clusters(self, n_clusters: Optional[int] = None, n_probe: int = 8):
        """Build a coarse-clustering (IVF) layer for sub-linear search.

        Partitions faces into ~sqrt(N) buckets; search then scans only the
        `n_probe` closest buckets instead of the whole database. This trades a
        little recall (the match may sit in an unprobed bucket) for a large
        speedup that stops growing with database size. Defaults: ~sqrt(N)
        clusters, 8 probes. Call after all faces are added.
        """
        self._flush_pending()
        if self.float_vectors is None or self.count == 0:
            raise RuntimeError("Add faces before building clusters")
        self.cluster = CoarseCluster(n_clusters=n_clusters, n_probe=n_probe)
        self.cluster.build(self.float_vectors)
        return self

    def search(self, query_embedding: np.ndarray, k: int = 1,
               n_candidates: Optional[int] = None,
               n_probe: Optional[int] = None) -> List[Tuple[str, float, int]]:
        """
        Two-phase search: Hamming filter → cosine rerank.
        Uses the Rust SIMD backend when available (much faster than NumPy).

        n_candidates is the search-effort knob, independent of k (the result
        count). More candidates → higher recall + more float reranks. Defaults
        to max(100, k*10), which reaches ≥99% recall in benchmarks.
        """
        # Flush any buffered add() calls before searching
        self._flush_pending()

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

        # Coarse clustering: restrict the scan to the closest buckets (sub-linear).
        # Without clustering, `pool` spans the whole database (full scan).
        if self.cluster is not None:
            pool = self.cluster.candidate_pool(query_embedding, n_probe=n_probe)
            scan_vectors = self.vectors[pool]
            n_cand = min(n_candidates, len(pool))
        else:
            pool = None
            scan_vectors = self.vectors
            n_cand = n_candidates

        if _HAS_RUST:
            topk = _rust.hamming_topk(query_packed, scan_vectors, n_cand)
            local_indices = np.asarray(topk).astype(np.intp)
        else:
            xor = np.bitwise_xor(scan_vectors, query_packed.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            local_indices = np.argpartition(distances, min(n_cand, len(distances) - 1))[:n_cand]

        # Map bucket-local indices back to global face indices
        candidate_indices = pool[local_indices] if pool is not None else local_indices

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

    def search_batch(self, query_embeddings: np.ndarray, k: int = 1,
                     n_candidates: Optional[int] = None,
                     parallel: bool = True) -> List[List[Tuple[str, float, int]]]:
        """Search many queries in one shot.

        Equivalent to calling search() per row, but the Hamming phase crosses
        the Python<->Rust boundary once for the whole batch instead of once per
        query. With parallel=True the queries are scanned across CPU cores
        (embarrassingly parallel). This is the throughput path -- use it for
        benchmarks and bulk identification; use search() for single lookups.

        Coarse clustering (IVF) is not applied here: batching targets full-scan
        throughput. Call search() per query if you need the clustered path.
        """
        self._flush_pending()
        if self.vectors is None or self.count == 0:
            return [[] for _ in range(len(query_embeddings))]

        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected (M, {self.dim}) queries, got {query_embeddings.shape}")
        m = query_embeddings.shape[0]

        # Encode all queries
        if self._pca_fitted:
            q_packed = self.quantizer.encode(query_embeddings)
        else:
            q_binary = (query_embeddings @ self.projection.T > 0).astype(np.uint8)
            q_packed = np.packbits(q_binary, axis=1)
        q_packed = np.ascontiguousarray(q_packed, dtype=np.uint8)

        if n_candidates is None:
            n_candidates = max(100, k * 10)
        n_cand = min(n_candidates, self.count - 1) if self.count > 1 else 1

        # Phase 1: one batched Hamming call -> (m, n_cand) candidate indices
        if _HAS_RUST:
            flat = _rust.hamming_topk_batch(q_packed, self.vectors, n_cand, parallel)
            cand = np.asarray(flat).astype(np.intp).reshape(m, -1)
        else:
            xor = np.bitwise_xor(self.vectors[None, :, :], q_packed[:, None, :])
            dists = _POPCOUNT_TABLE[xor].sum(axis=2)
            cand = np.argpartition(dists, min(n_cand, self.count - 1), axis=1)[:, :n_cand]

        # Phase 2: cosine rerank per query (vectorized over candidates)
        results: List[List[Tuple[str, float, int]]] = []
        for i in range(m):
            ci = cand[i]
            sims = self.float_vectors[ci] @ query_embeddings[i]
            order = np.argsort(-sims)[:k]
            results.append([
                (self.labels[ci[j]], float(sims[j]), int(ci[j])) for j in order
            ])
        return results

    def search_binary_only(self, query_embedding: np.ndarray, k: int = 1) -> List[Tuple[str, int, int]]:
        """Search using only binary codes (fastest, lower accuracy)."""
        self._flush_pending()
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
        self._flush_pending()
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
        # Persist coarse clusters if built (centroids + per-face assignment)
        if self.cluster is not None and self.cluster.centroids is not None:
            assign = np.empty(self.count, dtype=np.int32)
            for c, members in enumerate(self.cluster.members):
                assign[members] = c
            np.savez(p / "clusters.npz",
                     centroids=self.cluster.centroids,
                     assign=assign,
                     n_probe=self.cluster.n_probe)

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
        # Restore coarse clusters if present
        cl_path = p / "clusters.npz"
        if cl_path.exists():
            d = np.load(cl_path)
            self.cluster = CoarseCluster(n_clusters=int(d["centroids"].shape[0]),
                                         n_probe=int(d["n_probe"]))
            self.cluster.centroids = d["centroids"]
            assign = d["assign"]
            self.cluster.members = [np.where(assign == c)[0].astype(np.intp)
                                    for c in range(self.cluster.centroids.shape[0])]

    def stats(self) -> dict:
        """Return index statistics."""
        self._flush_pending()
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
