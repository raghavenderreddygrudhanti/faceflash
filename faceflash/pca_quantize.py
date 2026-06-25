"""
PCA Binary Quantizer for Face Embeddings.

The core insight: ArcFace embeddings concentrate identity-discriminative
information in a few principal directions. PCA aligns quantization boundaries
with those directions, producing binary codes that need fewer rerank
candidates than random projections to achieve the same recall.

No training loop. No hyperparameters. No identity labels needed.
Just fit on a sample of embeddings, then encode.

Uses Rust backend (faceflash_core) for hardware-accelerated Hamming search
when available. Falls back to NumPy otherwise.

Usage:
    quantizer = PCABinaryQuantizer(n_bits=512)
    quantizer.fit(sample_embeddings)          # ~1000 embeddings is enough
    codes = quantizer.encode(database)        # (N, 64) uint8 packed
    query_code = quantizer.encode(query)      # (1, 64) uint8 packed
"""

import numpy as np
from pathlib import Path

# Try to import Rust backend — packaged as faceflash._core, fallback to a
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


class PCABinaryQuantizer:
    """
    PCA-based binary quantization for face embeddings.

    fit() learns the principal directions from a sample.
    encode() projects embeddings onto those directions and takes the sign.
    search() does Hamming filter + cosine rerank in one call.
    """

    def __init__(self, n_bits: int = 512):
        self.n_bits = n_bits
        self.W: np.ndarray | None = None  # (n_bits, dim) projection matrix
        self.threshold: np.ndarray | None = None  # (n_bits,) per-bit threshold
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        """
        Learn PCA + ITQ projection from a sample of embeddings.

        Two stages:
        1. PCA: find principal directions (captures variance structure)
        2. ITQ: find orthogonal rotation that minimizes quantization error
           (spreads variance evenly across bits so each is comparably reliable)

        The rotation R is folded into W (W' = R^T @ W), so online encode
        cost is identical — just sign(x @ W'^T - threshold).

        Args:
            embeddings: (N, dim) float32, L2-normalized.
                        Should have N >= 1024 for full-rank PCA (N > n_bits).
        """
        X = embeddings.astype(np.float64)
        mean = X.mean(axis=0)
        X_centered = X - mean

        # Stage 1: PCA — find top-n_bits principal directions
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = Vt.shape[0]  # min(N, dim)

        dim = embeddings.shape[1]
        if self.n_bits <= n_components:
            pca_W = Vt[:self.n_bits].astype(np.float32)
        else:
            # Not enough samples for full rank — pad with random directions
            rng = np.random.default_rng(42)
            base = Vt[:n_components].astype(np.float32)
            n_extra = self.n_bits - n_components
            extra = rng.standard_normal((n_extra, dim)).astype(np.float32)
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            pca_W = np.vstack([base, extra])

        # Stage 2: ITQ — orthogonal rotation to minimize quantization error
        # Projects data, then iteratively finds R that makes sign(V@R) closest to V@R
        # This spreads variance evenly across bits so each bit is comparably reliable
        V = (X_centered @ pca_W.T).astype(np.float32)  # (N, n_bits)
        R = np.eye(self.n_bits, dtype=np.float32)

        for _ in range(20):  # Procrustes iterations (converges in ~10)
            Z = V @ R
            B = np.sign(Z)
            B[B == 0] = 1.0
            # Optimal R: minimize ||B - V@R||_F via SVD of B^T @ V
            U_r, _, Vt_r = np.linalg.svd((B.T @ V).astype(np.float64), full_matrices=False)
            R = (Vt_r.T @ U_r.T).astype(np.float32)

        # Fold rotation into projection: W' = R^T @ pca_W
        self.W = (R.T @ pca_W).astype(np.float32)

        # Threshold: median of projections for balanced bit marginals
        projections = embeddings @ self.W.T
        self.threshold = np.median(projections, axis=0).astype(np.float32)

        self.fitted = True

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings to packed binary codes.

        Args:
            embeddings: (N, dim) or (dim,) float32

        Returns:
            (N, n_bits//8) uint8 packed binary codes
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before encode()")

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        projected = embeddings @ self.W.T - self.threshold
        binary = (projected > 0).astype(np.uint8)
        return np.packbits(binary, axis=1)

    def search(self, query: np.ndarray, database: np.ndarray,
               database_codes: np.ndarray, k: int = 1,
               n_candidates: int = 50) -> list:
        """
        Two-phase search: Hamming filter → cosine rerank.

        Uses Rust backend for Hamming computation when available (50x faster).
        Falls back to NumPy popcount table otherwise.

        Args:
            query: (dim,) float32 normalized embedding
            database: (N, dim) float32 embeddings (for reranking)
            database_codes: (N, n_bits//8) packed binary codes
            k: number of results to return
            n_candidates: candidates to shortlist before reranking

        Returns:
            List of (index, similarity) tuples, sorted by similarity.
        """
        query_code = self.encode(query).ravel()
        n_cand = min(n_candidates, len(database_codes))

        # Phase 1: Hamming filter (Rust or NumPy)
        if _HAS_RUST:
            candidate_idx = np.asarray(
                _rust.hamming_topk(query_code, database_codes, n_cand)
            ).astype(np.intp)
        else:
            xor = np.bitwise_xor(database_codes, query_code.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            candidate_idx = np.argpartition(distances, n_cand)[:n_cand]

        # Phase 2: Cosine rerank
        cand_vecs = database[candidate_idx]
        cosine_sims = cand_vecs @ query
        top_k_local = np.argsort(-cosine_sims)[:k]

        results = []
        for idx in top_k_local:
            orig_idx = candidate_idx[idx]
            results.append((int(orig_idx), float(cosine_sims[idx])))
        return results

    def save(self, path: str):
        """Save quantizer parameters."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "pca_W.npy", self.W)
        np.save(p / "pca_threshold.npy", self.threshold)

    def load(self, path: str):
        """Load quantizer parameters."""
        p = Path(path)
        self.W = np.load(p / "pca_W.npy")
        self.threshold = np.load(p / "pca_threshold.npy")
        self.n_bits = self.W.shape[0]
        self.fitted = True


# Precomputed popcount lookup table (0-255 → bit count)
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def backend_info() -> str:
    """Return which search backend is active."""
    if _HAS_RUST:
        return "faceflash_core (Rust, POPCNT-accelerated)"
    return "NumPy (fallback — install faceflash-core for 50x faster search)"
