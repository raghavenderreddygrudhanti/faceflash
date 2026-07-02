"""
PCA+ITQ quantizer. Compresses 512-d ArcFace embeddings to binary codes.

Key property: ArcFace identity information is concentrated in top PCA
directions, so 512 binary projections lose almost nothing for NN retrieval.
ITQ rotates bits for balanced marginals (fewer wasted bits).
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
    """Binary quantizer. fit() on a sample, encode() to get packed codes."""

    def __init__(self, n_bits: int = 512):
        self.n_bits = n_bits
        self.W: np.ndarray | None = None
        self.threshold: np.ndarray | None = None
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        """
        PCA + ITQ from a sample. ~1000 embeddings is plenty.
        R is folded into W so encode() is just sign(x @ W.T - threshold).
        """
        X = embeddings.astype(np.float64)
        mean = X.mean(axis=0)
        X_centered = X - mean

        # PCA
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_components = Vt.shape[0]

        dim = embeddings.shape[1]
        if self.n_bits <= n_components:
            pca_W = Vt[:self.n_bits].astype(np.float32)
        else:
            # Pad with random directions when we don't have enough samples
            rng = np.random.default_rng(42)
            base = Vt[:n_components].astype(np.float32)
            n_extra = self.n_bits - n_components
            extra = rng.standard_normal((n_extra, dim)).astype(np.float32)
            extra /= np.linalg.norm(extra, axis=1, keepdims=True)
            pca_W = np.vstack([base, extra])

        # ITQ: Procrustes rotation to balance bit marginals
        V = (X_centered @ pca_W.T).astype(np.float32)
        R = np.eye(self.n_bits, dtype=np.float32)

        for _ in range(20):
            Z = V @ R
            B = np.sign(Z)
            B[B == 0] = 1.0
            U_r, _, Vt_r = np.linalg.svd((B.T @ V).astype(np.float64), full_matrices=False)
            R = (Vt_r.T @ U_r.T).astype(np.float32)

        # Fold R into W so encode is a single matmul
        self.W = (R.T @ pca_W).astype(np.float32)

        # Median threshold → balanced bits
        projections = embeddings @ self.W.T
        self.threshold = np.median(projections, axis=0).astype(np.float32)

        self.fitted = True

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """Project + sign + packbits. Returns (N, n_bits//8) uint8."""
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
        """Hamming shortlist → cosine rerank. Returns [(index, similarity), ...]."""
        query_code = self.encode(query).ravel()
        n_cand = min(n_candidates, len(database_codes))

        # Hamming filter (Rust or NumPy)
        if _HAS_RUST:
            candidate_idx = np.asarray(
                _rust.hamming_topk(query_code, database_codes, n_cand)
            ).astype(np.intp)
        else:
            candidate_idx = hamming_topk_numpy(query_code, database_codes, n_cand)

        # Cosine rerank
        cand_vecs = database[candidate_idx]
        cosine_sims = cand_vecs @ query
        top_k_local = np.argsort(-cosine_sims)[:k]

        results = []
        for idx in top_k_local:
            orig_idx = candidate_idx[idx]
            results.append((int(orig_idx), float(cosine_sims[idx])))
        return results

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "pca_W.npy", self.W)
        np.save(p / "pca_threshold.npy", self.threshold)

    def load(self, path: str):
        p = Path(path)
        self.W = np.load(p / "pca_W.npy")
        self.threshold = np.load(p / "pca_threshold.npy")
        self.n_bits = self.W.shape[0]
        self.fitted = True


# Popcount LUT for NumPy fallback path
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def hamming_topk_numpy(query_code: np.ndarray, database_codes: np.ndarray,
                       k: int) -> np.ndarray:
    """Exact Hamming top-k with deterministic (distance, index) tie-breaking.

    The Rust kernel breaks ties at the k boundary by index; argpartition
    breaks them arbitrarily. Packing (distance, index) into one integer key
    makes both backends return the identical candidate set, so results don't
    depend on which backend is installed.
    """
    xor = np.bitwise_xor(database_codes, query_code.reshape(1, -1))
    dists = _POPCOUNT_TABLE[xor].sum(axis=1)
    n = len(dists)
    k = min(k, n)
    key = dists.astype(np.int64) * n + np.arange(n, dtype=np.int64)
    idx = np.argpartition(key, k - 1)[:k]
    return idx[np.argsort(key[idx])].astype(np.intp)


def backend_info() -> str:
    if _HAS_RUST:
        return "faceflash._core (Rust, SIMD-accelerated)"
    return "NumPy (fallback — Rust backend not built; reinstall from a wheel for SIMD search)"
