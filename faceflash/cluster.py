"""
Coarse clustering (IVF-style) for sub-linear search.

The base index does a full Hamming scan — work grows linearly with the database.
This layer partitions faces into ~sqrt(N) buckets via spherical k-means on the
ArcFace embeddings. At search time we compare the query against the (few thousand)
centroids, pick the closest `n_probe` buckets, and scan only those faces. That
turns an O(N) scan into roughly O(N / n_clusters * n_probe).

Tradeoff: if the true match sits in a bucket we didn't probe, we miss it — so
recall depends on `n_probe`. More probes → higher recall, slower search.

ArcFace embeddings are L2-normalized, so cosine similarity is a dot product and
k-means becomes spherical k-means (assign by max dot, re-normalize centroids).
"""

import numpy as np
from typing import List, Optional


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _spherical_kmeans(X: np.ndarray, k: int, n_iter: int = 10,
                      seed: int = 42) -> np.ndarray:
    """Fit k unit-norm centroids on X (assumed L2-normalized) via Lloyd's.

    Returns centroids (k, dim), L2-normalized. Empty clusters are re-seeded to
    random points so we always return exactly k usable centroids.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    # k-means++-lite: start from k distinct random samples
    centroids = X[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(n_iter):
        # Assign: nearest centroid by dot product (batched to bound memory)
        assign = _assign(X, centroids)
        # Update: mean of members, then renormalize back onto the sphere
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            members = X[assign == c]
            if len(members):
                new_centroids[c] = members.mean(axis=0)
            else:
                new_centroids[c] = X[rng.integers(n)]  # re-seed empty cluster
        new_centroids = _normalize(new_centroids)
        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < 1e-4:
            break
    return centroids


def _assign(X: np.ndarray, centroids: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Assign each row of X to the nearest centroid (max dot). Batched."""
    out = np.empty(X.shape[0], dtype=np.int32)
    cT = centroids.T
    for i in range(0, X.shape[0], batch):
        sims = X[i:i + batch] @ cT
        out[i:i + batch] = sims.argmax(axis=1)
    return out


class CoarseCluster:
    """IVF-style coarse quantizer over face embeddings."""

    def __init__(self, n_clusters: Optional[int] = None, n_probe: int = 8):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.centroids: Optional[np.ndarray] = None        # (K, dim), unit norm
        self.members: List[np.ndarray] = []               # members[c] = face indices

    def build(self, float_vectors: np.ndarray, fit_sample: int = 50000,
              seed: int = 42):
        """Cluster all faces. Centroids are fit on a subsample; every face is
        then assigned to its nearest centroid."""
        n = float_vectors.shape[0]
        if self.n_clusters is None:
            # ~sqrt(N) buckets is the standard IVF heuristic
            self.n_clusters = max(1, int(np.sqrt(n)))
        self.n_clusters = min(self.n_clusters, n)

        X = _normalize(np.asarray(float_vectors, dtype=np.float32))
        rng = np.random.default_rng(seed)
        fit_idx = (rng.choice(n, size=fit_sample, replace=False)
                   if n > fit_sample else np.arange(n))
        self.centroids = _spherical_kmeans(X[fit_idx], self.n_clusters, seed=seed)

        assign = _assign(X, self.centroids)
        self.members = [np.where(assign == c)[0].astype(np.intp)
                        for c in range(self.n_clusters)]
        return self

    def candidate_pool(self, query_embedding: np.ndarray,
                       n_probe: Optional[int] = None) -> np.ndarray:
        """Return the face indices in the `n_probe` closest buckets."""
        if self.centroids is None:
            raise RuntimeError("CoarseCluster.build() must be called first")
        n_probe = n_probe or self.n_probe
        n_probe = min(n_probe, self.n_clusters)
        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.centroids @ q
        top = np.argpartition(-sims, n_probe - 1)[:n_probe] if n_probe < self.n_clusters \
            else np.arange(self.n_clusters)
        if len(top) == 1:
            return self.members[int(top[0])]
        return np.concatenate([self.members[c] for c in top])

    def to_dict(self) -> dict:
        return {"n_clusters": self.n_clusters, "n_probe": self.n_probe}
