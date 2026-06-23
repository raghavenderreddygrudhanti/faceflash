"""Tests for FaceIndex."""
import numpy as np
import sys
sys.path.insert(0, ".")
from faceflash.index import FaceIndex


def test_add_and_search():
    rng = np.random.default_rng(42)
    idx = FaceIndex()

    embs = rng.standard_normal((1100, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    labels = [f"person_{i}" for i in range(1100)]

    idx.add_batch(embs, labels)
    assert idx.count == 1100
    assert idx._pca_fitted is True

    results = idx.search(embs[0], k=1)
    assert results[0][0] == "person_0"
    assert results[0][1] > 0.99


def test_binary_only():
    rng = np.random.default_rng(42)
    idx = FaceIndex()

    embs = rng.standard_normal((1100, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    labels = [f"p_{i}" for i in range(1100)]

    idx.add_batch(embs, labels)
    results = idx.search_binary_only(embs[0], k=1)
    assert results[0][0] == "p_0"


if __name__ == "__main__":
    test_add_and_search()
    test_binary_only()
    print("All tests passed.")
