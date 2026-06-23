"""Tests for PCA binary quantizer."""
import numpy as np
import sys
sys.path.insert(0, ".")
from faceflash.pca_quantize import PCABinaryQuantizer


def test_fit_encode():
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((1000, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(embs[:500])
    codes = pca.encode(embs)

    assert codes.shape == (1000, 64)
    assert codes.dtype == np.uint8
    assert pca.fitted is True


def test_search_self():
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((500, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(embs)
    codes = pca.encode(embs)

    results = pca.search(embs[0], embs, codes, k=1, n_candidates=50)
    assert results[0][0] == 0  # Self-match
    assert results[0][1] > 0.99  # Similarity ~1.0


def test_save_load(tmp_path):
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((500, 512)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    pca = PCABinaryQuantizer(n_bits=512)
    pca.fit(embs)

    pca.save(str(tmp_path))

    pca2 = PCABinaryQuantizer()
    pca2.load(str(tmp_path))

    assert np.allclose(pca.W, pca2.W)
    assert pca2.fitted is True


if __name__ == "__main__":
    test_fit_encode()
    test_search_self()
    print("All tests passed.")
