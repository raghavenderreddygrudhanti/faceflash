"""Tests for FaceIndex — buffered add, input validation, search correctness."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pytest
from faceflash.index import FaceIndex


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def random_embedding(rng, dim=512):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestBufferedAdd:
    def test_single_add_works(self, rng):
        idx = FaceIndex(dim=512, n_bits=256)
        for i in range(100):
            idx.add(random_embedding(rng), f'person_{i}')
        assert idx.count == 100
        results = idx.search(random_embedding(rng), k=3)
        assert len(results) == 3

    def test_large_add_triggers_pca(self, rng):
        idx = FaceIndex(dim=512, n_bits=256)
        for i in range(1100):
            idx.add(random_embedding(rng), f'person_{i}')
        # PCA should be fitted after 1024 items
        idx._flush_pending()
        assert idx._pca_fitted
        assert idx.count == 1100
        assert idx.vectors.shape[0] == 1100

    def test_add_batch_matches_single_add(self, rng):
        idx1 = FaceIndex(dim=512, n_bits=256)
        idx2 = FaceIndex(dim=512, n_bits=256)
        embs = np.array([random_embedding(rng) for _ in range(50)], dtype=np.float32)
        labels = [f'p_{i}' for i in range(50)]

        for i in range(50):
            idx1.add(embs[i], labels[i])
        idx2.add_batch(embs, labels)

        assert idx1.count == idx2.count == 50


class TestInputValidation:
    def test_wrong_dim_raises(self):
        idx = FaceIndex(dim=512, n_bits=256)
        with pytest.raises(ValueError, match="Expected"):
            idx.add(np.zeros(256, dtype=np.float32), 'bad')

    def test_batch_wrong_dim_raises(self):
        idx = FaceIndex(dim=512, n_bits=256)
        with pytest.raises(ValueError, match="Expected"):
            idx.add_batch(np.zeros((5, 256), dtype=np.float32), ['a'] * 5)

    def test_batch_length_mismatch_raises(self):
        idx = FaceIndex(dim=512, n_bits=256)
        with pytest.raises(ValueError, match="labels length"):
            idx.add_batch(np.zeros((5, 512), dtype=np.float32), ['a'] * 3)


class TestSearchCorrectness:
    def test_finds_exact_match(self, rng):
        idx = FaceIndex(dim=512, n_bits=512)
        target = random_embedding(rng)
        idx.add(target, 'target')
        for i in range(99):
            idx.add(random_embedding(rng), f'other_{i}')
        results = idx.search(target, k=1)
        assert results[0][0] == 'target'
        assert results[0][1] > 0.99  # cosine similarity near 1

    def test_empty_index_returns_empty(self):
        idx = FaceIndex(dim=512, n_bits=256)
        results = idx.search(np.zeros(512, dtype=np.float32), k=1)
        assert results == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
