"""
Comprehensive tests for the FaceIndex core: persistence, mmap, buffered add,
input validation, the NumPy fallback path, and edge cases.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import faceflash.index as index_module
from faceflash.index import FaceIndex


def _unit(n, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _build(n=1200, n_bits=512):
    idx = FaceIndex(n_bits=n_bits)
    embs = _unit(n)
    idx.add_batch(embs, [f"p{i}" for i in range(n)])
    return idx, embs


# ── persistence ──────────────────────────────────────────────────────────────
def test_save_load_roundtrip(tmp_path):
    idx, embs = _build()
    idx.save(str(tmp_path))

    loaded = FaceIndex()
    loaded.load(str(tmp_path))
    assert loaded.count == idx.count
    assert loaded._pca_fitted == idx._pca_fitted
    # search after load returns the self-match
    res = loaded.search(embs[42], k=1)
    assert res[0][0] == "p42"
    assert res[0][1] > 0.99


def test_quantizer_persisted(tmp_path):
    """The fitted PCA quantizer must survive save/load or query encoding breaks."""
    idx, embs = _build()
    assert idx._pca_fitted
    idx.save(str(tmp_path))
    loaded = FaceIndex()
    loaded.load(str(tmp_path))
    assert loaded.quantizer.fitted
    np.testing.assert_allclose(loaded.quantizer.W, idx.quantizer.W, rtol=1e-5)


def test_float_vectors_mmapped_after_load(tmp_path):
    idx, _ = _build()
    idx.save(str(tmp_path))
    loaded = FaceIndex()
    loaded.load(str(tmp_path))
    assert isinstance(loaded.float_vectors, np.memmap)  # not resident in RAM


# ── buffered single add() ────────────────────────────────────────────────────
def test_buffered_single_add():
    idx = FaceIndex()
    embs = _unit(1200)
    for i in range(len(embs)):
        idx.add(embs[i], f"p{i}")
    assert idx.count == 1200
    res = idx.search(embs[7], k=1)        # search flushes the buffer
    assert res[0][0] == "p7"
    assert idx.vectors.shape[0] == 1200    # everything flushed


def test_add_then_search_before_flush_threshold():
    """Fewer than the buffer size: still searchable (search flushes)."""
    idx = FaceIndex()
    embs = _unit(50)
    for i in range(len(embs)):
        idx.add(embs[i], f"p{i}")
    res = idx.search(embs[3], k=1)
    assert res[0][0] == "p3"


# ── input validation ─────────────────────────────────────────────────────────
def test_add_wrong_dim_raises():
    idx = FaceIndex(dim=512)
    with pytest.raises(ValueError):
        idx.add(np.zeros(256, dtype=np.float32), "bad")


def test_add_batch_wrong_shape_raises():
    idx = FaceIndex(dim=512)
    with pytest.raises(ValueError):
        idx.add_batch(np.zeros((5, 256), dtype=np.float32), ["a"] * 5)


def test_add_batch_label_length_mismatch_raises():
    idx = FaceIndex(dim=512)
    with pytest.raises(ValueError):
        idx.add_batch(np.zeros((5, 512), dtype=np.float32), ["a"] * 3)


# ── NumPy fallback (no Rust) ──────────────────────────────────────────────────
def test_numpy_fallback_matches(monkeypatch):
    idx, embs = _build()
    monkeypatch.setattr(index_module, "_HAS_RUST", False)
    res = idx.search(embs[100], k=1)       # forces the NumPy popcount path
    assert res[0][0] == "p100"
    assert res[0][1] > 0.99


# ── edge cases ───────────────────────────────────────────────────────────────
def test_empty_index_returns_empty():
    assert FaceIndex().search(_unit(1)[0]) == []


@pytest.mark.parametrize("n_bits", [256, 384, 512])
def test_n_bits_variants(n_bits):
    idx, embs = _build(n=1200, n_bits=n_bits)
    assert idx.vectors.shape[1] == n_bits // 8
    res = idx.search(embs[5], k=1)
    assert res[0][0] == "p5"


def test_topk_returns_k_results():
    idx, embs = _build()
    res = idx.search(embs[0], k=5)
    assert len(res) == 5
    assert res[0][0] == "p0"               # nearest is itself


def test_search_binary_only():
    idx, embs = _build()
    res = idx.search_binary_only(embs[9], k=1)
    assert res[0][0] == "p9"


def test_stats_keys():
    idx, _ = _build()
    s = idx.stats()
    for key in ("count", "binary_memory_mb", "compression_ratio", "rust_backend"):
        assert key in s
    assert s["count"] == 1200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
