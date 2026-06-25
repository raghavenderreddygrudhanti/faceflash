"""Quick test for O(n) add() fix and input validation."""
import sys
sys.path.insert(0, '.')
import numpy as np
from faceflash.index import FaceIndex

def test_buffered_add():
    idx = FaceIndex(dim=512, n_bits=256)
    rng = np.random.default_rng(42)
    for i in range(500):
        v = rng.standard_normal(512).astype(np.float32)
        v /= np.linalg.norm(v)
        idx.add(v, f'person_{i}')
    query = rng.standard_normal(512).astype(np.float32)
    query /= np.linalg.norm(query)
    results = idx.search(query, k=3)
    assert len(results) == 3
    assert idx.count == 500
    assert idx.vectors.shape[0] == 500
    print(f'OK: buffered add + search works ({idx.count} items)')

def test_input_validation():
    idx = FaceIndex(dim=512, n_bits=256)
    try:
        idx.add(np.zeros(256), 'bad')
        assert False, "Should have raised"
    except ValueError:
        pass
    try:
        idx.add_batch(np.zeros((5, 256)), ['a']*5)
        assert False, "Should have raised"
    except ValueError:
        pass
    print('OK: input validation works')

if __name__ == '__main__':
    test_buffered_add()
    test_input_validation()
    print('ALL TESTS PASS')
