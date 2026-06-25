"""Quick SIMD benchmark — measures raw Hamming scan speed."""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from faceflash.pca_quantize import backend_info

try:
    import faceflash._core as core
except ImportError:
    import faceflash_core as core

print(f'Backend: {backend_info()}')

# Generate test data (100K faces, 512 bits = 64 bytes each)
rng = np.random.default_rng(42)
n_db = 100000
n_bits = 512
db = rng.integers(0, 256, (n_db, n_bits // 8), dtype=np.uint8)
query = rng.integers(0, 256, n_bits // 8, dtype=np.uint8)

# Warmup
for _ in range(10):
    core.hamming_topk(query, db, 100)

# Benchmark single-threaded
times = []
for _ in range(50):
    t0 = time.perf_counter()
    core.hamming_topk(query, db, 100)
    times.append((time.perf_counter() - t0) * 1000)

print(f'\n100K x 512b hamming_topk(100) — single-threaded:')
print(f'  Mean: {np.mean(times):.3f}ms')
print(f'  P50:  {np.median(times):.3f}ms')
print(f'  P95:  {np.percentile(times, 95):.3f}ms')
print(f'  QPS:  {1000 / np.mean(times):.0f}')

# Benchmark parallel
times_par = []
for _ in range(50):
    t0 = time.perf_counter()
    core.hamming_topk_parallel(query, db, 100)
    times_par.append((time.perf_counter() - t0) * 1000)

print(f'\n100K x 512b hamming_topk_parallel(100) — multi-threaded:')
print(f'  Mean: {np.mean(times_par):.3f}ms')
print(f'  P50:  {np.median(times_par):.3f}ms')
print(f'  P95:  {np.percentile(times_par, 95):.3f}ms')
print(f'  QPS:  {1000 / np.mean(times_par):.0f}')
print(f'  Speedup vs single: {np.mean(times) / np.mean(times_par):.1f}x')
