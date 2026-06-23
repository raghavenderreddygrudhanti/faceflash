# FaceFlash

**Fast face retrieval via PCA+ITQ binary quantization + Rust POPCNT search.**

~10x faster than DeepFace/InsightFace. 48x less committed memory than HNSW at equal recall. Best at 10K–100K scale.

```bash
pip install faceflash
```

## Quick Start

```python
from faceflash import FaceFlash

ff = FaceFlash()

# Register faces
ff.register("Alice", "alice.jpg")
ff.register("Bob", "bob.jpg")

# Search (returns in sub-millisecond)
result = ff.search("query.jpg")
print(result)
# → {"matches": [{"name": "Alice", "confidence": 0.92}], "search_time_ms": 0.5}

# Verify two faces
result = ff.verify("photo1.jpg", "photo2.jpg")
print(result)
# → {"match": True, "confidence": 0.87, "time_ms": 45.2}

# Index entire folder
ff.register_folder("employees/")  # folder/person_name/photo.jpg
ff.save("my_index/")
```

## How It Works

```
Image → Face Detection → ArcFace (512-dim) → PCA Binary Quantization (64 bytes)
                                                        ↓
Query → Same pipeline → Hamming Search (Rust POPCNT) → Top-K Rerank (cosine) → Match
```

1. **Detect** — Find face in image (OpenCV)
2. **Embed** — ArcFace converts face to 512-dim vector (ONNX, CPU)
3. **Quantize** — PCA-aligned binary codes (64 bytes, fitted on your data)
4. **Search** — Hamming distance scan via Rust POPCNT (50x faster than NumPy)
5. **Rerank** — Top candidates verified with exact cosine similarity

---

## Benchmarks

All benchmarks: single-threaded, Apple M2 Pro, `time.perf_counter()` per query, warmup excluded.

### Tier 1: vs Vector Search Engines (VGGFace2, 100K real faces)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FAISS-Flat (exact) | 100% | 4.04ms | 248 | 195 MB |
| FAISS-IVF (nprobe=16) | 97.0% | 0.66ms | 1,526 | 195 MB |
| HNSWLIB (ef=32) | 99.0% | 0.44ms | 2,249 | ~292 MB |
| HNSWLIB (ef=64) | 100% | 0.73ms | 1,362 | ~292 MB |
| USearch (HNSW) | 99.0% | 0.21ms | 4,805 | ~253 MB |
| NMSLIB (ef=32) | 97.0% | 0.13ms | — | ~292 MB |
| NMSLIB (ef=64) | 100.0% | 0.21ms | — | ~292 MB |
| **FaceFlash (top-100)** | **99.5%** | **0.49ms** | **2,042** | **6 MB ᵃ** |
| **FaceFlash (top-300)** | **100%** | **0.55ms** | **1,826** | **6 MB ᵃ** |

ᵃ 6 MB = binary index (committed/resident). Float vectors for reranking (+195 MB) are mmap'd from disk — see Limitations.

**At equal 100% recall: FaceFlash (0.55ms) beats HNSWLIB ef=64 (0.73ms) while using 48x less committed memory.**
USearch/NMSLIB are faster (0.21ms) but require 40–48x more RAM and don't guarantee 100% recall at that setting.

### Tier 2: Additional Baselines (VGGFace2, 100K)

| Method | Recall@1 | Latency | Notes |
|---|---|---|---|
| Annoy (100 trees) | 0% | 0.03ms | Broken with dot product at this scale |
| NMSLIB (ef=64) | 100.0% | 0.21ms | Graph-based, high memory |
| DiskANN | — | — | Linux-only, not tested |
| ScaNN | — | — | Linux-only, not tested |

### Tier 3: vs Face Libraries (100K faces, retrieval speed)

DeepFace and InsightFace both use brute-force cosine for search internally.

| Method | Recall@1 | Latency | Speedup |
|---|---|---|---|
| DeepFace (brute force) | 100% | 4.85ms | 1x |
| InsightFace (brute force) | 100% | 4.67ms | 1x |
| **FaceFlash (top-100)** | **99.5%** | **0.49ms** | **9.9x** |
| **FaceFlash (top-300)** | **100%** | **0.57ms** | **8.5x** |

FaceFlash is a drop-in replacement for the search step in DeepFace/InsightFace. Same embeddings, ~10x faster retrieval.

---

### LFW — 13K faces, 5,749 identities (accuracy validation)

| Method | R@1 | R@10 | R@100 | Latency | Memory |
|---|---|---|---|---|---|
| FAISS-Flat (exact) | 100% | 100% | 100% | 0.56ms | 25.3 MB |
| FAISS-IVF | 94.3% | 94.3% | 94.3% | 0.15ms | 25.3 MB |
| **FaceFlash (30 cands)** | **99.0%** | **99.0%** | — | **0.05ms** | **0.8 MB** |
| FaceFlash (100 cands) | 99.7% | 99.7% | 99.7% | 0.08ms | 0.8 MB |
| FaceFlash (500 cands) | 100% | 100% | 100% | 0.17ms | 0.8 MB |

Timing breakdown (30 candidates): binary scan 0.039ms + cosine rerank 0.010ms.

### VGGFace2 — 100K faces, 291 identities (scale validation)

| Candidates | Recall@1 | Latency | Speedup vs Brute Force |
|---|---|---|---|
| 50 | 95.5% | 0.33ms | 11.6x |
| 100 | 97.3% | 0.34ms | 11.2x |
| 200 | 98.3% | 0.37ms | 10.4x |
| 300 | 98.6% | 0.40ms | 9.7x |
| 500 | 98.6% | 0.45ms | 8.5x |
| 1000 | 99.3% | 0.60ms | 6.4x |

---

### vs HNSWLIB (LFW, 3K faces)

| Method | Recall@1 | Latency | Memory |
|---|---|---|---|
| HNSWLIB (ef=32) | 98.5% | 0.348ms | 8.2 MB |
| HNSWLIB (ef=64) | 100% | 0.533ms | 8.2 MB |
| **FaceFlash (100 cands)** | **97.0%** | **0.030ms** | **0.2 MB** |
| FaceFlash (300 cands) | 99.5% | 0.056ms | 0.2 MB |

FaceFlash: 9x faster than HNSWLIB, 40x less memory.

---

### Model Agnosticism

FaceFlash works with any 512-dim face embedding model:

| Model | Size | R@1 (100 cands) | R@1 (300 cands) |
|---|---|---|---|
| ArcFace-R50 (w600k) | 166 MB | 97.5% | 99.5% |
| Buffalo-L (R50) | 166 MB | 97.0% | 99.5% |
| Buffalo-S (MobileFaceNet) | 13 MB | 97.0% | 99.5% |
| FaceNet (InceptionResNet) | 90 MB | 94.5% | 98.5% |

Swap the embedding model without changing search performance. FaceNet has slightly lower recall due to different embedding distribution, but still works well.

---

### Latency Scaling (Rust backend)

Validates that search latency scales linearly with database size.
Database: real LFW embeddings + random unit vector padding.
Random vectors are valid for latency measurement (XOR + POPCNT cost is content-independent).

**This table does NOT validate recall at scale.** Recall is validated on VGGFace2 (100K real faces, see above).

| Scale | Binary Scan | Total (scan + rerank) | Speedup vs FAISS | Binary Memory |
|---|---|---|---|---|
| 13K | 0.06ms | 0.08ms | 7x | 0.8 MB |
| 50K | 0.22ms | 0.24ms | 8x | 3 MB |
| 100K | 0.44ms | 0.46ms | 9x | 6 MB |
| 500K | 2.18ms | 2.21ms | 9x | 31 MB |
| 1M | 5.65ms | 5.68ms | 7x | 61 MB |

Latency scales linearly. Memory = 64 bytes per face. ~8x faster than FAISS-Flat.

---

## Architecture

```
faceflash/
├── engine.py          # High-level API (register, search, verify)
├── detect.py          # Face detection (OpenCV)
├── embed.py           # ArcFace ONNX (512-dim, auto-downloads)
├── index.py           # Binary index (auto-fits PCA after 100 vectors)
├── pca_quantize.py    # PCA quantizer + Rust/NumPy backend
rust/
├── src/lib.rs         # Hamming search (POPCNT + Rayon)
├── Cargo.toml         # PyO3 bindings
```

## Key Design Decisions

**Why PCA binary codes?**
ArcFace embeddings concentrate identity-discriminative information along principal axes. PCA aligns quantization boundaries with those axes. Random projections waste bits on low-variance directions.

**Why not HNSW?**
HNSW requires 1.5x the memory of raw vectors (graph edges). FaceFlash's binary index is 32x smaller (6 MB vs 195 MB at 100K). Float vectors are still needed for reranking but can be memory-mapped from disk — only K candidate rows (~100 KB) are paged in per query. On memory-constrained devices, resident RAM is dominated by the 6 MB binary index.

**Why Rust?**
The Hamming scan uses hardware POPCNT instructions. Rust compiles to tight SIMD loops — 50x faster than NumPy.

**Why cosine reranking?**
Binary codes approximate nearest neighbors. Reranking top-K with exact cosine recovers accuracy. At 100 candidates on 100K faces, reranking costs only 0.04ms.

## Limitations

- Recall at 100K: 99.7% at 100 candidates — hard cases need 200-300 candidates for 100%
- FAISS batch mode is faster per-query (0.017ms) due to BLAS parallelism
- PCA+ITQ must be fitted on representative data (≥1024 samples)
- Requires ArcFace model download (~166 MB) on first use
- DiskANN/ScaNN not benchmarked (Linux-only libraries)
- **Memory footprint:** Binary index is 6 MB at 100K (resident in RAM). Float vectors (195 MB) are memory-mapped from disk after `save()`/`load()` — only candidate rows (~100 KB) are paged per query. Resident RAM: ~6 MB committed, 195 MB cold on disk (evictable). Rerank latency is page-cache-dependent: ~0.47ms when floats are cached by the OS; on a memory-constrained device where they cannot stay cached, rerank becomes I/O-bound — measure on target hardware.
- **Build requires full RAM:** Building the index (`add_batch`) holds all float vectors in memory. The mmap benefit only applies to the query/serving process after `save()`/`load()`. Edge deployment model: build in cloud, ship `.npy` files, mmap-load on device.

## Installation

```bash
# Python package
pip install faceflash

# With Rust backend (recommended, 50x faster search)
cd rust && maturin develop --release

# With benchmark dependencies
pip install faceflash[benchmark]
```

## Reproducing Benchmarks

```bash
# Extract real embeddings
python scripts/extract_lfw_embeddings.py
python scripts/extract_vggface2_fast.py

# Run benchmarks
python benchmarks/bench_comprehensive.py        # All metrics, both datasets
python benchmarks/bench_tier1.py                # vs FAISS, HNSWLIB, USearch (100K)
python benchmarks/bench_tier2.py                # vs Annoy, NMSLIB (100K)
python benchmarks/bench_tier3_facelibs.py       # vs DeepFace, InsightFace
python benchmarks/bench_multimodel.py           # Multiple embedding models + HNSWLIB
python benchmarks/bench_lfw_rigorous.py         # LFW timing breakdown
python benchmarks/bench_scale.py                # Scalability to 1M
```

## Contributing

Key areas for contribution:
- [ ] YOLOv8-Face detector (faster than Haar)
- [ ] DiskANN / ScaNN comparison (Linux)
- [ ] IJB-C / MegaFace protocols
- [ ] Mobile deployment (ONNX + CoreML)
- [ ] Streaming insertion (add faces without refitting PCA)
- [ ] Multi-probe search for higher recall on hard cases
- [ ] Adaptive candidate selection (auto-tune rerank count)

## License

MIT
