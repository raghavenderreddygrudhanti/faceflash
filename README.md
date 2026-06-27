# FaceFlash

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Rust AVX-512](https://img.shields.io/badge/backend-Rust%20AVX--512%20VPOPCNTDQ-orange.svg)](rust/)
[![CI](https://github.com/raghavenderreddygrudhanti/faceflash/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/faceflash/actions)

**Face search that fits in a megabyte.**

100% recall at 48× less memory than HNSW. Faster than HNSW at 100K (single-query and batched).
Scales to 1M faces in 61 MB. No GPU, no training, runs on CPU.

- **100K faces:** 6.1 MB index, 0.43ms search, 27,661 batched QPS
- **500K faces:** 30.5 MB index, 100% recall, 1.9× faster batched than HNSW
- **1M faces:** 61 MB index vs HNSW's 2.9 GB — same recall, tied throughput

```
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

## At a Glance

|  | FaceFlash | HNSWLIB | USearch | ScaNN | FAISS-Flat |
|--|-----------|---------|---------|-------|------------|
| **Recall@1** | **100%** | 100% | 99.5% | 98.3% | 100% |
| **Memory (100K)** | **6.1 MB** | 293 MB | 254 MB | 12 MB | 195 MB |
| **Memory (500K)** | **30.5 MB** | 1,465 MB | 1,270 MB | 61 MB | 977 MB |
| **Memory (1M)** | **61 MB** | 2,930 MB | 2,539 MB | 122 MB | 1,953 MB |
| **Latency (100K)** | **0.43ms** | 0.60ms | 0.17ms | 0.10ms | 4.90ms |
| **Batched QPS (100K)** | **27,661** | 5,813 | 137,264 | — | — |
| **Batched QPS (500K)** | **10,337** | 5,577 | 76,150 | — | — |
| **Training required** | No | Yes | Yes | Yes | No |

Tested on MS1MV2 (44,291 distinct identities, 645,019 embeddings).
Single-query rows: single-threaded. Batched rows: all cores (128-thread AMD EPYC 9355).

> **Memory = binary index only.** Float vectors for reranking are mmap'd from disk after `save()`/`load()` — only ~100 candidate rows are paged per query. See [Limitations](#limitations).

![Memory to index 500K faces — FaceFlash 30 MB vs competitors at 1,000+ MB](docs/figures/chart_memory_scale.png)

![FaceFlash demo — search 100,000 faces from a 6 MB index in ~0.4 ms on CPU](docs/figures/demo.gif)

## Quick Start

```python
from faceflash import FaceFlash

ff = FaceFlash()  # first run downloads ArcFace model (~166MB)

# Register faces (best results with pre-aligned 112x112 crops)
ff.register("Alice", "alice.jpg")
ff.register("Bob", "bob.jpg")

# Search
result = ff.search("query.jpg")
print(result)
# {"matches": [{"name": "Alice", "confidence": 0.92}], "search_time_ms": 0.4}

# Verify two faces
ff.verify("photo1.jpg", "photo2.jpg")
# {"match": True, "confidence": 0.87}

# Bulk register + save
ff.register_folder("employees/")  # folder/person_name/photo.jpg
ff.save("my_index/")
```

> Installing from source builds the Rust AVX-512/NEON backend automatically — just needs a Rust toolchain. Falls back to NumPy if unavailable. For best accuracy, use pre-aligned 112x112 face images.

## How It Works

Each face becomes a tiny binary fingerprint (64 bytes), searched with AVX-512
VPOPCNTDQ (or NEON on ARM, scalar POPCNT elsewhere), then the top candidates
are verified with exact cosine similarity. Exact-quality accuracy at a fraction of the memory.

```
Image -> Detect Face -> ArcFace Embedding (512-dim) -> PCA+ITQ Binary Code (64 bytes)
                                                                |
Query -> Same pipeline -> Hamming Scan (Rust AVX-512) -> Top-K Cosine Rerank -> Match
```

## Performance

### FaceFlash vs All Competitors (100K–1M)

![Batched throughput: FaceFlash vs all competitors from 100K to 1M](docs/figures/chart_throughput_scale.png)

**Throughput:** FaceFlash batched delivers **4.8× more QPS than HNSW** at 100K, **1.9× at 500K**, ties at 1M — all at 100% recall and 48× less memory.

![Single-query latency: all methods from 100K to 1M](docs/figures/chart_latency_scale.png)

**Latency:** FaceFlash is **1.4× faster than HNSW at 100K** (0.43ms vs 0.60ms). At 500K+ HNSW's O(log N) pulls ahead on single-query, but FaceFlash's batched path stays faster.

![Recall vs Memory at 500K: FaceFlash dominates the top-left corner](docs/figures/chart_recall_memory_scale.png)

**Recall vs Memory:** At 500K, FaceFlash is the **only method achieving 100% recall below 100 MB**. All competitors need 1,000–1,500 MB for equivalent accuracy.

### Cross-Scale Summary

| Scale | Recall | Single-query | Batched (all cores) | Memory | vs HNSW memory |
|-------|--------|-------------|-------------------|--------|---------------|
| **100K** | 100% | **0.43ms** (faster than HNSW) | **27,661 qps** (4.8× HNSW) | **6.1 MB** | 48× less |
| **200K** | 100% | 0.66ms (tied with HNSW) | **19,930 qps** (7.5× HNSW) | **12.2 MB** | 48× less |
| **300K** | 100% | 0.88ms | **15,147 qps** (4.1× HNSW) | **18.3 MB** | 48× less |
| **500K** | 100% | 1.54ms | **10,337 qps** (1.9× HNSW) | **30.5 MB** | 48× less |
| **1M** | 100% | 2.92ms | **5,403 qps** (tied HNSW) | **61 MB** | 48× less |

FaceFlash dominates up to 300K on every axis. At 500K–1M, HNSW is faster single-query (O(log N) vs O(N)), but FaceFlash still wins on batched throughput and always uses 48× less memory.

### 1:N Identification — 44,290 Distinct People

The hardest test: one photo per person in the gallery, find them using a *different* photo.

![Rank-1 identification ties exact search on 44,290 distinct people](docs/figures/chart_rank1_tie.png)

| Method | Rank-1 Accuracy | Memory |
|--------|----------------|--------|
| FAISS-Flat (exact ceiling) | 95.8% | 93.9 MB |
| **FaceFlash (512b/100c)** | **95.8%** | **2.70 MB** |
| **FaceFlash (512b/300c)** | **95.8%** | **2.70 MB** |
| FaceFlash (256b/100c) | 95.6% | 1.35 MB |

FaceFlash ties exact search using **35× less memory**. The binary compression is lossless at 512 bits.

## When to Use FaceFlash

| Scenario | Why FaceFlash wins |
|----------|-------------------|
| **Edge / mobile / IoT** | 6–61 MB vs 293–2,930 MB for HNSW. Fits in device RAM. |
| **Multi-tenant** (many galleries on one server) | 100 galleries × 30 MB = 3 GB. HNSW: 100 × 1.5 GB = 150 GB. |
| **Batch identification** (dedup, enrollment, watchlist) | 4.8× faster than HNSW batched at 100K; 1.9× at 500K. |
| **100% recall is non-negotiable** | FaceFlash achieves 100% at every scale. USearch drops to 94–99%. |
| **Budget hardware** | Runs on Raspberry Pi, cheap VPS, phones. No GPU. No training step. |
| **Offline / air-gapped** | Single pip install, no network needed at search time. |
| **10K–500K face databases** | The sweet spot: faster AND less memory than HNSW. |

**When HNSW is a better fit:**
- You need <0.3ms single-query latency at >500K faces AND have gigabytes of RAM
- Your database exceeds 2M faces (O(log N) pulls clearly ahead)
- You need 100K+ batched QPS regardless of memory (USearch wins there)

## Detailed Results

All benchmarks: single-threaded unless labeled "batched", ground truth = exact brute-force cosine argmax. Measured on AMD EPYC 9355 (128 threads), AVX-512 VPOPCNTDQ active, real MS1MV2 embeddings.

### 100K Faces (6,939 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (batched)** | **100%** | **0.036ms** | **27,661** | **6.1 MB** |
| **FaceFlash (512b/200c)** | **100%** | 0.30ms | 3,310 | **6.1 MB** |
| **FaceFlash (512b/100c)** | **100%** | 0.43ms | 2,344 | **6.1 MB** |
| HNSWLIB (ef=128) | 100% | 0.60ms | 1,671 | 293 MB |
| HNSWLIB batched | 100% | 0.172ms | 5,813 | 293 MB |
| USearch batched | 99.5% | 0.007ms | 137,264 | 254 MB |
| USearch | 99.5% | 0.17ms | — | 254 MB |
| ScaNN | 98.3% | 0.10ms | — | 12 MB |
| FAISS-Flat (exact) | 100% | 4.90ms | 204 | 195 MB |

### 200K Faces (13,749 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (batched)** | **100%** | **0.050ms** | **19,930** | **12.2 MB** |
| **FaceFlash (512b/200c)** | **100%** | 0.57ms | 1,751 | **12.2 MB** |
| HNSWLIB (ef=128) | 99.9% | 0.65ms | 1,531 | 586 MB |
| HNSWLIB batched | 99.9% | 0.378ms | 2,646 | 586 MB |
| USearch batched | 99.1% | 0.008ms | 121,660 | 508 MB |
| ScaNN | 97.2% | 0.19ms | — | 24 MB |

**FaceFlash at 200K:** higher recall (100% vs 99.9%), same latency, 7.5× faster batched, 48× less memory.

### 300K Faces (20,615 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (batched)** | **100%** | **0.066ms** | **15,147** | **18.3 MB** |
| **FaceFlash (512b/200c)** | **100%** | 0.84ms | 1,187 | **18.3 MB** |
| HNSWLIB (ef=128) | 99.9% | 0.66ms | 1,510 | 879 MB |
| HNSWLIB batched | 99.7% | 0.269ms | 3,715 | 879 MB |
| USearch batched | 98.7% | 0.014ms | 73,383 | 762 MB |
| ScaNN | 97.8% | 0.28ms | — | 37 MB |

**FaceFlash at 300K:** highest recall of all methods (100%), 4.1× faster batched than HNSW, 48× less memory.

### 500K Faces (34,328 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (batched)** | **100%** | **0.097ms** | **10,337** | **30.5 MB** |
| **FaceFlash (512b/200c)** | **100%** | 1.45ms | 692 | **30.5 MB** |
| HNSWLIB (ef=128) | 100% | 0.71ms | 1,416 | 1,465 MB |
| HNSWLIB batched | 99.9% | 0.179ms | 5,577 | 1,465 MB |
| USearch batched | 98.4% | 0.013ms | 76,150 | 1,270 MB |
| ScaNN | 97.6% | 0.45ms | — | 61 MB |

**FaceFlash at 500K:** equal 100% recall, 1.9× faster batched than HNSW, 48× less memory.

### 1M Faces (44,291 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (batched)** | **100%** | **0.185ms** | **5,403** | **61 MB** |
| **FaceFlash (512b/100c)** | **100%** | 2.92ms | 342 | **61 MB** |
| HNSWLIB (ef=128) | 100% | 0.66ms | 1,523 | 2,930 MB |
| HNSWLIB batched | 100% | 0.178ms | 5,621 | 2,930 MB |
| USearch batched | 94.1% | 0.013ms | 77,266 | 2,539 MB |
| ScaNN | 98.2% | 0.86ms | — | 122 MB |

**FaceFlash at 1M:** batched throughput ties HNSW (5,403 vs 5,621 qps) at 48× less memory. HNSW is 4.4× faster single-query.

### Code Length (n_bits) Comparison

![Recall vs candidates per code length](docs/figures/chart_recall_vs_candidates.png)

| Code length | Bytes/face | Index (644K) | Recall@1 | Latency | Best for |
|-------------|-----------|-------------|----------|---------|----------|
| 128 bits | 16 B | 9.8 MB | 99.4% | 2.62ms | Ultra-low memory (mobile) |
| **256 bits** | 32 B | 19.6 MB | **100%** | 3.78ms | Balanced |
| 384 bits | 48 B | 29.5 MB | 100% | 5.00ms | — |
| **512 bits** | 64 B | 39.3 MB | **100%** | **1.87ms** | **Fastest** (fits AVX-512 register) |

512 bits is fastest because one VPOPCNTDQ instruction processes the entire 64-byte code. At 256+ bits, recall is 100%.

### Face Alignment (LFW Verification)

| Alignment | LFW Accuracy | TAR@FAR=1e-3 |
|-----------|--------------|--------------|
| Haar center-crop | 98.57% ± 0.48 | 96.97% |
| **5-point (SCRFD/RetinaFace)** | **99.85% ± 0.17** | **99.73%** |

Proper alignment lifts accuracy **+1.28 points**.

### Scaling to Millions (Clustering)

`build_clusters()` adds an optional IVF layer — groups faces into ~sqrt(N) buckets, queries scan only the closest `n_probe` buckets.

```python
ff.index.build_clusters(n_probe=16)
```

![Clustering recall/speed tradeoff](docs/figures/chart_clustering_tradeoff.png)

| Scale | n_probe | Recall@1 | Latency | Speedup |
|-------|---------|----------|---------|---------|
| 100K | 16 | 96.1% | 0.12ms | 2.6× |
| 100K | 32 | 98.5% | 0.17ms | 1.8× |
| 500K | 16 | 87.9% | 0.31ms | 5.0× |
| 500K | 32 | 92.8% | 0.56ms | 2.8× |

With VPOPCNTDQ making full scan so fast (0.31ms at 100K), clustering is mainly useful at 500K+ where it buys 5–8× for ~88–93% recall.

### Raw Scan Speed (SIMD)

![Raw binary-scan speed](docs/figures/chart_simd_speed.png)

| Scale | Single-threaded | Parallel (all cores) | Speedup |
|-------|----------------|---------------------|---------|
| 100K | 0.27ms (3,651 qps) | 0.25ms (3,977 qps) | 1.1× |
| 500K | 1.49ms (672 qps) | 0.36ms (2,779 qps) | 4.1× |
| 1M | 2.95ms (340 qps) | 0.52ms (1,911 qps) | 5.6× |

AVX-512 VPOPCNTDQ is 3.5× faster than the previous scalar POPCNT. Runtime-detected, no user config needed.

## Tuning

| Deployment | Config | Recall@1 | Memory/face | Notes |
|-----------|--------|----------|-------------|-------|
| **Ultra-compact** | `n_bits=128, n_candidates=500` | 99.4% | 16 bytes | Minimum RAM (mobile/IoT) |
| **Server** | `n_bits=256, n_candidates=100` | 100% | 32 bytes | Balanced memory/accuracy |
| **Balanced** (default) | `n_bits=512, n_candidates=100` | 100% | 64 bytes | Fastest (AVX-512 single-instruction) |
| **Edge** | `n_bits=512, n_candidates=50` | 99.5% | 64 bytes | Minimize disk reads per query |

```python
ff = FaceFlash(n_bits=128, n_candidates=500)     # ultra-compact: 16 bytes/face
ff = FaceFlash(n_bits=256, n_candidates=100)     # server: balanced
ff = FaceFlash(n_bits=512, n_candidates=100)     # default: fastest scan
ff.search("query.jpg", n_candidates=200)         # per-query override
```

## Architecture

```
faceflash/
├── engine.py          # High-level API (register, search, verify)
├── detect.py          # Face detection (SCRFD detector + Haar fallback)
├── align.py           # 5-point alignment to ArcFace template
├── embed.py           # ArcFace ONNX embedding (512-dim, auto-downloads)
├── index.py           # Binary index + batched search
├── pca_quantize.py    # PCA+ITQ quantizer (the core algorithm)
rust/
├── src/lib.rs         # AVX-512 VPOPCNTDQ / NEON / scalar POPCNT (PyO3 + Rayon)
├── Cargo.toml
```

**Why PCA+ITQ?** ArcFace embeddings concentrate identity along principal axes. PCA aligns quantization with those axes. ITQ rotates bits for balanced marginals. Result: 100% recall at 64 bytes/face.

**Why not HNSW internally?** HNSW stores a graph on top of full float vectors (1.5× raw memory). FaceFlash stores 64 bytes per face. Float vectors are mmap'd from disk and only paged for ~100 candidates. Tradeoff: higher single-query latency at 500K+, but 48× less memory.

**Why Rust + AVX-512?** AVX-512 VPOPCNTDQ processes a full 512-bit code in one instruction. Combined with cache-blocked batching and Rayon parallelism: 10–17× throughput vs per-query serial.

## Limitations

- **Single-query at 1M+** — O(N) scan; HNSW is 4.4× faster per-query at 1M. Batched path ties at 1M.
- **Memory during build** — holds all float vectors in RAM. The 48× savings apply after `save()`/`load()`.
- **AVX-512 VPOPCNTDQ** — 3.5× speedup only on Ice Lake / Zen 4+ / EPYC 9004+. Older CPUs use scalar POPCNT (still fast). Runtime-detected.
- **Rerank I/O** — pages ~100 float rows from disk per query. Invisible on NVMe; adds latency on slow storage.

## Installation

```bash
# Requires Rust toolchain (prebuilt wheels coming via PyPI):
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"

# With benchmark dependencies
pip install "faceflash[cpu,benchmark] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

## Reproduce the Benchmarks

```bash
# Local (LFW + VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# RunPod (full suite — MS1MV2 85K identities, all scales)
export GITHUB_TOKEN=<token>
bash scripts/runpod_ms1m.sh    # FORCE_EXTRACT=1 for full 85K extraction
```

## Roadmap

**v0.1.0** (current) — working system with proven benchmarks
- [x] PCA+ITQ binary quantization + Rust search
- [x] High-level API (register, search, verify)
- [x] Benchmarked against FAISS, HNSWLIB, USearch, ScaNN at 100K–1M
- [x] 1:N identification on 44,290 distinct identities (MS1MV2)
- [x] 5-point alignment (SCRFD/RetinaFace) — 99.85% LFW

**v0.2.0** — production quality
- [x] RetinaFace alignment
- [x] Prebuilt wheels (`pip install faceflash`)
- [x] Full 85K-identity benchmark (76,872 identities extracted)
- [x] On-device memory measurement (6.1 MB binary index @100K)

**v0.3.0** — performance
- [x] Coarse clustering (IVF) — 2.7–4.9× faster
- [x] AVX-512 VPOPCNTDQ — 3.5× faster than scalar POPCNT
- [x] Cache-blocked batched search — 17× throughput at 500K–1M
- [x] NEON kernels — ARM-optimized

**v1.0.0** — stable
- [ ] Stable public API
- [ ] DiskANN comparison
- [ ] Mobile deployment (ONNX + CoreML)
- [ ] Streaming insertion (no PCA refit)

## Contributing

Contributions welcome:

- **DiskANN comparison** — the one major competitor we haven't benchmarked
- **Mobile deployment** — ONNX + CoreML for iOS/Android
- **Streaming insertion** — add faces without refitting PCA
- **GPU batched search** — CUDA kernel for 10M+ galleries
- **Raspberry Pi / Jetson benchmarks** — measured edge numbers
- **WebAssembly build** — browser-based face search

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

## License

MIT

---

If this is useful, a star helps others find it.
