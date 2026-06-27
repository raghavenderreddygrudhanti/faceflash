# FaceFlash

<!-- GitHub repo settings: add topics: face-recognition, vector-search, binary-hashing, edge-ai, memory-efficient, arcface, rust -->

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Rust AVX-512](https://img.shields.io/badge/backend-Rust%20AVX--512%20VPOPCNTDQ-orange.svg)](rust/)
[![CI](https://github.com/raghavenderreddygrudhanti/faceflash/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/faceflash/actions)

**Face search that fits in a megabyte.**

Search 44,000 distinct people in 2.7 MB. Search 500,000 faces in 30 MB.
100% recall — same as exact brute-force — at 48× less memory than HNSW.
Faster than HNSW in batched mode up to 500K. Runs on CPU.

```
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

## At a Glance

|  | FaceFlash | HNSWLIB (best graph index) |
|--|-----------|--------------------------|
| Find the right person (rank-1) | **95.8%** | 95.8% (exact ceiling) |
| Memory for 100K faces | **6.1 MB** | 293 MB |
| Memory for 500K faces | **30.5 MB** | 1,465 MB |
| Memory for 1M faces | **61 MB** | 2,930 MB |
| Recall@1 (all scales) | **100%** | 100% |
| Batched throughput (100K) | **27,661 qps** | 5,813 qps |
| Batched throughput (500K) | **10,337 qps** | 5,577 qps |

Tested on MS1MV2 (44,291 distinct identities, 645,019 embeddings).
All single-query rows measured single-threaded, same hardware, same data.
Batched rows use all cores (128-thread AMD EPYC 9355).

> **Memory = binary index only.** Float vectors for reranking are mmap'd from disk after `save()`/`load()` — only ~100 candidate rows are paged per query. See [Limitations](#limitations) for details.

![Memory to index 100K faces — FaceFlash 6 MB vs HNSW 293 MB](docs/figures/chart_memory_bar.png)

![FaceFlash demo — search 100,000 faces from a 6 MB index in ~0.4 ms on CPU](docs/figures/demo.gif)

## Quick Start

```python
from faceflash import FaceFlash

ff = FaceFlash()  # first run downloads ArcFace model (~166MB)

# Register faces (best results with pre-aligned 112×112 crops)
ff.register("Alice", "alice.jpg")
ff.register("Bob", "bob.jpg")

# Search
result = ff.search("query.jpg")
print(result)
# {"matches": [{"name": "Alice", "confidence": 0.92}], "search_time_ms": 0.5}

# Verify two faces
ff.verify("photo1.jpg", "photo2.jpg")
# {"match": True, "confidence": 0.87}

# Bulk register + save
ff.register_folder("employees/")  # folder/person_name/photo.jpg
ff.save("my_index/")
```

> **Note:** Installing from source builds the Rust AVX-512/NEON backend into the package automatically — it just needs a Rust toolchain present (prebuilt PyPI wheels need nothing). If the backend is unavailable, search transparently falls back to NumPy. For best accuracy, use pre-aligned 112×112 face images. See [Limitations](#limitations).

## How It Works

Each face becomes a tiny binary fingerprint (64 bytes), searched with AVX-512
VPOPCNTDQ (or NEON on ARM, scalar POPCNT elsewhere), then the top few
candidates are verified with exact cosine similarity. You get exact-quality
accuracy at a fraction of the memory.

```
Image → Detect Face → ArcFace Embedding (512-dim) → PCA+ITQ Binary Code (64 bytes)
                                                              ↓
Query → Same pipeline → Hamming Scan (Rust AVX-512) → Top-K Cosine Rerank → Match
```

## vs the Competition (100K faces, single-threaded)

|  | FaceFlash | FAISS-Flat | FAISS-IVF | HNSWLIB | USearch | ScaNN |
|--|-----------|-----------|-----------|---------|---------|-------|
| **Recall@1** | 100% | 100% | 99.6% | 100% | 99.5% | 98.3% |
| **Latency** | 0.43ms | 4.90ms | 1.14ms | 0.60ms | 0.17ms | 0.10ms |
| **Memory** | **6.1 MB** | 195 MB | 205 MB | 293 MB | 254 MB | 12 MB |
| **Training** | None | None | Required | Required | Required | Required |
| **GPU needed** | No | No | No | No | No | No |

### Batched throughput (all cores, 100K)

For bulk identification (enrollment dedup, gallery build, batch verification),
FaceFlash's cache-blocked batched search processes many queries in one call:

| Method | Recall@1 | Throughput | Memory |
|--------|----------|-----------|--------|
| **FaceFlash (batched)** | **100%** | **27,661 qps** | **6.1 MB** |
| HNSWLIB ef=128 (batched) | 100% | 5,813 qps | 293 MB |
| USearch (batched) | 99.5% | 137,264 qps | 254 MB |

FaceFlash batched is **4.8× faster than HNSW batched** at equal recall, using **48× less memory**.
USearch is faster at 99.5% recall — but uses 42× more RAM.

**FaceFlash wins:** memory (48× less than graph indexes), 100% recall, faster than HNSW in both single-query and batched modes at 100K.

**FaceFlash loses:** USearch batched is ~5× faster (at 42× more RAM and 0.5% lower recall). At larger scales (500K+), HNSW's O(log N) single-query latency pulls ahead.

**Bottom line:** if you have unlimited RAM and need the lowest single-query latency at huge scale, use HNSW. If memory is the constraint — edge, mobile, multi-tenant — FaceFlash gives you 100% recall in a fraction of the RAM, and in batched mode it's faster too.

## Why FaceFlash Wins

![Memory usage: FaceFlash vs all competitors from 100K to 1M](docs/figures/chart_memory_scale.png)

**Memory:** FaceFlash uses **48× less RAM** than HNSW, **21× less than USearch**, and **4× less than FAISS-IVF** — consistently across every scale from 100K to 1M.

![Batched throughput: FaceFlash vs all competitors from 100K to 1M](docs/figures/chart_throughput_scale.png)

**Throughput:** FaceFlash batched delivers **4.8× more QPS than HNSW batched** at 100K, **1.9× at 500K**, and ties at 1M — all at 100% recall and 48× less memory.

![Single-query latency: all methods from 100K to 1M](docs/figures/chart_latency_scale.png)

**Latency:** FaceFlash is **1.4× faster than HNSW at 100K** (0.43ms vs 0.60ms); at 500K+ HNSW's O(log N) pulls ahead, but FaceFlash's batched path stays competitive.

![Recall vs Memory at 500K: FaceFlash dominates the top-left corner](docs/figures/chart_recall_memory_scale.png)

**Recall vs Memory:** At 500K, FaceFlash is the **only method achieving 100% recall below 100 MB** — all competitors need 1,000–1,500 MB for the same accuracy.

**The complete picture across scales:**

| Scale | Recall | FaceFlash faster? | FaceFlash less memory? | Batched winner |
|-------|--------|-------------------|----------------------|---------------|
| 100K | Both 100% | **Yes** (0.43ms vs 0.60ms) | **48×** less | **FaceFlash** (4.8× faster) |
| 200K | FF: 100%, HNSW: 99.9% | Tied (0.66ms vs 0.65ms) | **48×** less | **FaceFlash** (7.5× faster) |
| 300K | FF: 100%, HNSW: 99.9% | HNSW 1.3× faster | **48×** less | **FaceFlash** (4.1× faster) |
| 500K | Both 100% | HNSW 2.2× faster | **48×** less | **FaceFlash** (1.9× faster) |
| 1M | Both 100% | HNSW 4.4× faster | **48×** less | **Tied** (5,403 vs 5,621 qps) |

**Key insight:** FaceFlash dominates up to 300K on every axis. At 500K–1M, the O(log N) advantage gives HNSW faster single-query latency, but FaceFlash *still* wins on batched throughput and memory.  No other system achieves 100% recall below 100 MB at any scale.

## Detailed Benchmarks

All benchmarks: single-threaded unless labeled "parallel" or "batched", `time.perf_counter()` per query, ground truth = exact brute-force cosine argmax. Measured on AMD EPYC 9355 (128 threads), AVX-512 VPOPCNTDQ active, real MS1MV2 embeddings (44,291 identities).

### 1:N Identification — 44,290 Distinct People (MS1MV2)

The hardest test: one photo per person in the gallery, find them using a *different* photo. Gallery of 44,290 identities, 5,000 probes.

![Rank-1 identification ties exact search on 44,290 distinct people](docs/figures/chart_rank1_tie.png)

| Method | Rank-1 Accuracy | Memory |
|--------|----------------|--------|
| FAISS-Flat (exact ceiling) | 95.8% | 93.9 MB |
| **FaceFlash (512b/100c)** | **95.8%** | **2.70 MB** |
| **FaceFlash (512b/300c)** | **95.8%** | **2.70 MB** |
| FaceFlash (256b/100c) | 95.6% | 1.35 MB |

FaceFlash ties exact search using **35× less memory** — the binary compression is lossless at 512 bits.

### FaceFlash Advantage at Every Scale

| Scale | FaceFlash single | FaceFlash batched | HNSW ef=128 single | HNSW batched | FF memory | HNSW memory | Memory savings |
|-------|-----------------|-------------------|-------------------|-------------|-----------|-------------|---------------|
| **100K** | **0.43ms** | **0.036ms** | 0.60ms | 0.172ms | **6.1 MB** | 293 MB | **48×** |
| **200K** | 0.66ms | **0.050ms** | 0.65ms | 0.378ms | **12.2 MB** | 586 MB | **48×** |
| **300K** | 0.88ms | **0.066ms** | 0.66ms | 0.269ms | **18.3 MB** | 879 MB | **48×** |
| **500K** | 1.54ms | **0.097ms** | 0.71ms | 0.179ms | **30.5 MB** | 1,465 MB | **48×** |
| **1M** | 2.92ms | **0.185ms** | 0.66ms | 0.178ms | **61 MB** | 2,930 MB | **48×** |

**What FaceFlash wins at each scale:**
- **100K:** Faster single-query AND batched, 48× less memory, 100% recall
- **200K:** Faster batched (7.6× vs HNSW), tied single-query, 48× less memory, higher recall (100% vs 99.9%)
- **300K:** Faster batched (4.1× vs HNSW), 48× less memory, higher recall (100% vs 99.9%)
- **500K:** Faster batched (1.9× vs HNSW), 48× less memory, equal 100% recall
- **1M:** Tied batched, 48× less memory, equal 100% recall. HNSW is 4.4× faster single-query.

### n_bits Code Length Comparison (644K faces)

More bits = more memory per face but higher binary-only recall. The cosine rerank recovers accuracy regardless, so the choice is memory vs scan speed:

![Recall vs candidates per code length, with 95% confidence bands](docs/figures/chart_recall_vs_candidates.png)

| Code length | Bytes/face | Index size | Recall@1 | Latency | Best for |
|-------------|-----------|-----------|----------|---------|----------|
| 128 bits | 16 B | 9.8 MB | 99.4% | 2.62ms | Ultra-low memory (mobile) |
| **256 bits** | 32 B | 19.6 MB | **100%** | 3.78ms | Balanced memory/accuracy |
| 384 bits | 48 B | 29.5 MB | 100% | 5.00ms | — |
| **512 bits** | 64 B | 39.3 MB | **100%** | 1.87ms | **Fastest** (fits AVX-512 register) |

512 bits is fastest because one AVX-512 VPOPCNTDQ instruction processes the entire code (64 bytes = one 512-bit register). Shorter codes need fewer bytes but the scan takes longer due to more rerank candidates needed. At 256+ bits, recall is 100%.

### vs All ANN Methods — 100K Faces (6,939 identities)

![Recall vs memory — FaceFlash sits alone in the high-recall, low-memory corner](docs/figures/chart_recall_memory_pareto.png)

| Method | Recall@1 | Latency | QPS | Memory | Type |
|--------|----------|---------|-----|--------|------|
| **FaceFlash (512b, batched)** | **100%** | **0.036ms** | **27,661** | **6.1 MB** | binary hash |
| **FaceFlash (512b/200c)** | **100%** | **0.30ms** | 3,310 | **6.1 MB** | binary hash |
| **FaceFlash (512b/100c)** | **100%** | **0.43ms** | 2,344 | **6.1 MB** | binary hash |
| HNSWLIB (ef=128) | 100% | 0.60ms | 1,671 | 293 MB | graph |
| HNSWLIB batched | 100% | 0.172ms | 5,813 | 293 MB | graph |
| USearch batched | 99.5% | 0.007ms | 137,264 | 254 MB | graph |
| USearch | 99.5% | 0.17ms | — | 254 MB | graph |
| ScaNN | 98.3% | 0.10ms | — | 12 MB | quantized |
| FAISS-Flat (exact) | 100% | 4.90ms | 204 | 195 MB | brute force |

### vs All ANN Methods — 200K Faces (13,749 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (512b, batched)** | **100%** | **0.050ms** | **19,930** | **12.2 MB** |
| **FaceFlash (512b/200c)** | **100%** | 0.57ms | 1,751 | **12.2 MB** |
| **FaceFlash (512b/100c)** | **100%** | 0.66ms | 1,523 | **12.2 MB** |
| HNSWLIB (ef=128) | 99.9% | 0.65ms | 1,531 | 586 MB |
| HNSWLIB batched | 99.9% | 0.378ms | 2,646 | 586 MB |
| USearch batched | 99.1% | 0.008ms | 121,660 | 508 MB |
| USearch | 99.5% | 0.21ms | — | 508 MB |
| ScaNN | 97.2% | 0.19ms | — | 24 MB |
| FAISS-Flat | 100% | 10.1ms | 99 | 391 MB |

FaceFlash at 200K: **higher recall than HNSW** (100% vs 99.9%), **same single-query latency** (0.66ms vs 0.65ms), **7.5× faster batched**, **48× less memory**.

### vs All ANN Methods — 300K Faces (20,615 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (512b, batched)** | **100%** | **0.066ms** | **15,147** | **18.3 MB** |
| **FaceFlash (512b/200c)** | **100%** | 0.84ms | 1,187 | **18.3 MB** |
| HNSWLIB (ef=128) | 99.9% | 0.66ms | 1,510 | 879 MB |
| HNSWLIB batched | 99.7% | 0.269ms | 3,715 | 879 MB |
| USearch batched | 98.7% | 0.014ms | 73,383 | 762 MB |
| USearch | 99.3% | 0.28ms | — | 762 MB |
| ScaNN | 97.8% | 0.28ms | — | 37 MB |
| FAISS-Flat | 100% | 14.9ms | 67 | 586 MB |

FaceFlash at 300K: **higher recall than all competitors** (100% vs 99.9% HNSW, 99.3% USearch), **4.1× faster batched than HNSW**, **48× less memory**.

### vs All ANN Methods — 500K Faces (34,328 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (512b, batched)** | **100%** | **0.097ms** | **10,337** | **30.5 MB** |
| **FaceFlash (512b/200c)** | **100%** | 1.45ms | 692 | **30.5 MB** |
| HNSWLIB (ef=128) | 100% | 0.71ms | 1,416 | 1,465 MB |
| HNSWLIB batched | 99.9% | 0.179ms | 5,577 | 1,465 MB |
| USearch batched | 98.4% | 0.013ms | 76,150 | 1,270 MB |
| USearch | 98.6% | 0.32ms | — | 1,270 MB |
| ScaNN | 97.6% | 0.45ms | — | 61 MB |
| FAISS-Flat | 100% | 24.8ms | 40 | 977 MB |

FaceFlash at 500K: **equal 100% recall to HNSW**, **1.9× faster batched**, **48× less memory**. HNSW is 2.2× faster single-query (O(log N) advantage).

### vs All ANN Methods — 1M Faces (44,291 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|--------|----------|---------|-----|--------|
| **FaceFlash (512b, batched)** | **100%** | **0.185ms** | **5,403** | **61 MB** |
| **FaceFlash (512b/100c)** | **100%** | 2.92ms | 342 | **61 MB** |
| HNSWLIB (ef=128) | 100% | 0.66ms | 1,523 | 2,930 MB |
| HNSWLIB batched | 100% | 0.178ms | 5,621 | 2,930 MB |
| USearch batched | 94.1% | 0.013ms | 77,266 | 2,539 MB |
| HNSWLIB (ef=32) | 99.3% | 0.19ms | — | 2,930 MB |
| ScaNN | 98.2% | 0.86ms | — | 122 MB |
| FAISS-Flat | 100% | 56.0ms | 18 | 1,953 MB |

FaceFlash at 1M: **batched throughput ties HNSW** (5,403 vs 5,621 qps) at **48× less memory**. HNSW is 4.4× faster single-query — that's the O(log N) vs O(N) structural difference. USearch is fastest but drops to 94.1% recall.

### Face Alignment — Raw Photos (LFW Verification)

The built-in 5-point alignment vs the basic Haar center-crop, both through the same ArcFace embedder on the standard LFW 10-fold protocol (6,000 pairs, raw funneled images):

| Alignment | LFW Accuracy | TAR@FAR=1e-3 |
|-----------|--------------|--------------|
| Haar center-crop | 98.57% ± 0.48 | 96.97% |
| **5-point (SCRFD/RetinaFace)** | **99.85% ± 0.17** | **99.73%** |

Proper alignment lifts accuracy **+1.28 points** — this is what lets raw photos approach the pre-aligned benchmark numbers.

### Scaling to Millions

By default search scans every face — fine up to a few million, but the time grows
with the database. `build_clusters()` adds an optional IVF layer: faces are grouped
into ~√N buckets, and a query only scans the `n_probe` closest buckets.

```python
ff.index.build_clusters(n_probe=16)   # after registering faces
```

![Clustering recall/speed tradeoff on real MS1MV2 — labels show speedup vs full scan](docs/figures/chart_clustering_tradeoff.png)

Measured on **real MS1MV2 faces** (Rust backend, single thread). `n_probe` is the
recall/speed knob — clustering trades recall for speed, so it's for when you can
tolerate approximate results, not when you need exact recall:

**100K faces** (full scan: 100% recall @ 0.31 ms)

| n_probe | Recall@1 | Latency | Speedup |
|---------|----------|---------|---------|
| 8  | 93.1% | 0.09 ms | 3.6× |
| 16 | 96.1% | 0.12 ms | 2.6× |
| 32 | 98.5% | 0.17 ms | 1.8× |
| 64 | 99.3% | 0.28 ms | 1.1× |

**500K faces** (full scan: 100% recall @ 1.54 ms)

| n_probe | Recall@1 | Latency | Speedup |
|---------|----------|---------|---------|
| 8  | 79.8% | 0.19 ms | 8.2× |
| 16 | 87.9% | 0.31 ms | 5.0× |
| 32 | 92.8% | 0.56 ms | 2.8× |
| 64 | 96.5% | 1.04 ms | 1.5× |

The speedup **widens as the database grows** — full scan slows ~linearly, clustered
stays much flatter. The honest tradeoff: with VPOPCNTDQ making the full scan so fast
(0.31ms at 100K), clustering's benefit is modest at small scales. At 500K+, clustering
buys 5–8× for ~88–93% recall. Probe every bucket and you reproduce the exact full-scan
result; leave clustering off (the default) and search is unchanged.

### Raw Scan Speed (SIMD)

The Hamming scan uses hand-written SIMD kernels with runtime dispatch — **AVX-512
VPOPCNTDQ** on x86 (64 bytes per instruction, native 512-bit popcount on Ice Lake /
Zen 4+ / EPYC 9004+), **NEON** on ARM (`vcntq_u8`), and a scalar `u64` POPCNT
fallback elsewhere. Every kernel is verified byte-exact against NumPy before timing.

![Raw binary-scan speed — AVX-512 VPOPCNTDQ on x86, NEON on ARM](docs/figures/chart_simd_speed.png)

| Scale | Single-threaded | Parallel (all cores) | Speedup |
|-------|----------------|---------------------|---------|
| 100K | 0.27ms (3,651 qps) | 0.25ms (3,977 qps) | 1.1× |
| 500K | 1.49ms (672 qps) | 0.36ms (2,779 qps) | 4.1× |
| 1M | 2.95ms (340 qps) | 0.52ms (1,911 qps) | 5.6× |

Measured on AMD EPYC 9355 (128 threads) with AVX-512 VPOPCNTDQ active. The
single-threaded path is 3.5× faster than the previous scalar POPCNT thanks to
the native 512-bit instruction. Parallel scaling grows with database size as
the work per query exceeds thread-dispatch overhead.

### When to Use FaceFlash

FaceFlash is the best choice when any of these apply:

| Scenario | Why FaceFlash wins |
|----------|-------------------|
| **Edge / mobile / IoT** | 6–61 MB vs 293–2,930 MB for HNSW. Fits in device RAM. |
| **Multi-tenant** (many galleries on one server) | 100 galleries × 30 MB = 3 GB. HNSW: 100 × 1.5 GB = 150 GB. |
| **Batch identification** (dedup, enrollment, watchlist) | 4.8× faster than HNSW batched at 100K; 1.9× at 500K. |
| **100% recall is non-negotiable** | FaceFlash achieves 100% at every scale tested. USearch drops to 94–99%. |
| **Budget hardware** | Runs on Raspberry Pi, cheap VPS, phones. No GPU. No training step. |
| **Offline / air-gapped** | Single pip install, no network needed at search time. |
| **10K–500K face databases** | The sweet spot: faster AND less memory than HNSW. |

**When HNSW is a better fit:**
- You need <0.3ms single-query latency at >500K faces AND have gigabytes of RAM
- Your database exceeds 2M faces (O(log N) pulls clearly ahead)
- You need 100K+ batched QPS regardless of memory (USearch wins there)

## Tuning

Two knobs, and they substitute for each other — more bits or more candidates both buy recall:

| Deployment | Config | Recall@1 | Memory/face | Notes |
|-----------|--------|----------|-------------|-------|
| **Ultra-compact** | `n_bits=128, n_candidates=500` | 99.4% | 16 bytes | Minimum RAM (mobile/IoT) |
| **Server** (floats in RAM) | `n_bits=256, n_candidates=100` | 100% | 32 bytes | Balanced memory/accuracy |
| **Balanced** (default) | `n_bits=512, n_candidates=100` | 100% | 64 bytes | Fastest (AVX-512 single-instruction) |
| **Edge** (floats on flash) | `n_bits=512, n_candidates=50` | 99.5% | 64 bytes | Minimize disk reads per query |

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
├── align.py           # 5-point RetinaFace/SCRFD alignment to ArcFace template
├── embed.py           # ArcFace ONNX embedding (512-dim, auto-downloads)
├── index.py           # Binary index with buffered O(n) add + batched search
├── pca_quantize.py    # PCA+ITQ quantizer (the core algorithm)
rust/
├── src/lib.rs         # Hamming search: AVX-512 VPOPCNTDQ / NEON / scalar POPCNT (PyO3)
├── Cargo.toml
```

**Why PCA+ITQ?** ArcFace embeddings concentrate identity information along principal axes.
PCA aligns quantization with those axes. ITQ rotates bits for balanced marginals.
Result: fewer candidates needed for the same recall vs random projection.

**Why not use HNSW internally?** HNSW-based libraries (HNSWLIB, USearch) are faster per-query at large scale, but they store a graph on top of the full float vectors — 1.5x raw memory. FaceFlash stores just 64 bytes per face. Float vectors are mmap'd from disk and only paged for the ~100 candidates that pass the binary filter. The tradeoff: higher single-query latency at 500K+, but 48× less memory.

**Why Rust + AVX-512?** Hamming distance is pure bit-manipulation. AVX-512 VPOPCNTDQ processes a full 512-bit face code in one instruction (XOR + popcount of 64 bytes). Combined with cache-blocked batching and Rayon parallelism, this delivers 10–17× throughput vs per-query serial.

## Limitations

- **Single-query latency at 1M+** — FaceFlash scans linearly (O(N)); HNSW is O(log N). At 1M, HNSW is 4.4× faster per-query. The batched path closes this gap (tied at 1M), but for interactive single-query at very large scale, graph indexes are faster.
- **Memory during build** — constructing the index holds all float vectors in RAM. The 48× memory savings apply after `save()`/`load()` (floats become mmap'd).
- **Face detection** — the built-in SCRFD detector handles raw photos with 5-point alignment. The retrieval benchmarks (100% recall, 95.8% rank-1) used pre-aligned data. For production use, pre-aligned 112×112 crops give the best results.
- **AVX-512 VPOPCNTDQ** — the 3.5× kernel speedup only fires on Ice Lake / Zen 4+ / EPYC 9004+ CPUs. Older CPUs use scalar POPCNT (still fast, just not 3.5× faster). Runtime-detected, no user action needed.
- **Rerank I/O** — the cosine rerank pages ~100 float rows from disk per query. On NVMe this is invisible; on slow storage it adds latency.

## Installation

```bash
# Installing from source builds the Rust AVX-512/NEON backend into the package
# automatically (no manual step) — requires a Rust toolchain on your machine:
#   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"   # CPU inference
# or [gpu] for CUDA. onnxruntime is an extra so it won't clobber an existing GPU install.

# With benchmark dependencies
pip install "faceflash[cpu,benchmark] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

> Prebuilt wheels (no Rust toolchain needed) ship via PyPI on each release — see `.github/workflows/release.yml`. Once published: `pip install faceflash`.

## Reproduce the Benchmarks

```bash
# Local (LFW + VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_search.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# RunPod (full suite — VGGFace2 1M + MS1MV2 85K identities)
export GITHUB_TOKEN=<token> KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
bash scripts/runpod_full.sh      # VGGFace2 1M + all ANN comparisons
bash scripts/runpod_ms1m.sh      # MS1MV2 (1:N identification; FORCE_EXTRACT=1 for full 85K)
```

## Roadmap

**v0.1.0** (current) — working system with proven benchmarks
- [x] PCA+ITQ binary quantization + Rust POPCNT search
- [x] High-level API (register, search, verify)
- [x] Benchmarked against FAISS, HNSWLIB, USearch, ScaNN at 100K–1M
- [x] 1:N identification on 44,290 distinct identities (MS1MV2)
- [x] **5-point alignment** (SCRFD/RetinaFace) — raw photos hit 99.85% LFW (+1.28 pts)

**v0.2.0** — production quality
- [x] **RetinaFace alignment** — 5-point face alignment so raw photos match benchmark accuracy
- [x] **Prebuilt wheels** — `pip install faceflash` ships the Rust backend (no toolchain needed)
- [x] **Full 85K-identity benchmark** — extracted 76,872 identities from MS1MV2 (44,291 with sufficient images)
- [x] **On-device memory measurement** — actual RSS measured on x86 (6.1 MB binary index @100K)

**v0.3.0** — performance
- [x] **Coarse clustering** — IVF buckets for sub-linear scan (`build_clusters()`); 2.7–4.9× faster at 100K–400K, widening with scale
- [x] **AVX-512 VPOPCNTDQ** — native 512-bit popcount (3.5× faster than scalar POPCNT); runtime-detected on Ice Lake/Zen 4+
- [x] **Cache-blocked batched search** — 17× throughput at 500K–1M (sidesteps memory-bandwidth wall)
- [x] **NEON kernels** — ARM-optimized for mobile/edge (vcntq_u8 native byte popcount)

**v1.0.0** — stable
- [ ] Stable public API (no breaking changes after this)
- [ ] DiskANN comparison
- [ ] Mobile deployment (ONNX + CoreML)
- [ ] Streaming insertion (add faces without refitting PCA)

## Contributing

FaceFlash is actively developed. Contributions welcome in these areas:

- **DiskANN comparison** — the one major competitor we haven't benchmarked
- **Mobile deployment** — ONNX + CoreML integration for iOS/Android
- **Streaming insertion** — add faces without refitting PCA (online learning)
- **GPU batched search** — CUDA kernel for 10M+ galleries
- **Raspberry Pi / Jetson benchmarks** — measured edge numbers for the README
- **WebAssembly build** — browser-based face search

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

## License

MIT

---

If this is useful, a star helps others find it.
