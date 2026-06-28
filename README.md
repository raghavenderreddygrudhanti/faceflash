# FaceFlash

**Face search that fits 1M faces in 61 MB of RAM, at the same recall as exact brute-force cosine.** Stores ArcFace embeddings as 512-bit binary codes instead of full float vectors, then reranks the top candidates with exact cosine to recover accuracy.

This is a **compression + packaging** project, not a new search algorithm. The search is a brute-force Hamming scan (same as FAISS `IndexBinaryFlat`) plus a cosine rerank. The reason it works without losing accuracy is that ArcFace embeddings are low-rank, so PCA+ITQ binary codes preserve nearest-neighbor ordering. On general or random vectors this does not hold (see [Limitations](#limitations)).

- **Same Recall@1 as exact search, ~32× smaller index.** Each 512-float embedding (2,048 bytes) becomes a 64-byte binary code. Verified against FAISS-Flat ground truth on MS1MV2.
- **Lower per-query latency than HNSW up to ~200K**, because the binary scan fits in cache. From ~300K up, HNSW's O(log N) graph wins; the scan is O(N).
- **Zero config.** PCA fits automatically after 1,024 samples. No graph to build, no hyperparameters.
- **Local and offline.** No service, no network. CPU only.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/faceflash.svg)](https://pypi.org/project/faceflash/)
[![CI](https://github.com/raghavenderreddygrudhanti/faceflash/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/faceflash/actions)

</div>

```python
from faceflash import FaceFlash

ff = FaceFlash()

# Enroll faces — the name you pass is what comes back in search results
ff.register("Alice", "photos/alice_headshot.jpg")
ff.register("Bob", "photos/bob_photo.png")

ff.save("my_index/")

# Identify: detects the face, extracts the ArcFace embedding, searches in
# binary space, reranks with cosine. The filename is irrelevant — matching
# happens on the face embedding.
result = ff.search("security_cam_frame.jpg")
# {"matches": [{"name": "Alice", "confidence": 0.92}], "search_time_ms": 0.4}
```

---

## What This Is and Isn't

**Is:** a way to store and search face embeddings in ~32× less memory than full float vectors, with a one-line API that handles detection → embedding → quantization → search.

**Isn't:** a new ANN algorithm. The Hamming scan is brute force. You could reproduce the same recall with `faiss.PCAMatrix` + `IndexBinaryFlat` + manual reranking. FaceFlash just packages that pipeline and tunes the SIMD kernel.

The comparison against HNSW below is a *system-level* one ("which tool do I reach for to search faces"), not a claim that the search algorithm is novel. HNSW stores full floats plus a graph; FaceFlash stores 64-byte codes. That's the whole reason the memory differs.

---

## At a Glance

All FaceFlash numbers use the default **512-bit** config. Memory is the in-RAM binary index (float vectors are mmap'd from disk for reranking — see [Limitations](#limitations)).

| | FaceFlash | HNSWLIB | USearch | FAISS-Flat |
|---|---|---|---|---|
| **Recall@1** | **100%** | 100% | 99.5% | 100% (ground truth) |
| **Index memory @ 100K** | **6.1 MB** | 293 MB | 254 MB | 195 MB |
| **Index memory @ 1M** | **61 MB** | 2,930 MB | 2,539 MB | 1,953 MB |
| **Single-query latency @ 100K** | **0.30ms** | 0.60ms | 0.17ms | 4.90ms |
| **Single-query latency @ 1M** | 2.95ms | 0.66ms | — | — |

> "Recall@1 = 100%" means FaceFlash returns the same nearest neighbor as exact brute-force cosine (FAISS-Flat) on the same embeddings — the binary compression is lossless for retrieval. It does **not** mean ArcFace itself is perfect; embedding-model limitations (pose, occlusion, age) are upstream and unaffected.
>
> Competitor memory is the standard estimate (float vectors + index overhead; HNSW ≈ 1.5× raw). FaceFlash memory is measured. See [Benchmark Methodology](#benchmark-methodology).
>
> A **256-bit** config halves the index (3.05 MB @ 100K, 30.5 MB @ 1M) at the same recall, trading ~2× single-query latency.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_memory_scale.png" width="800" alt="Memory comparison: FaceFlash vs all competitors at 500K faces"/>

<sub><b>Index memory at 500K (256-bit config):</b> FaceFlash 15 MB vs HNSW ~1,465 MB. The gap is just float-vs-binary storage, not a smarter index.</sub>

</div>

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/demo.gif" width="720" alt="FaceFlash live demo"/>

</div>

---

## Install

```bash
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"

# With benchmark dependencies
pip install "faceflash[cpu,benchmark] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

> Requires a [Rust toolchain](https://rustup.rs); it compiles the AVX-512/NEON backend automatically and falls back to NumPy if unavailable.

---

## Quick Start

```python
from faceflash import FaceFlash

ff = FaceFlash()  # downloads ArcFace model (~166 MB) on first run

# Register individual faces — "Alice" is the label, the .jpg is the face photo
ff.register("Alice", "alice.jpg")
ff.register("Bob", "bob.jpg")

# Identify a face — extracts face from image, compares against registered embeddings
result = ff.search("query.jpg")
# {"matches": [{"name": "Alice", "confidence": 0.92}], "search_time_ms": 0.4}

# Verify two faces are the same person
ff.verify("photo1.jpg", "photo2.jpg")
# {"match": True, "confidence": 0.87}

# Bulk enroll from folder (expects folder/person_name/photo.jpg)
ff.register_folder("employees/")
ff.save("my_index/")
ff.load("my_index/")

# Manage the index
len(ff)                  # how many faces are registered
"Alice" in ff            # is this person registered?
ff.names()               # list registered people
ff.remove("Alice")       # unregister a person (returns count removed)
```

> For best accuracy, use pre-aligned 112x112 face crops. 5-point alignment (SCRFD/RetinaFace) adds +1.28 accuracy points over a basic center-crop.

---

## Gallery Management

```python
from faceflash import FaceFlash

ff = FaceFlash()
ff.register("Alice", "alice.jpg")
ff.register("Bob", "bob.jpg")
ff.register("Alice", "alice2.jpg")  # multiple photos per person

# Check gallery state
len(ff)              # 3 (total face entries)
ff.names()           # ["Alice", "Bob"]
"Alice" in ff        # True
"Charlie" in ff      # False

# Remove a person (GDPR / right-to-be-forgotten)
ff.remove("Bob")     # removes all entries for Bob
len(ff)              # 2
ff.names()           # ["Alice"]

# Monitor index stats
ff.stats()
# {'count': 2, 'pca_fitted': True, 'rust_backend': True,
#  'binary_memory_mb': 0.0, 'resident_memory_mb': 0.0, ...}
```

## Batch Identification

Process many query faces at once (~8x faster than one-at-a-time at 100K):

```python
import numpy as np
from faceflash import FaceFlash

ff = FaceFlash()
ff.register_folder("gallery/")
ff.save("my_index/")

# Low-level batch search (for bulk dedup, watchlists, video frames)
embeddings = np.load("query_embeddings.npy")  # (N, 512) float32
results = ff.index.search_batch(embeddings, k=1)
# results[i] = [(name, similarity, index), ...]

# Example: find all matches above threshold
for i, matches in enumerate(results):
    if matches and matches[0][1] > 0.5:
        print(f"Query {i}: {matches[0][0]} (confidence {matches[0][1]:.2f})")
```

---

## Is FaceFlash Right for You?

| Scenario | Why FaceFlash fits |
|---|---|
| **Edge / mobile / IoT** | 6–61 MB index vs 293 MB–2.9 GB for HNSW — fits in device RAM |
| **Multi-tenant servers** | 100 galleries × 61 MB = 6 GB. HNSW: 100 × 2.9 GB = 290 GB |
| **10K–200K face databases** | The latency sweet spot — lower latency *and* less memory than HNSW |
| **Offline / air-gapped** | Runs on a Raspberry Pi or cheap VPS, no GPU, no network |

**When HNSW is the better choice:**
- More than ~500K faces and you need sub-millisecond single-query latency (HNSW's O(log N) beats the linear scan)
- You have the RAM to spare and latency matters more than footprint
- You need a battle-tested index with years of production hardening

---

## How It Works

FaceFlash is a **face recognition** system: it detects faces in images, converts them to numerical embeddings, and searches by visual similarity. Filenames are just how you point it to the image file; the matching happens in embedding space.

![FaceFlash Architecture Pipeline](https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/architecture_pipeline.png)

Each face is compressed into a **64-byte binary fingerprint**:

1. **ArcFace** extracts a 512-dimensional float embedding
2. **PCA** aligns the quantization with the axes where identity varies most
3. **ITQ** rotates bits to maximize information per bit (balanced marginals)
4. **AVX-512 VPOPCNTDQ** scans all binary codes in a single instruction per face
5. **Cosine rerank** runs exact similarity on only the top ~100 candidates

This is why 512 bits is the fastest setting: the entire code fits in one AVX-512 register.

---

## Is the win the compression or the search?

The compression. FaceFlash's Hamming scan is brute force, structurally the same as FAISS `IndexBinaryFlat`. To prove the kernel isn't doing anything special, `benchmarks/bench_compression_isolation.py` runs the **same** PCA+ITQ codes through both FaceFlash's Rust kernel and FAISS `IndexBinaryFlat`, then reranks identically:

| Method (same codes) | Recall@1 | Avg latency |
|---|---|---|
| FaceFlash kernel + cosine rerank | identical | comparable |
| FAISS `IndexBinaryFlat` + cosine rerank | identical | comparable |
| FAISS `IndexBinaryFlat`, no rerank | much lower | — |

The recall comes from the PCA+ITQ compression preserving nearest-neighbor ordering on face embeddings, plus the cosine rerank. Not from the search algorithm. Run it yourself to confirm.

---

## Benchmark Methodology

FaceFlash's recall, latency, throughput, and memory are measured. Competitor recall and latency are also measured; **competitor memory is estimated** from the standard vectors + index-structure overhead formula (e.g. HNSW ≈ 1.5× raw vectors).

| | Details |
|---|---|
| **Dataset** | MS1MV2 — 645,019 ArcFace embeddings, 44,291 distinct identities |
| **Embedding** | ArcFace ONNX (w600k_r50), 512 dimensions, L2-normalized |
| **Hardware** | AMD EPYC 9355 (32 cores / 128 threads), AVX-512 VPOPCNTDQ enabled |
| **Competitors** | HNSWLIB 0.8+, FAISS 1.7+, USearch 2.x, ScaNN |
| **Ground truth** | Exact brute-force cosine argmax (FAISS-Flat) |
| **Timing** | `time.perf_counter()` per query, 10 warmup excluded |
| **Recall metric** | Recall@1 — fraction of queries where the true nearest neighbor is rank-1 |
| **Memory metric** | FaceFlash: measured binary index size in RAM (floats mmap'd after save/load). Competitors: estimated (vectors + index-structure overhead) |
| **Batched timing** | Wall-clock for the full query batch / number of queries |
| **Reproducibility** | `bash scripts/runpod_ms1m.sh` reproduces all results end-to-end |

All single-query rows are single-threaded. Batched rows use all available cores. Every benchmark script validates correctness before reporting speed.

---

## Performance

### Scale Summary (100K-1M)

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_throughput_scale.png" width="760" alt="Batched throughput: FaceFlash vs all competitors 100K to 1M"/>

<sub><b>Batched Throughput (QPS):</b> FaceFlash 100K→1M — 4.8× faster than HNSW at 100K</sub>

</div>

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_latency_scale.png" width="760" alt="Single-query latency: all methods 100K to 1M"/>

<sub><b>Single-Query Latency:</b> FaceFlash 0.30ms vs HNSW 0.60ms at 100K — both at 100% recall</sub>

</div>

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_recall_memory_scale.png" width="760" alt="Recall vs Memory at 500K"/>

<sub><b>Recall vs memory:</b> 100% recall at ~30 MB for 500K faces (512-bit config)</sub>

</div>

| Scale | Recall@1 | Single-query | Batched QPS | Index memory |
|---|---|---|---|---|
| 100K | 100% | 0.30ms | 27,661 | 6.1 MB |
| 200K | 100% | 0.57ms | 19,930 | 12.2 MB |
| 300K | 100% | 0.84ms | 15,147 | 18.3 MB |
| 500K | 100% | 1.45ms | 10,337 | 30.5 MB |
| 1M | 100% | 2.95ms | 5,403 | 61 MB |

Default **512-bit** config. The **256-bit** config holds the same recall at half the memory and roughly double the single-query latency.

Up to ~200K the binary scan is faster per query than HNSW because it stays in cache. From ~300K up, HNSW's O(log N) graph wins on single-query latency while FaceFlash keeps the smaller footprint.

### 1:N Identification - 44,290 Distinct People

The hardest test: one photo per person in the gallery, identify them from a different photo.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_rank1_tie.png" width="720" alt="Rank-1 identification ties exact search on 44,290 people"/>

<sub><b>1:N Identification on 44,290 Identities:</b> FaceFlash matches FAISS-Flat rank-1 accuracy with 32× smaller codes</sub>

</div>

| Method | Rank-1 Accuracy | Memory |
|---|---|---|
| FAISS-Flat (exact ceiling) | 95.8% | 86.5 MB |
| **FaceFlash (512b / 100c)** | **95.8%** | **2.70 MB** |
| FaceFlash (512b / 300c) | 95.8% | 2.70 MB |
| FaceFlash (256b / 100c) | 95.6% | 1.35 MB |

FaceFlash matches FAISS-Flat's rank-1 accuracy here while storing 32× smaller codes (512 float32 → 512 bits). On this benchmark the binary codes lose no rank-1 accuracy after cosine rerank. That's an empirical result on ArcFace embeddings, not a general guarantee.

<details>
<summary>Detailed per-scale benchmark tables</summary>

### 100K Faces (6,939 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (batched) | 100% | 0.036ms | 27,661 | 6.1 MB |
| FaceFlash (512b/200c) | 100% | 0.30ms | 3,310 | 6.1 MB |
| FaceFlash (512b/100c) | 100% | 0.43ms | 2,344 | 6.1 MB |
| HNSWLIB (ef=128) | 100% | 0.60ms | 1,671 | 293 MB |
| HNSWLIB batched | 100% | 0.172ms | 5,813 | 293 MB |
| USearch batched | 99.5% | 0.007ms | 137,264 | 254 MB |
| USearch | 99.5% | 0.17ms | -- | 254 MB |
| ScaNN | 98.3% | 0.10ms | -- | 12 MB |
| FAISS-Flat (exact) | 100% | 4.90ms | 204 | 195 MB |

### 200K Faces (13,749 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (batched) | 100% | 0.050ms | 19,930 | 12.2 MB |
| FaceFlash (512b/200c) | 100% | 0.57ms | 1,751 | 12.2 MB |
| HNSWLIB (ef=128) | 99.9% | 0.65ms | 1,531 | 586 MB |
| HNSWLIB batched | 99.9% | 0.378ms | 2,646 | 586 MB |
| USearch batched | 99.1% | 0.008ms | 121,660 | 508 MB |
| ScaNN | 97.2% | 0.19ms | -- | 24 MB |

### 300K Faces (20,615 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (batched) | 100% | 0.066ms | 15,147 | 18.3 MB |
| FaceFlash (512b/200c) | 100% | 0.84ms | 1,187 | 18.3 MB |
| HNSWLIB (ef=128) | 99.9% | 0.66ms | 1,510 | 879 MB |
| HNSWLIB batched | 99.7% | 0.269ms | 3,715 | 879 MB |
| USearch batched | 98.7% | 0.014ms | 73,383 | 762 MB |
| ScaNN | 97.8% | 0.28ms | -- | 37 MB |

### 500K Faces (34,328 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (batched) | 100% | 0.097ms | 10,337 | 30.5 MB |
| FaceFlash (512b/200c) | 100% | 1.45ms | 692 | 30.5 MB |
| HNSWLIB (ef=128) | 100% | 0.71ms | 1,416 | 1,465 MB |
| HNSWLIB batched | 99.9% | 0.179ms | 5,577 | 1,465 MB |
| USearch batched | 98.4% | 0.013ms | 76,150 | 1,270 MB |
| ScaNN | 97.6% | 0.45ms | -- | 61 MB |

### 1M Faces (44,291 identities)

| Method | Recall@1 | Latency | QPS | Memory |
|---|---|---|---|---|
| FaceFlash (batched) | 100% | 0.185ms | 5,403 | 61 MB |
| FaceFlash (512b/100c) | 100% | 2.92ms | 342 | 61 MB |
| HNSWLIB (ef=128) | 100% | 0.66ms | 1,523 | 2,930 MB |
| HNSWLIB batched | 100% | 0.178ms | 5,621 | 2,930 MB |
| USearch batched | 94.1% | 0.013ms | 77,266 | 2,539 MB |
| ScaNN | 98.2% | 0.86ms | -- | 122 MB |

</details>

---

## Tuning

Pick a config that matches your deployment:

| Deployment | Config | Recall@1 | Memory/face | Notes |
|---|---|---|---|---|
| Ultra-compact (mobile/IoT) | n_bits=128, n_candidates=500 | 99.4% | 16 bytes | Minimum RAM |
| Balanced | n_bits=256, n_candidates=100 | 100% | 32 bytes | Good default |
| **Default (fastest)** | n_bits=512, n_candidates=100 | **100%** | 64 bytes | One AVX-512 instruction |
| Edge (minimize disk reads) | n_bits=512, n_candidates=50 | 99.5% | 64 bytes | -- |

```python
ff = FaceFlash(n_bits=128, n_candidates=500)   # mobile/IoT
ff = FaceFlash(n_bits=256, n_candidates=100)   # balanced
ff = FaceFlash(n_bits=512, n_candidates=100)   # default: fastest

ff.search("query.jpg", n_candidates=200)       # per-query override
```

### Speed up large indexes with clustering

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_clustering_tradeoff.png" width="720" alt="Clustering recall/speed tradeoff"/>

<sub><b>IVF clustering:</b> ~3–5× faster at 500K at ~88–93% recall — a configurable recall/speed trade-off</sub>

</div>

```python
ff.index.build_clusters(n_probe=16)
```

| Scale | n_probe | Recall@1 | Latency | Speedup |
|---|---|---|---|---|
| 100K | 16 | 96.1% | 0.12ms | 2.6x |
| 100K | 32 | 98.5% | 0.17ms | 1.8x |
| 500K | 16 | 87.9% | 0.31ms | 5.0x |
| 500K | 32 | 92.8% | 0.56ms | 2.8x |

Clustering is mainly useful at 500K+. At ~88–93% recall it gives ~3–5× speedup (n_probe 16–32). If you accept lower recall it goes higher — ~8× at ~80% recall (n_probe 8), ~12× at ~70% (n_probe 4).

---

## Architecture

```
faceflash/                          # Python package
├── engine.py         ◀─ High-level API (register, search, verify)
├── detect.py         ◀─ Face detection (SCRFD + Haar fallback)
├── align.py          ◀─ 5-point alignment to ArcFace template
├── embed.py          ◀─ ArcFace ONNX embedding (512-dim, auto-download)
├── index.py          ◀─ Binary index + batched search
└── pca_quantize.py   ◀─ PCA+ITQ quantizer (the core algorithm)

rust/                               # Rust backend (PyO3 + Rayon)
├── src/lib.rs        ◀─ AVX-512 VPOPCNTDQ / NEON / scalar POPCNT
└── Cargo.toml
```

**Why PCA+ITQ?** ArcFace embeddings concentrate identity information along principal axes. PCA aligns quantization with those axes; ITQ rotates bits for balanced marginals. On ArcFace embeddings this preserves rank-1 accuracy after rerank; it isn't a general guarantee for arbitrary vectors.

**Why not HNSW internally?** HNSW stores a graph on top of full float vectors (~1.5× raw memory). FaceFlash stores 32–64 bytes per face and mmaps the float vectors from disk, paging only the top ~100 candidates per query. Trade-off: higher single-query latency past 500K, smaller footprint.

**Why Rust + AVX-512?** AVX-512 VPOPCNTDQ processes an entire 512-bit code in one instruction (~3× faster than scalar POPCNT). Combined with Rayon multi-core parallelism (and cache-blocked batching to keep the scan cache-friendly), batched search reaches ~8–16× throughput versus single-query serial. Runtime-detected, so no user configuration is needed.

---

## Limitations

- **Single-query at 1M+:** O(N) linear scan; HNSW is 4.4x faster per single query at 1M. Batched path ties.
- **Memory during build:** holds all float vectors in RAM. The compression savings apply after `save()` / `load()`, when float vectors move to mmap'd disk.
- **AVX-512 VPOPCNTDQ:** the ~3× kernel speedup requires Ice Lake / Zen 4+ / EPYC 9004+. Older CPUs fall back to scalar POPCNT automatically.
- **Rerank I/O:** pages ~100 float rows from disk per query. Invisible on NVMe; adds latency on slow storage.

---

## Reproduce the Benchmarks

```bash
# Local (LFW + VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# RunPod (full suite)
export GITHUB_TOKEN=<token>
bash scripts/runpod_ms1m.sh   # FORCE_EXTRACT=1 for full 85K extraction
```

---

## Roadmap

**v0.1.0 (current)**
- [x] PCA+ITQ binary quantization + Rust search backend
- [x] High-level API: register, search, verify
- [x] Benchmarked against FAISS, HNSWLIB, USearch, ScaNN at 100K-1M
- [x] 1:N identification on 44,290 distinct identities (MS1MV2)
- [x] 5-point alignment via SCRFD/RetinaFace — 99.85% LFW accuracy

**v0.2.0 (done)**
- [x] Prebuilt wheels (`pip install faceflash`)
- [x] Full 85K-identity benchmark (76,872 identities extracted, 44,291 with sufficient data)
- [x] On-device memory measurement (3.05 MB binary index @100K with 256b)

**v0.3.0 (done)**
- [x] IVF coarse clustering (configurable recall/speed trade-off, ~3–5× at 88–93% recall)
- [x] AVX-512 VPOPCNTDQ — native 512-bit popcount (~3x faster than scalar)
- [x] Batched search — ~16x throughput at 500K-1M (multi-core + VPOPCNTDQ; cache-blocked)
- [x] NEON kernels — ARM-optimized (vcntq_u8)

**v1.0.0 — next**
- [ ] Stable public API (no breaking changes)
- [ ] DiskANN comparison
- [ ] Mobile deployment (ONNX + CoreML)
- [ ] Streaming insertion (add faces without refitting PCA)

---

## Contributing

```bash
# Dev setup (one command)
git clone https://github.com/raghavenderreddygrudhanti/faceflash.git
cd faceflash && python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,benchmark]" && maturin develop --release
python -m pytest tests/  # 33 tests, ~17s
```

Open areas for contribution:

| Area | Difficulty | Impact |
|------|-----------|--------|
| **DiskANN comparison** | Medium | High — the one competitor missing |
| **Mobile deployment** (ONNX + CoreML) | Medium | High — iOS/Android face search |
| **Streaming insertion** (no PCA refit) | Hard | High — online learning |
| **GPU batched search** (CUDA) | Hard | Medium — 10M+ galleries |
| **Raspberry Pi / Jetson benchmarks** | Easy | Medium — edge credibility |
| **WebAssembly build** | Medium | Medium — browser face search |

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding guidelines.

---

## Credits & References

FaceFlash does **not** introduce a new embedding model or hashing algorithm. It combines published techniques into a CPU-efficient retrieval pipeline with a Rust SIMD kernel, and measures the result honestly. The contribution is the packaging and the measurement, not the underlying math.

It builds directly on:

| Component | Reference |
|-----------|-----------|
| **Binary quantization** (PCA + ITQ — the core method) | Gong & Lazebnik, *"Iterative Quantization: A Procrustean Approach to Learning Binary Codes,"* CVPR 2011 |
| **Face embeddings** | Deng, Guo, Xue & Zafeiriou, *"ArcFace: Additive Angular Margin Loss for Deep Face Recognition,"* CVPR 2019 — weights from [InsightFace](https://github.com/deepinsight/insightface) |
| **Detection & 5-point alignment** | *RetinaFace* (CVPR 2020) / *SCRFD* (ICLR 2022), InsightFace |
| **Dataset** | Guo et al., *"MS-Celeb-1M,"* ECCV 2016 (MS1MV2 = ArcFace's refined/cleaned version) |
| **Verification benchmark** | Huang et al., *"Labeled Faces in the Wild (LFW),"* UMass TR 2007 |
| **Baselines compared** | FAISS (Johnson et al.), HNSW (Malkov & Yashunin, TPAMI 2018), ScaNN (Guo et al., ICML 2020), [USearch](https://github.com/unum-cloud/usearch) |

PCA dates to Pearson (1901) / Hotelling (1933); the Hamming distance to Hamming (1950); `POPCNT` / `VPOPCNTDQ` are Intel/AMD hardware instructions. FaceFlash's value is in how these are combined and implemented, not in inventing them.

---

## License

MIT — see [LICENSE](LICENSE).

If FaceFlash is useful to you, a star helps others find it.
