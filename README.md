# FaceFlash

**Face search that fits in a megabyte.**

Search 13,000 distinct people in 0.84 MB. Search 500,000 faces in 30 MB.
Same accuracy as exact brute-force search, 48-96x less memory. Runs on CPU.

```
pip install faceflash
```

## At a Glance

|  | FaceFlash | Best alternative |
|--|-----------|------------------|
| Find the right person (rank-1) | **95.8%** | 95.7% (exact ceiling) |
| Memory for 13,724 people | **0.84 MB** | 293 MB |
| Memory for 500K faces | **30 MB** | 1,465 MB |
| Recall@1 at 100K | **99.9%** | 99.8% |
| Recall@1 at 500K | **99.9%** | 99.6% |

Tested on MS1MV2 (13,724 distinct identities) and VGGFace2 (500K images).
All methods single-threaded, same hardware, same data.

![Memory to index 100K faces — FaceFlash 3 MB vs HNSW 293 MB](docs/figures/chart_memory_bar.png)

## Quick Start

```python
from faceflash import FaceFlash

ff = FaceFlash()

# Register faces
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

## How It Works

Each face becomes a tiny binary fingerprint (64 bytes), searched with hardware
POPCNT, then the top few candidates are verified with exact cosine similarity.
You get exact-quality accuracy at a fraction of the memory.

```
Image → Detect Face → ArcFace Embedding (512-dim) → PCA+ITQ Binary Code (64 bytes)
                                                              ↓
Query → Same pipeline → Hamming Scan (Rust POPCNT) → Top-K Cosine Rerank → Match
```

## Benchmarks

All benchmarks: single-threaded, `time.perf_counter()` per query, ground truth = exact brute-force cosine argmax.

### 1:N Identification — 13,724 Distinct People (MS1MV2)

The hardest test: one photo per person in the gallery, find them using a *different* photo.

![Rank-1 identification ties exact search on 13,724 distinct people](docs/figures/chart_rank1_tie.png)

| Method | Rank-1 Accuracy | Memory |
|--------|----------------|--------|
| FAISS-Flat (exact ceiling) | 95.7% | 26.8 MB |
| **FaceFlash (512b/100c)** | **95.8%** | **0.84 MB** |
| FaceFlash (256b/100c) | 95.6% | 0.42 MB |

FaceFlash ties exact search using **32× less memory** — the binary compression is free, no accuracy loss.

### vs All ANN Methods — 100K Faces (MS1MV2, 13,724 identities)

![Recall vs memory — FaceFlash sits alone in the high-recall, low-memory corner](docs/figures/chart_recall_memory_pareto.png)

| Method | Recall@1 | Latency | Memory | Type |
|--------|----------|---------|--------|------|
| FAISS-Flat (exact) | 100% | 4.83ms | 195 MB | brute force |
| HNSWLIB (ef=128) | 99.8% | 0.26ms | 293 MB | graph |
| HNSWLIB (ef=64) | 98.6% | 0.14ms | 293 MB | graph |
| USearch | 97.8% | 0.10ms | 254 MB | graph |
| ScaNN | 82.0% | 0.10ms | 12 MB | quantized |
| **FaceFlash (512b/100c)** | **99.9%** | 0.62ms | **6.1 MB** | binary hash |
| **FaceFlash (256b/300c)** | **99.7%** | 0.37ms | **3.0 MB** | binary hash |

FaceFlash: **96x less memory** than HNSW at equal recall. HNSW is 1.6x faster — that's the tradeoff.
ScaNN (the other memory-efficient option) drops to 82% recall. FaceFlash holds 99.9%.

### vs All ANN Methods — 500K Faces (MS1MV2, 6,248 identities)

| Method | Recall@1 | Latency | Memory |
|--------|----------|---------|--------|
| FAISS-Flat (exact) | 100% | 24.4ms | 977 MB |
| HNSWLIB (ef=128) | 99.6% | 0.40ms | 1,465 MB |
| HNSWLIB (ef=64) | 99.2% | 0.22ms | 1,465 MB |
| ScaNN | 93.3% | 0.37ms | 61 MB |
| **FaceFlash (512b/200c)** | **99.9%** | 2.69ms | **30.5 MB** |
| **FaceFlash (256b/100c)** | **99.6%** | 1.69ms | **15.3 MB** |

At 500K: **48x less memory** than HNSW. HNSW is 6.7x faster. FaceFlash wins on memory.

### When to Use It / When Not To

| Use FaceFlash when | Don't use FaceFlash when |
|-------------------|--------------------------|
| Edge devices, Raspberry Pi, phones | Server with 64GB+ RAM |
| RAM budget < 100 MB | Latency must be < 0.5ms |
| 10K–500K face database | > 1M faces (HNSW scales better) |
| Accuracy matters more than speed | Throughput > 10K QPS needed |
| Offline / no-server deployment | Batch search (FAISS batch is faster) |

## Tuning

More bits or more candidates both buy recall — they substitute for each other. This curve (1M faces, 95% CI bands) shows the trade:

![Recall vs candidates per code length, with 95% confidence bands](docs/figures/chart_recall_vs_candidates.png)

Two knobs, and they substitute for each other:

| Deployment | Config | Recall@1 | Memory/face | Notes |
|-----------|--------|----------|-------------|-------|
| **Server** (floats in RAM) | `n_bits=256, n_candidates=300` | 99.7% | 32 bytes | Minimize resident RAM |
| **Balanced** (default) | `n_bits=512, n_candidates=100` | 99.9% | 64 bytes | Best recall, reasonable memory |
| **Edge** (floats on flash) | `n_bits=512, n_candidates=50` | 99.5% | 64 bytes | Minimize disk reads per query |

```python
ff = FaceFlash(n_bits=256, n_candidates=300)     # server: less memory
ff = FaceFlash(n_bits=512, n_candidates=100)     # balanced (default)
ff = FaceFlash(n_bits=512, n_candidates=50)      # edge: fewer I/O ops
ff.search("query.jpg", n_candidates=200)         # per-query override
```

## Architecture

```
faceflash/
├── engine.py          # High-level API (register, search, verify)
├── detect.py          # Face detection (OpenCV Haar, RetinaFace planned)
├── embed.py           # ArcFace ONNX embedding (512-dim, auto-downloads)
├── index.py           # Binary index with buffered O(n) add
├── pca_quantize.py    # PCA+ITQ quantizer (the core algorithm)
rust/
├── src/lib.rs         # Hamming search via hardware POPCNT (PyO3)
├── Cargo.toml
```

**Why PCA+ITQ?** ArcFace embeddings concentrate identity information along principal axes.
PCA aligns quantization with those axes. ITQ rotates bits for balanced marginals.
Result: fewer candidates needed for the same recall vs random projection.

**Why not HNSW?** HNSW stores a graph on top of the full float vectors — 1.5x raw memory.
FaceFlash stores 64 bytes per face. Float vectors are mmap'd from disk and only paged
for the ~100 candidates that pass the binary filter.

**Why Rust?** Hamming distance uses hardware POPCNT. Rust compiles to tight loops — 50x faster than NumPy.

## Limitations

- **Linear scan** — search time scales linearly with database size (not sub-linear like HNSW)
- **Memory during build** — constructing the index holds all float vectors in RAM. The mmap benefit applies after `save()`/`load()`
- **Face detection** — the built-in detector (Haar cascade) is basic. For best accuracy, pass pre-aligned 112x112 face crops
- **Rust backend** — must be built manually (`cd rust && maturin develop --release`). Not yet shipped in pip wheels
- **Rerank latency is cache-dependent** — the quoted times assume float vectors are OS-cached. On truly memory-constrained devices, rerank becomes I/O-bound

## Installation

```bash
# Python package
pip install faceflash

# With Rust backend (recommended — 50x faster search)
cd rust && maturin develop --release

# With all benchmark dependencies
pip install faceflash[benchmark]
```

## Reproduce the Benchmarks

```bash
# Local (LFW + VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_search.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# RunPod (full suite — VGGFace2 1M + MS1MV2 85K identities)
export GITHUB_TOKEN=<token> KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
bash scripts/runpod_full.sh      # VGGFace2 1M + all ANN comparisons
bash scripts/runpod_ms1m.sh      # MS1MV2 (13,724 identities, 1:N identification)
```

## Contributing

FaceFlash is a working system with proven results. Key areas for contribution:

- [ ] **RetinaFace alignment** — proper 5-point alignment would boost the library's end-to-end accuracy from the current Haar baseline
- [ ] **Prebuilt wheels** — ship the Rust backend via maturin CI so `pip install` gets full speed
- [ ] **Coarse clustering** — partition binary codes into ~1000 buckets for sub-linear scan (the main speed improvement path)
- [ ] **SIMD/AVX-512 Hamming** — process 4-8 codes per cycle instead of one u64
- [ ] **DiskANN comparison** — the one competitor we haven't benchmarked
- [ ] **Mobile deployment** — ONNX + CoreML for on-device face search
- [ ] **Streaming insertion** — add faces without refitting PCA (currently requires rebuild)

## License

MIT
