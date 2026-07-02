# FaceFlash

Face search that keeps 1M faces in 61 MB of RAM at the same recall@1 as exact brute-force cosine. ArcFace embeddings are stored as 512-bit binary codes instead of full float vectors, scanned by Hamming distance, and the top ~100 candidates are reranked with exact cosine.

One thing to understand up front: the 61 MB is the in-RAM index. The original float vectors (~2 GB at 1M) stay on disk, and the rerank pages in ~100 rows per query. So it's 61 MB of RAM plus 2 GB of disk — not 61 MB total. RAM is usually the scarce resource, which is why the number is framed that way.

This is a compression + packaging project, not a new search algorithm. The search is a brute-force Hamming scan (same as FAISS `IndexBinaryFlat`) plus a cosine rerank. You could build the same pipeline from `faiss.PCAMatrix` + `IndexBinaryFlat` + manual reranking; FaceFlash packages it behind one API and tunes the SIMD kernel. The reason it works without losing accuracy is that ArcFace embeddings are low-rank, so PCA+ITQ binary codes preserve nearest-neighbor ordering. On general or random vectors this does not hold (see [Limitations](#limitations)).

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/faceflash.svg)](https://pypi.org/project/faceflash/)
[![CI](https://github.com/raghavenderreddygrudhanti/faceflash/actions/workflows/ci.yml/badge.svg)](https://github.com/raghavenderreddygrudhanti/faceflash/actions)

</div>

```python
from faceflash import FaceFlash

ff = FaceFlash()                       # downloads ArcFace model (~166 MB) on first run
ff.register("Alice", "alice.jpg")      # the name is the label you get back
ff.register("Bob", "bob.jpg")
ff.register_folder("employees/")       # bulk enroll (folder/person_name/photo.jpg)
ff.save("my_index/")

result = ff.search("query.jpg")        # detect → embed → binary search → cosine rerank
# {"matches": [{"name": "Alice", "confidence": 0.92}], ...}

ff.verify("photo1.jpg", "photo2.jpg")  # 1:1 — same person?
# {"match": True, "confidence": 0.87}

ff.remove("Bob")                       # GDPR / right-to-be-forgotten
```

There's also a batch path (`ff.index.search_batch(embeddings, k=1)`) for bulk dedup, watchlists, and video frames — roughly 8× the throughput of one-at-a-time at 100K.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/demo.gif" width="720" alt="FaceFlash live demo"/>

</div>

---

## The numbers

All FaceFlash numbers use the default 512-bit config, measured on MS1MV2 embeddings with an AMD EPYC 9355 (AVX-512). Recall@1 = returns the same nearest neighbor as exact brute-force cosine (FAISS-Flat ground truth). That's a statement about the compression being lossless for retrieval — not about ArcFace being perfect; embedding-model limits (pose, occlusion, age) are upstream.

| | FaceFlash | HNSWLIB | USearch | FAISS-Flat |
|---|---|---|---|---|
| **Recall@1** | **100%** | 100% | 99.5% | 100% (ground truth) |
| **Index RAM @ 100K** | **6.1 MB** | 293 MB | 254 MB | 195 MB |
| **Index RAM @ 1M** | **61 MB** | 2,930 MB | 2,539 MB | 1,953 MB |
| **Single-query @ 100K** | **0.30ms** | 0.60ms | 0.17ms | 4.90ms |
| **Single-query @ 1M** | 2.95ms | 0.66ms | — | — |

Two honesty notes on this table:

- Competitor memory was *estimated* (vectors + overhead, HNSW ≈ 1.5× raw). The benchmark now measures saved-index file sizes instead, and the estimates run high — measured at 100K, HNSW is 222 MB and USearch is 112 MB. Measured numbers land here after the next full run; until then treat competitor memory as an upper bound.
- The 1M scale is tiled from 645K real embeddings. Recall scoring is tie-aware (an equidistant duplicate counts as correct for every method), so tiling doesn't favor anyone.

| Scale | Recall@1 | Single-query | Batched QPS | Index RAM |
|---|---|---|---|---|
| 100K | 100% | 0.30ms | 27,661 | 6.1 MB |
| 200K | 100% | 0.57ms | 19,930 | 12.2 MB |
| 300K | 100% | 0.84ms | 15,147 | 18.3 MB |
| 500K | 100% | 1.45ms | 10,337 | 30.5 MB |
| 1M | 100% | 2.95ms | 5,403 | 61 MB |

Up to ~200K the binary scan is faster per query than HNSW because the codes fit in cache. From ~300K up, HNSW's O(log N) graph wins on single-query latency while FaceFlash keeps the smaller footprint. A 256-bit config halves the index at the same recall for about 2× the single-query latency.

<div align="center">

<img src="https://raw.githubusercontent.com/raghavenderreddygrudhanti/faceflash/main/docs/figures/chart_memory_scale.png" width="760" alt="Memory comparison at 500K faces"/>

<sub>Index memory at 500K: the gap is float-vs-binary storage, not a smarter index.</sub>

</div>

<details>
<summary>Per-scale benchmark tables (all methods)</summary>

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

### 1:N identification, 44,290 distinct people

One photo per person in the gallery, identify them from a different photo — the hardest test:

| Method | Rank-1 Accuracy | Memory |
|---|---|---|
| FAISS-Flat (exact ceiling) | 95.8% | 86.5 MB |
| **FaceFlash (512b / 100c)** | **95.8%** | **2.70 MB** |
| FaceFlash (256b / 100c) | 95.6% | 1.35 MB |

The binary codes lose no rank-1 accuracy after rerank. That's an empirical result on ArcFace embeddings, not a general guarantee.

### What a confidence threshold actually buys you

The `confidence` field is raw cosine similarity, not a probability — 0.92 does not mean "92% sure". What a threshold means in error rates, measured on LFW (15K genuine / 200K impostor pairs, `benchmarks/bench_far_frr.py`):

| Threshold | False accept rate | False reject rate |
|---|---|---|
| 0.30 | 0.012% | 3.4% |
| **0.40 (default)** | < 0.001% | 10.1% |
| 0.50 | < 0.001% | 27.2% |

The equal-error rate is ~1.0% at threshold ~0.17. The 0.4 default is deliberately conservative: essentially zero false accepts, at the cost of rejecting ~10% of genuine matches. If your gallery photos are good and a false reject just means "try again", 0.4 is right; if false rejects are expensive, drop toward 0.3. Rerun the script on embeddings from your own population — these rates shift with demographics and photo quality.

### Is the win the compression or the search?

The compression. Two benchmarks prove it: `bench_compression_isolation.py` runs the same PCA+ITQ codes through FaceFlash's Rust kernel and FAISS `IndexBinaryFlat` with identical reranking — same recall, so the kernel does nothing accuracy-wise. And `bench_ann_comparison.py` includes `IndexBinaryFlat` and `IndexBinaryHNSW` rows over the same 64-byte codes, so the compression win and the search-structure cost are separate columns you can read off.

One result from that worth knowing: binary-HNSW over the same codes is much faster than the linear scan but drops recall (92% vs 99.5%+ at 100K, same top-100 rerank) — the graph over Hamming space misses candidates the exhaustive scan finds. That, not the SIMD kernel, is the actual argument for scanning.

### Methodology, briefly

MS1MV2 (645,019 embeddings, 44,291 identities), ArcFace w600k_r50, L2-normalized. Ground truth is exact brute-force cosine. Single-query rows are single-threaded; batched rows use all cores on both sides. Memory is measured from saved-index file size where the library supports it (HNSWLIB `save_index`, USearch `save`, FAISS `write_index`); each result row's `memory_note` says whether it was measured or estimated. A dedicated mmap-rerank row measures FaceFlash latency with floats on disk — the mode the RAM number describes. `bash scripts/runpod_ms1m.sh` reproduces everything end-to-end.

---

## Install

```bash
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"
```

Needs a [Rust toolchain](https://rustup.rs) to compile the AVX-512/NEON backend; falls back to NumPy without it. Add `,benchmark` to the extras to get the comparison dependencies. For best accuracy, feed it pre-aligned 112×112 face crops — 5-point alignment (SCRFD/RetinaFace) adds +1.28 accuracy points over a basic center-crop.

## When to use it

- Edge / mobile / IoT — megabytes of RAM instead of hundreds; floats stay on disk
- Multi-tenant servers — 100 galleries × 61 MB = 6 GB of RAM (plus ~200 GB of float stores on disk); HNSW needs its ~290 GB *in RAM*
- 10K–200K galleries — the sweet spot where the scan beats HNSW on latency *and* memory
- Offline / air-gapped — a Raspberry Pi or cheap VPS, no GPU, no network

Use HNSW instead when you're past ~500K faces and need sub-millisecond single queries, when RAM isn't your constraint, or when you want years of production hardening.

## How it works

1. ArcFace extracts a 512-d float embedding
2. PCA aligns the quantization with the axes where identity varies most
3. ITQ rotates the bits so each carries maximal information
4. The Rust kernel scans all codes — AVX-512 VPOPCNTDQ counts a whole 512-bit code's set bits in one popcount instruction (each compare is still load + XOR + popcount + accumulate)
5. Exact cosine on the top ~100 Hamming candidates picks the winner

512 bits is the fastest setting because the whole code fits in one AVX-512 register. Config options if you need them:

```python
ff = FaceFlash(n_bits=128, n_candidates=500)   # minimum RAM (16 B/face, 99.4% recall)
ff = FaceFlash(n_bits=256, n_candidates=100)   # half the memory, ~2x the latency
ff = FaceFlash(n_bits=512, n_candidates=100)   # default, fastest
ff.search("query.jpg", n_candidates=200)       # per-query override
```

Past 500K you can trade recall for speed with IVF clustering: `ff.index.build_clusters(n_probe=16)` gives ~5× at ~88% recall, `n_probe=32` gives ~2.8× at ~93%.

```
faceflash/
├── engine.py         high-level API (register, search, verify)
├── detect.py         face detection (SCRFD + Haar fallback)
├── align.py          5-point alignment to the ArcFace template
├── embed.py          ArcFace ONNX embedding (auto-download)
├── index.py          binary index + batched search
└── pca_quantize.py   PCA+ITQ quantizer (the core)
rust/src/lib.rs       AVX-512 VPOPCNTDQ / NEON / scalar POPCNT kernels
```

## Limitations

- Single-query latency at 1M+ is O(N); HNSW is ~4.4× faster there. The batched path ties.
- The RAM number is not the storage number: 61 MB @ 1M in RAM, ~2 GB of floats on disk for reranking.
- Building the index holds all floats in RAM; the savings apply after `save()`/`load()`, when floats move to mmap'd disk.
- The rerank pages ~100 float rows from disk per query — measured at ~0.2ms over RAM-resident rerank at 100K (warm cache); slow storage adds more.
- The single-popcount kernel needs Ice Lake / Zen 4+ / EPYC 9004+; older CPUs fall back to AVX2 or scalar automatically.
- The recall result is specific to face embeddings (low-rank). Don't expect it on arbitrary vectors.

## Reproduce

```bash
# Local quick check (LFW / VGGFace2 100K)
python scripts/extract_lfw_embeddings.py
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# Full suite on any Linux box
bash scripts/runpod_ms1m.sh
```

Or the [Colab notebook](https://colab.research.google.com/github/raghavenderreddygrudhanti/faceflash/blob/main/examples/faceflash_reproduce_colab.ipynb) — 5 minutes on a free CPU, no tokens. It runs a NumPy fallback (no Rust on Colab), so recall shows ~98–99% instead of 100% due to `argpartition` tie-breaking; memory numbers are identical.

## Contributing

```bash
git clone https://github.com/raghavenderreddygrudhanti/faceflash.git
cd faceflash && python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,benchmark]" && maturin develop --release
python -m pytest tests/   # ~20s
```

Things I haven't gotten to, roughly in order of how much I want them: a DiskANN comparison (the one competitor missing), mobile deployment (ONNX + CoreML), streaming insertion without a PCA refit, Raspberry Pi / Jetson benchmarks, a WebAssembly build, GPU batched search. Issues are tagged; see [CONTRIBUTING.md](CONTRIBUTING.md). Done work lives in the [CHANGELOG](CHANGELOG.md).

## Credits

FaceFlash doesn't introduce a new embedding model or hashing algorithm — the contribution is the packaging and the measurement. It builds on:

- **PCA+ITQ** — Gong & Lazebnik, *Iterative Quantization*, CVPR 2011 (the core method)
- **ArcFace** — Deng et al., CVPR 2019; weights from [InsightFace](https://github.com/deepinsight/insightface)
- **Detection/alignment** — RetinaFace (CVPR 2020) / SCRFD (ICLR 2022), InsightFace
- **Data** — MS-Celeb-1M (ECCV 2016, MS1MV2 cleanup), LFW (UMass TR 2007)
- **Baselines** — FAISS, HNSW (Malkov & Yashunin), ScaNN, [USearch](https://github.com/unum-cloud/usearch)

## License

MIT — see [LICENSE](LICENSE). If this is useful to you, a star helps others find it.
