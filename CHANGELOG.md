# Changelog

All notable changes to FaceFlash are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-27

First public release. (Everything below shipped in the initial release; earlier
0.1/0.2/0.3 labels were internal development milestones, never published.)

### Added
- **Core retrieval** — PCA+ITQ binary quantization with a two-phase search
  (Hamming filter → exact cosine rerank). 100% Recall@1 vs exact search at
  48–96× less memory than HNSW, measured on MS1MV2 from 100K to 1M faces.
- **High-level API** (`FaceFlash`) — `register`, `register_folder`, `search`,
  `verify`, plus index management: `remove`, `names`, `save`, `load`, `stats`,
  and Pythonic `len()` / `in` / `repr()`.
- **Low-level index** (`FaceIndex`) — direct embedding add/search, `add_batch`,
  `search_batch` (batched bulk throughput), and `build_clusters` (IVF).
- **Rust SIMD backend** — runtime-dispatched AVX-512 VPOPCNTDQ (x86, ~3× faster
  than scalar), NEON (ARM), and scalar POPCNT fallback. Multi-core, cache-blocked
  batched search (10–17× throughput vs single-query serial on a 128-thread EPYC).
- **5-point alignment** — SCRFD/RetinaFace alignment to the ArcFace template for
  raw photos (+1.28 LFW points over a basic center-crop), with a Haar fallback.
- **Coarse clustering (IVF)** — optional sub-linear scan for large galleries.
- **Packaging** — `pip install faceflash` ships the Rust backend; CPU/GPU
  inference via the `[cpu]` / `[gpu]` extras. Fully type-hinted (PEP 561).
- **Benchmarks** — reproducible suite vs FAISS, HNSWLIB, USearch, and ScaNN at
  100K–1M, with tie-aware recall, a fair batched-throughput comparison, and
  measured (not estimated) FaceFlash memory. 1:N identification: 95.8% rank-1
  on 44,290 identities (ties exact search).

### Known limitations
- Single-query latency at 1M is O(N); HNSW is faster there (batched ties).
- Competitor memory figures are estimated (vectors + index-overhead formula);
  FaceFlash's memory is measured.
- The 1M benchmark scale tiles 645K real embeddings 2× (flagged in results).

[0.1.0]: https://github.com/raghavenderreddygrudhanti/faceflash/releases/tag/v0.1.0
