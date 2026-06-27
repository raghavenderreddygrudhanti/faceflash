# Changelog

## 0.3.0 (2026-06-27)

### Performance
- AVX-512 VPOPCNTDQ kernel — native 512-bit popcount (3.5x faster than scalar POPCNT on Ice Lake/Zen 4+/EPYC 9004+)
- Cache-blocked batched search — 17x throughput at 500K-1M, sidesteps memory-bandwidth wall
- NEON kernels — ARM-optimized (vcntq_u8 native byte popcount)
- Coarse clustering (IVF) — 2.7-4.9x faster at 100K-400K with `build_clusters()`

### Benchmarks (canonical: AMD EPYC 9355, 128 threads, AVX-512 active)
- 100% Recall@1 at every scale (100K, 200K, 300K, 500K, 1M)
- 96x less memory than HNSW (256b config: 3 MB at 100K, 15 MB at 500K, 30 MB at 1M)
- Batched: 27,661 QPS at 100K (4.8x HNSW); 10,337 QPS at 500K (1.9x HNSW)
- Single-query: 0.30ms at 100K (2x faster than HNSW ef=128)
- 1:N identification: 95.8% rank-1 on 44,290 identities (ties exact search)
- Benchmarked against FAISS-Flat, FAISS-IVF, HNSWLIB, USearch, ScaNN at 100K-1M

## 0.2.0 (2026-06-26)

### Features
- 5-point face alignment (SCRFD/RetinaFace) — 99.85% LFW accuracy (+1.28 over Haar)
- Prebuilt wheels via maturin CI (no Rust toolchain needed for end users)
- Full 85K-identity extraction from MS1MV2 (76,872 identities, 645K embeddings)
- On-device memory measurement (actual RSS: 6.1 MB binary index at 100K)
- `FaceIndex.search_batch()` — batched search via single FFI call

### Infrastructure
- `runpod_ms1m.sh` — self-contained benchmark pipeline (clone → build → extract → benchmark → push)
- Multi-scale charts (memory, throughput, latency, recall-vs-memory)
- `bench_batch_qps.py` — isolated throughput benchmark (per-query vs batched)

## 0.1.0 (2026-06-25)

Initial public release.

### Features
- PCA+ITQ binary quantization for face embeddings (128/256/384/512 bits)
- Rust POPCNT backend (hardware-accelerated Hamming distance)
- Two-phase search: Hamming filter -> cosine rerank
- High-level API: `register()`, `search()`, `verify()`, `register_folder()`
- Buffered O(n) `add()` with auto-flush
- `save()`/`load()` with mmap'd float vectors (resident memory = binary index only)
- Configurable `n_bits` and `n_candidates` knobs
- Face detection (SCRFD + Haar fallback)
- ArcFace ONNX embedding (512-dim, auto-downloads)

### Benchmarks
- MS1MV2: 100% recall@1 at 100K, 6.1 MB index
- MS1MV2: 95.8% rank-1 identification (ties exact brute-force)
- Compared against FAISS-Flat, FAISS-IVF, HNSWLIB, USearch, ScaNN
