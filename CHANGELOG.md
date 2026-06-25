# Changelog

## 0.1.0 (2026-06-25)

Initial public release.

### Features
- PCA+ITQ binary quantization for face embeddings (128/256/384/512 bits)
- Rust POPCNT backend (50x faster than NumPy)
- Two-phase search: Hamming filter → cosine rerank
- High-level API: `register()`, `search()`, `verify()`, `register_folder()`
- Buffered O(n) `add()` with auto-flush
- `save()`/`load()` with mmap'd float vectors (resident memory = binary index only)
- Configurable `n_bits` and `n_candidates` knobs

### Benchmarks
- MS1MV2: 99.9% recall@1 at 100K (13,724 identities), 6.1 MB index
- MS1MV2: 95.8% rank-1 identification (ties exact brute-force)
- 500K faces: 99.9% recall, 30 MB vs HNSW's 1,465 MB (48x less memory)
- Compared against FAISS-Flat, FAISS-IVF, HNSWLIB, USearch, ScaNN

### Known Limitations
- Linear scan (not sub-linear) — latency scales with database size
- Rust backend must be built manually (not in pip wheel yet)
- Built-in face detection is basic (Haar cascade) — pass pre-aligned faces for best results
