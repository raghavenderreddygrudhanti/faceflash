# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FaceFlash Pipeline                           │
│                                                                     │
│  Image → Detect → Embed (ArcFace) → PCA+ITQ Quantize → Search → Result │
│                                                                     │
│  Registration: ~40ms (detect + embed + quantize + store)            │
│  Search:       ~0.4ms (binary scan + rerank)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Algorithm Detail

### Phase 1: PCA + ITQ Binary Quantization

**Input:** 512-dimensional L2-normalized face embedding x ∈ R^512

**Offline (fit, requires N ≥ 1024 samples):**

Stage 1 — PCA:
1. Collect N sample embeddings X ∈ R^(N×512), center: X_c = X - mean(X)
2. SVD: X_c = U Σ V^T
3. Take top-512 right singular vectors: W_pca = V[:512] ∈ R^(512×512)

Stage 2 — ITQ (Iterative Quantization):
4. Project: V = X_c @ W_pca^T ∈ R^(N×512)
5. Initialize R = I_{512}
6. Repeat 20 iterations (Procrustes):
   - B = sign(V @ R)
   - SVD of B^T @ V = U_r Σ_r V_r^T
   - R = V_r^T @ U_r^T
7. Final projection: W = R^T @ W_pca

Stage 3 — Threshold:
8. t = median(X @ W^T) per bit (ensures balanced marginals)

**Online (encode):**
1. Project: z = x @ W^T - t ∈ R^512
2. Binarize: b = sign(z) ∈ {0, 1}^512
3. Pack: code = packbits(b) ∈ uint8^64

**Output:** 64-byte binary code per face

**Why PCA:** ArcFace embeddings have non-uniform variance across dimensions. PCA aligns the projection with high-variance directions, ensuring the first bits carry the most signal.

**Why ITQ:** After PCA, variance is concentrated in the first few components. Bits corresponding to low-variance components are unreliable — they flip on noise rather than identity. ITQ finds an orthogonal rotation R that spreads variance evenly across all 512 directions, making each bit comparably reliable. This directly improves small-K recall where every bit matters.

**Why median threshold:** Fixes bit bias (ensures ~50% zeros, ~50% ones per bit). This is necessary but not sufficient — it addresses marginal balance, not per-bit reliability. The reliability problem is handled by ITQ.

---

### Phase 2: Binary Search (Hamming Distance)

**Input:** Query code q ∈ uint8^64, database codes D ∈ uint8^(N×64)

**Algorithm:**
```
for each vector d_i in D:
    xor_u64[0..7] = d_i[0..7] ⊕ q[0..7]   # 8 u64 XORs (64 bytes = 8 × u64)
    distance_i = Σ popcount(xor_u64[j])      # 8 POPCNT instructions
```

**Implementation:** Rust casts the 64-byte slices to `&[u64]` (8 elements) and applies XOR + `count_ones()` per u64. On x86_64 this emits the POPCNT instruction; on ARM (Apple Silicon) it emits CNT + addv. This is explicit — not relying on autovectorization of byte-level loops.

**Complexity:** O(N × 8) u64 operations. At 100K: scanning 6.1 MB takes ~0.31ms (fits in L3 cache).

**Output:** N Hamming distances

---

### Phase 3: Candidate Selection

**Input:** N Hamming distances, parameter K (number of candidates)

**Algorithm:**
```
candidates = argpartition(distances, K)[:K]
```

Introselect: O(N) average to find K smallest without full sort.

**Typical K:** 100-300 depending on recall target.

---

### Phase 4: Cosine Reranking

**Input:** K candidate indices, original float embeddings, query embedding

**Algorithm:**
```
for each candidate index i:
    similarity_i = dot(database_float[i], query_float)  // 512 FMAs
result = argmax(similarity)
```

**Complexity:** O(K × 512) FLOPs. At K=100: ~0.04ms.

**Why rerank:** Binarization is lossy (512 floats → 512 bits = 32x compression). The binary scan identifies a small candidate set likely to contain the true nearest neighbor. Exact cosine on this small set recovers the accuracy lost to quantization. This two-phase approach gives near-exact recall at a fraction of the cost of scanning all float vectors.

---

## Computational Cost Breakdown (100K faces)

| Phase | Operation | Time | Data Touched |
|---|---|---|---|
| Binary scan | 8 XOR + 8 POPCNT per vector, N vectors | 0.31ms | 6.1 MB (binary) |
| Top-K selection | argpartition | 0.01ms | N integers |
| Cosine rerank | K dot products (512-d) | 0.04ms | K × 2KB (float) |
| **Total** | | **0.36ms** | |

---

## Memory Layout

### Per-face storage:

| Component | Size | Purpose |
|---|---|---|
| Binary code | 64 bytes | Search index (scanned every query) |
| Float vector | 2048 bytes | Reranking (accessed only for K candidates per query) |
| Label | ~20 bytes | Identity string |

### At scale:

| Faces | Binary Index | Float Storage | Total |
|---|---|---|---|
| 10K | 0.6 MB | 20 MB | 20.6 MB |
| 100K | 6.1 MB | 195 MB | 201 MB |
| 1M | 61 MB | 1.95 GB | 2.01 GB |

The binary index is what gets scanned every query. At 100K it fits in L3 cache. Float storage is randomly accessed for K vectors only — not scanned.

---

## Index Lifecycle

### Build:
1. Accumulate embeddings (add one-by-one or batch)
2. Before 1024 vectors: random projection fallback (no PCA, lower quality)
3. At 1024 vectors: auto-fit PCA+ITQ on accumulated data, re-encode all
4. After fit: new vectors encoded immediately with learned projection

### Search:
1. Encode query with W (PCA+ITQ folded projection)
2. Hamming scan entire binary index (Rust POPCNT)
3. Select top-K candidates (argpartition)
4. Rerank with float cosine
5. Return sorted results

### Persistence:
- `save(path)`: writes binary codes, float vectors, projection W, thresholds, metadata
- `load(path)`: loads all components

---

## Design Decisions

| Decision | Rationale |
|---|---|
| PCA + ITQ over random projection | PCA captures variance structure; ITQ equalizes per-bit reliability |
| 512 bits (64 bytes) | All 512 PCA directions retained (no dimensional truncation). Binarization is lossy — Phase 4 recovers this. |
| Brute-force binary scan | Simpler than graph index, fits in cache at 100K–1M scale |
| Cosine rerank | Recovers quantization error for <0.1ms cost |
| Rust with u64 POPCNT | Explicit hardware instruction, not relying on autovectorization |
| Auto-fit at N≥1024 | Need N > n_bits for full-rank SVD (512 bits requires >512 samples) |
| Median threshold | Fixes bit bias (balanced marginals). Reliability addressed by ITQ. |

---

## Limitations

1. **Linear scan:** O(N) binary search. At 10M+ faces, latency exceeds 30ms. Would need partitioning (IVF-style) for billion scale.
2. **Float storage:** Total memory is 2KB/face (binary + float). The 32x compression applies to the search index only — float vectors are still needed for reranking.
3. **Cold start:** Requires ≥1024 representative samples for PCA+ITQ fit. Below 1024, falls back to random projection (lower recall).
4. **Static projection:** Once fitted, W doesn't adapt to new data distribution. Re-fit needed if embedding model changes or data distribution shifts significantly.
5. **Bit reliability imbalance:** Even with ITQ, the last PCA directions carry less discriminative signal. At very low K (top-10), this manifests as reduced recall. Increasing K cheaply recovers accuracy.
