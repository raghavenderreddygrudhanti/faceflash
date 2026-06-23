# Benchmarks

## Datasets

| Dataset | Faces | Identities | Source |
|---|---|---|---|
| LFW | 13,233 | 5,749 | Real (ArcFace extracted) |
| VGGFace2 | 100,000 | 291 | Real (ArcFace extracted, streamed from HuggingFace) |

## Methodology

- Single-threaded (FAISS `omp_set_num_threads(1)`)
- Same hardware: Apple M2 Pro
- Timing: `time.perf_counter()` per query
- Warmup: 5 queries excluded
- Ground truth: exact brute-force cosine argmax

## Benchmark Scripts

| Script | What it measures |
|---|---|
| `bench_accuracy.py` | R@1, R@10, R@100, QPS on LFW + VGGFace2 |
| `bench_vector_search.py` | vs FAISS, HNSWLIB, USearch at 100K |
| `bench_additional.py` | vs Annoy, NMSLIB |
| `bench_face_libraries.py` | vs DeepFace, InsightFace retrieval |
| `bench_multimodel.py` | 4 embedding models + HNSWLIB |
| `bench_vggface2.py` | VGGFace2 candidate sweep |
| `bench_timing.py` | LFW timing breakdown (binary scan vs rerank) |
| `bench_scalability.py` | 13K to 1M latency scaling |

## Reproducing

```bash
./run_benchmarks.sh
```

Or individually:
```bash
.venv/bin/python benchmarks/bench_accuracy.py
.venv/bin/python benchmarks/bench_vector_search.py
```
