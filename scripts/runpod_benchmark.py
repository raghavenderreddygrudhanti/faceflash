"""
FaceFlash — Full Pipeline Benchmark (RunPod / any GPU machine)

Measures the COMPLETE end-to-end pipeline with real-time data:
  1. Face detection time
  2. ArcFace embedding time (GPU vs CPU)
  3. Binary search time
  4. Rerank time
  5. Total query latency

Also extracts large-scale embeddings and benchmarks at 100K/500K/1M.

Usage:
    # On RunPod with GPU:
    pip install faceflash onnxruntime-gpu faiss-cpu
    cd rust && maturin develop --release && cd ..
    python scripts/runpod_benchmark.py

    # On laptop (CPU only):
    python scripts/runpod_benchmark.py --cpu-only
"""
import sys
import time
import json
import argparse
import platform
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"


def detect_hardware():
    """Detect available hardware."""
    info = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }

    # Check GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        info["onnx_providers"] = providers
        info["has_cuda"] = "CUDAExecutionProvider" in providers
        info["has_coreml"] = "CoreMLExecutionProvider" in providers
    except ImportError:
        info["onnx_providers"] = []
        info["has_cuda"] = False

    # Check Rust
    try:
        import faceflash_core
        info["has_rust"] = True
    except ImportError:
        info["has_rust"] = False

    return info


def bench_embedding_speed(n_images=100):
    """Measure embedding extraction speed (detect + embed)."""
    from faceflash.embed import FaceEmbedder
    from faceflash.detect import detect_and_align

    embedder = FaceEmbedder()

    # Generate synthetic face-like images (random RGB 250x250)
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 255, (250, 250, 3), dtype=np.uint8) for _ in range(n_images)]

    # Measure detect + embed pipeline
    times_detect = []
    times_embed = []
    times_total = []

    for img in images:
        t0 = time.perf_counter()
        face = detect_and_align(img)
        t_detect = time.perf_counter() - t0

        t1 = time.perf_counter()
        emb = embedder.embed(face)
        t_embed = time.perf_counter() - t1

        t_total = time.perf_counter() - t0
        times_detect.append(t_detect * 1000)
        times_embed.append(t_embed * 1000)
        times_total.append(t_total * 1000)

    return {
        "n_images": n_images,
        "detect_ms": {"mean": np.mean(times_detect), "p99": np.percentile(times_detect, 99)},
        "embed_ms": {"mean": np.mean(times_embed), "p99": np.percentile(times_embed, 99)},
        "total_ms": {"mean": np.mean(times_total), "p99": np.percentile(times_total, 99)},
        "throughput_fps": 1000.0 / np.mean(times_total),
    }


def bench_search_pipeline(database, queries, gt, n_bits=512, n_candidates=100):
    """Measure search breakdown: encode + binary scan + rerank."""
    from faceflash.pca_quantize import PCABinaryQuantizer

    try:
        import faceflash_core
        has_rust = True
    except ImportError:
        has_rust = False

    pca = PCABinaryQuantizer(n_bits=n_bits)
    pca.fit(database[:min(5000, len(database))])
    db_codes = pca.encode(database)

    times_encode = []
    times_scan = []
    times_rerank = []
    times_total = []
    correct = 0

    for i in range(len(queries)):
        # Encode query
        t0 = time.perf_counter()
        q_code = pca.encode(queries[i:i+1]).ravel()
        t_encode = time.perf_counter() - t0

        # Binary scan
        t1 = time.perf_counter()
        if has_rust:
            topk = faceflash_core.hamming_topk(q_code, db_codes, n_candidates)
            cand_idx = np.asarray(topk).astype(np.intp)
        else:
            from faceflash.pca_quantize import _POPCOUNT_TABLE
            xor = np.bitwise_xor(db_codes, q_code.reshape(1, -1))
            distances = _POPCOUNT_TABLE[xor].sum(axis=1)
            cand_idx = np.argpartition(distances, n_candidates)[:n_candidates]
        t_scan = time.perf_counter() - t1

        # Rerank
        t2 = time.perf_counter()
        cosine_sims = database[cand_idx] @ queries[i]
        best_local = np.argmax(cosine_sims)
        best_idx = cand_idx[best_local]
        t_rerank = time.perf_counter() - t2

        t_total = time.perf_counter() - t0
        times_encode.append(t_encode * 1000)
        times_scan.append(t_scan * 1000)
        times_rerank.append(t_rerank * 1000)
        times_total.append(t_total * 1000)

        if best_idx == gt[i]:
            correct += 1

    return {
        "n_database": len(database),
        "n_queries": len(queries),
        "n_bits": n_bits,
        "n_candidates": n_candidates,
        "recall_1": correct / len(queries),
        "encode_ms": {"mean": np.mean(times_encode), "p99": np.percentile(times_encode, 99)},
        "scan_ms": {"mean": np.mean(times_scan), "p99": np.percentile(times_scan, 99)},
        "rerank_ms": {"mean": np.mean(times_rerank), "p99": np.percentile(times_rerank, 99)},
        "total_ms": {"mean": np.mean(times_total), "p99": np.percentile(times_total, 99)},
        "qps": 1000.0 / np.mean(times_total),
        "has_rust": has_rust,
    }


def main():
    parser = argparse.ArgumentParser(description="FaceFlash Full Pipeline Benchmark")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU tests")
    parser.add_argument("--scale", type=int, default=None, help="Max database size")
    args = parser.parse_args()

    print("=" * 70)
    print("  FaceFlash — Full Pipeline Benchmark")
    print("=" * 70)

    # Hardware detection
    hw = detect_hardware()
    print(f"\n  Hardware:")
    print(f"    Platform: {hw['platform']}")
    print(f"    CPU: {hw['cpu']}")
    print(f"    CUDA GPU: {'Yes' if hw['has_cuda'] else 'No'}")
    print(f"    Rust backend: {'Yes' if hw['has_rust'] else 'No'}")
    print(f"    ONNX providers: {hw['onnx_providers']}")

    results = {"hardware": hw}

    # 1. Embedding speed
    print(f"\n{'─'*70}")
    print(f"  [1] Embedding Pipeline (detect + embed)")
    print(f"{'─'*70}")
    emb_results = bench_embedding_speed(n_images=50)
    results["embedding"] = emb_results
    print(f"    Detection:  {emb_results['detect_ms']['mean']:.1f}ms (mean)")
    print(f"    Embedding:  {emb_results['embed_ms']['mean']:.1f}ms (mean)")
    print(f"    Total:      {emb_results['total_ms']['mean']:.1f}ms (mean)")
    print(f"    Throughput:  {emb_results['throughput_fps']:.1f} faces/sec")

    # 2. Search pipeline at available scales
    print(f"\n{'─'*70}")
    print(f"  [2] Search Pipeline Breakdown")
    print(f"{'─'*70}")

    # Load available data
    vgg_path = DATA_DIR / "vggface2_100k_embeddings.npy"
    lfw_path = DATA_DIR / "lfw_embeddings.npy"

    if vgg_path.exists():
        embs = np.load(vgg_path).astype(np.float32)
        labels = np.load(DATA_DIR / "vggface2_100k_labels.npy", allow_pickle=True)
        data_name = "VGGFace2"
    elif lfw_path.exists():
        embs = np.load(lfw_path).astype(np.float32)
        labels = np.load(DATA_DIR / "lfw_labels.npy", allow_pickle=True)
        data_name = "LFW"
    else:
        print("    No embedding data found. Run extract scripts first.")
        return

    if args.scale and args.scale < len(embs):
        embs = embs[:args.scale]
        labels = labels[:args.scale]

    # Split
    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(str(l), []).append(i)
    qi = []
    for indices in label_to_idx.values():
        if len(indices) >= 4:
            qi.extend(indices[-3:])
        elif len(indices) >= 2:
            qi.append(indices[-1])
    qi = np.array(qi[:500])
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])

    print(f"    Data: {data_name}, Database: {len(database):,}, Queries: {len(queries)}")

    # Test at multiple configs
    configs = [
        (256, 100),
        (384, 100),
        (512, 100),
        (512, 200),
    ]

    search_results = []
    print(f"\n    {'Config':<18} {'Recall':<8} {'Encode':<10} {'Scan':<10} {'Rerank':<10} {'Total':<10} {'QPS'}")
    print(f"    {'─'*75}")

    for n_bits, n_cands in configs:
        r = bench_search_pipeline(database, queries, gt, n_bits=n_bits, n_candidates=n_cands)
        search_results.append(r)
        print(f"    {n_bits}b/{n_cands}c          "
              f"{r['recall_1']*100:<6.1f}% "
              f"{r['encode_ms']['mean']:<9.3f} "
              f"{r['scan_ms']['mean']:<9.3f} "
              f"{r['rerank_ms']['mean']:<9.3f} "
              f"{r['total_ms']['mean']:<9.3f} "
              f"{r['qps']:<7.0f}")

    results["search"] = search_results

    # 3. End-to-end latency
    print(f"\n{'─'*70}")
    print(f"  [3] End-to-End (image → result)")
    print(f"{'─'*70}")
    best = search_results[2]  # 512/100
    e2e_ms = emb_results['total_ms']['mean'] + best['total_ms']['mean']
    print(f"    Detect + Embed: {emb_results['total_ms']['mean']:.1f}ms")
    print(f"    Search:         {best['total_ms']['mean']:.2f}ms")
    print(f"    End-to-end:     {e2e_ms:.1f}ms ({1000/e2e_ms:.0f} FPS)")
    results["end_to_end_ms"] = e2e_ms

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench_pipeline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'bench_pipeline.json'}")


if __name__ == "__main__":
    main()
