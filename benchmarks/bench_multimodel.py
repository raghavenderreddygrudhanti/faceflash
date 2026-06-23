"""
FaceFlash — Multi-Model & HNSWLIB Benchmark

Tests FaceFlash with different embedding models to prove it's model-agnostic.
Also compares against HNSWLIB (the most popular HNSW library).

Models tested:
  1. ArcFace R50 (w600k, 166MB) — current default
  2. Buffalo_L R50 (166MB) — InsightFace large
  3. Buffalo_S MobileFaceNet (13MB) — lightweight mobile model

Usage:
    .venv/bin/python benchmarks/bench_multimodel.py
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss
import hnswlib

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS_DIR = Path.home() / ".faceflash" / "models"
LFW_DIR = DATA_DIR / "lfw_funneled"

MODELS = {
    "ArcFace-R50 (w600k)": MODELS_DIR / "arcface_r100.onnx",
    "Buffalo-L (R50)": MODELS_DIR / "buffalo_l_r50.onnx",
    "Buffalo-S (MobileFaceNet)": MODELS_DIR / "buffalo_s_mbf.onnx",
    "FaceNet (InceptionResNet)": MODELS_DIR / "facenet.onnx",
}


class GenericEmbedder:
    """Load any ONNX face model and extract embeddings."""
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_size = shape[2] if len(shape) == 4 else 112

    def embed(self, face_img):
        img = Image.fromarray(face_img).resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = (arr - 127.5) / 127.5
        arr = arr.transpose(2, 0, 1)[np.newaxis]
        out = self.session.run(None, {self.input_name: arr})[0][0]
        return out / np.linalg.norm(out)


def extract_embeddings(model_name, model_path, images, max_n=3000):
    """Extract embeddings for a list of (image_path, label) pairs."""
    embedder = GenericEmbedder(model_path)
    embeddings = []
    labels = []
    for img_path, label in tqdm(images[:max_n], desc=f"  {model_name}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            mh, mw = int(h * 0.15), int(w * 0.15)
            face = img_np[mh:h-mh, mw:w-mw]
            emb = embedder.embed(face)
            embeddings.append(emb)
            labels.append(label)
        except Exception:
            pass
    dim = embeddings[0].shape[0] if embeddings else 512
    return np.array(embeddings, dtype=np.float32), np.array(labels), dim


def bench_hnswlib(database, queries, gt_top1, ef_values=[32, 64, 128]):
    """HNSWLIB benchmark at multiple ef_search values."""
    dim = database.shape[1]
    n = len(database)

    # Build
    index = hnswlib.Index(space='ip', dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=32)

    t0 = time.perf_counter()
    index.add_items(database, np.arange(n))
    build_time = time.perf_counter() - t0

    results = []
    for ef in ef_values:
        index.set_ef(ef)
        # Warmup
        index.knn_query(queries[:3], k=1)

        times = []
        correct = 0
        for i in range(len(queries)):
            t0 = time.perf_counter()
            labels_out, distances = index.knn_query(queries[i:i+1], k=1)
            times.append((time.perf_counter() - t0) * 1000)
            if labels_out[0][0] == gt_top1[i]:
                correct += 1

        results.append({
            "method": f"HNSWLIB (ef={ef})",
            "recall_1": correct / len(queries),
            "latency_mean_ms": np.mean(times),
            "latency_p99_ms": np.percentile(times, 99),
            "memory_mb": database.nbytes / 1024**2 * 1.5,
            "build_time_s": build_time,
        })

    return results


def bench_faceflash_on_embeddings(database, queries, gt_top1, n_cands=[50, 100, 300]):
    """FaceFlash benchmark given pre-computed embeddings."""
    pca = PCABinaryQuantizer(n_bits=min(512, database.shape[1]))
    pca.fit(database[:min(3000, len(database))])
    db_codes = pca.encode(database)
    q_codes = pca.encode(queries)

    results = []
    for n_cand in n_cands:
        times = []
        correct = 0
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cand)
            cand_idx = np.asarray(topk).astype(np.intp)
            cosine_sims = database[cand_idx] @ queries[i]
            best_idx = cand_idx[np.argmax(cosine_sims)]
            times.append((time.perf_counter() - t0) * 1000)
            if best_idx == gt_top1[i]:
                correct += 1

        results.append({
            "method": f"FaceFlash (top-{n_cand})",
            "recall_1": correct / len(queries),
            "latency_mean_ms": np.mean(times),
            "latency_p99_ms": np.percentile(times, 99),
            "memory_mb": db_codes.nbytes / 1024 / 1024,
        })
    return results


def main():
    print("\n" + "=" * 70)
    print("  FaceFlash — Multi-Model & HNSWLIB Benchmark")
    print("=" * 70)
    print(f"  Backend: {backend_info()}")

    # Collect LFW images
    if not LFW_DIR.exists():
        print("  LFW not found. Using pre-extracted embeddings for HNSWLIB test only.")
        run_hnswlib_only()
        return

    persons = sorted([d for d in LFW_DIR.iterdir() if d.is_dir()])
    all_images = []
    for person_dir in persons:
        name = person_dir.name
        for img_path in sorted(person_dir.glob("*.jpg")):
            all_images.append((img_path, name))
    print(f"  LFW images: {len(all_images)}")

    # Use 3000 images for multi-model test (fast enough)
    n_test = 3000
    print(f"  Using {n_test} images for multi-model comparison")

    all_results = {}

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\n  {model_name}: MODEL NOT FOUND, skipping")
            continue

        print(f"\n  {'─'*60}")
        print(f"  Model: {model_name}")
        print(f"  {'─'*60}")

        # Extract
        embeddings, labels, dim = extract_embeddings(model_name, model_path, all_images, max_n=n_test)
        print(f"  Extracted: {len(embeddings)} embeddings, dim={dim}")

        # Split
        label_to_idx = {}
        for i, l in enumerate(labels):
            label_to_idx.setdefault(l, []).append(i)
        qi = [indices[-1] for indices in label_to_idx.values() if len(indices) >= 2][:200]
        qi = np.array(qi)
        mask = np.ones(len(embeddings), dtype=bool)
        mask[qi] = False
        database = embeddings[np.where(mask)[0]]
        queries = embeddings[qi]

        # Ground truth
        gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])
        print(f"  Database: {len(database)}, Queries: {len(queries)}")

        # FaceFlash
        ff_results = bench_faceflash_on_embeddings(database, queries, gt)

        # HNSWLIB
        hnsw_results = bench_hnswlib(database, queries, gt)

        # FAISS-Flat
        faiss.omp_set_num_threads(1)
        idx = faiss.IndexFlatIP(dim)
        idx.add(database)
        times = []
        for q in queries:
            t0 = time.perf_counter()
            idx.search(q.reshape(1, -1), 1)
            times.append((time.perf_counter() - t0) * 1000)
        D, I = idx.search(queries, 1)
        faiss_result = {
            "method": "FAISS-Flat",
            "recall_1": float(np.mean(I[:, 0] == gt)),
            "latency_mean_ms": np.mean(times),
            "latency_p99_ms": np.percentile(times, 99),
            "memory_mb": database.nbytes / 1024**2,
        }

        # Print
        model_results = [faiss_result] + hnsw_results + ff_results
        print(f"\n  {'Method':<25} {'R@1':<8} {'Mean ms':<10} {'P99 ms':<10} {'Memory':<8}")
        print(f"  {'─'*60}")
        for r in model_results:
            print(f"  {r['method']:<25} {r['recall_1']*100:<6.1f}% "
                  f"{r['latency_mean_ms']:<10.3f}{r['latency_p99_ms']:<10.3f}"
                  f"{r['memory_mb']:<6.1f}MB")

        all_results[model_name] = model_results

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bench_multimodel.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


def run_hnswlib_only():
    """Run HNSWLIB test using pre-extracted LFW embeddings."""
    embs = np.load(DATA_DIR / "lfw_embeddings.npy").astype(np.float32)
    labels = np.load(DATA_DIR / "lfw_labels.npy", allow_pickle=True)

    label_to_idx = {}
    for i, l in enumerate(labels):
        label_to_idx.setdefault(l, []).append(i)
    qi = [indices[-1] for indices in label_to_idx.values() if len(indices) >= 2][:300]
    qi = np.array(qi)
    mask = np.ones(len(embs), dtype=bool)
    mask[qi] = False
    database = embs[np.where(mask)[0]]
    queries = embs[qi]
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])

    print(f"\n  HNSWLIB vs FaceFlash (LFW, {len(database)} faces)")
    hnsw_results = bench_hnswlib(database, queries, gt)
    ff_results = bench_faceflash_on_embeddings(database, queries, gt)

    print(f"\n  {'Method':<25} {'R@1':<8} {'Mean ms':<10} {'Memory':<8}")
    print(f"  {'─'*50}")
    for r in hnsw_results + ff_results:
        print(f"  {r['method']:<25} {r['recall_1']*100:<6.1f}% "
              f"{r['latency_mean_ms']:<10.3f}{r['memory_mb']:<6.1f}MB")


if __name__ == "__main__":
    main()
