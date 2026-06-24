#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — RunPod Complete Pipeline
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT THIS DOES:
#   Step 1: Install all dependencies (Python, Rust, ONNX with GPU)
#   Step 2: Build the Rust POPCNT backend (faceflash_core)
#   Step 3: Download ArcFace model (~166MB) for face embedding
#   Step 4: Stream VGGFace2 from HuggingFace and extract 1M face embeddings
#   Step 5: Run benchmarks at 100K, 500K, 1M scale
#   Step 6: Generate publication figures (PNG)
#   Step 7: Push all results + figures to GitHub
#
# REQUIREMENTS:
#   - RunPod with GPU (A100/4090/3090 recommended)
#   - 50GB+ disk space
#   - ~60-90 minutes total runtime
#
# HOW TO USE:
#   1. Open a terminal on RunPod
#   2. Paste this entire script
#   3. Wait for completion
#   4. Results appear in results/ and docs/figures/
#
# ═══════════════════════════════════════════════════════════════════════════
set -e

LOG_FILE="/workspace/faceflash_pipeline.log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# RunPod is EPHEMERAL — results only survive if they reach GitHub.
# Set GITHUB_TOKEN before running so the push works automatically:
#   export GITHUB_TOKEN=ghp_xxx ; bash scripts/runpod_full.sh
REMOTE_SLUG="***REMOVED***rudhanti/faceflash"
RUN_TS="$(date +%Y%m%d_%H%M%S)"   # one timestamp for the whole run (no overwrites)
export RUN_TS

commit_and_push() {
    # commit_and_push "<message>" <paths...>
    local msg="$1"; shift
    git config user.email "raghavenderreddy1212@gmail.com"
    git config user.name "Raghavender Grudhanti"
    git add "$@" 2>/dev/null || true
    if git diff --cached --quiet 2>/dev/null; then
        log "  (nothing new to commit)"
        return 0
    fi
    git commit -q -m "$msg" || { log "  ⚠ commit failed"; return 1; }
    if [ -n "$GITHUB_TOKEN" ]; then
        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REMOTE_SLUG}.git"
    fi
    if git push origin HEAD:main; then
        log "  ✓ Pushed to GitHub — results are safe even after this pod is destroyed"
        return 0
    fi
    log "  ✗✗✗ PUSH FAILED — results are committed LOCALLY but NOT on GitHub."
    log "      RunPod is ephemeral: stopping the pod now will LOSE these results."
    log "      Fix: export GITHUB_TOKEN=<token> and re-run, or copy the JSON"
    log "      printed at the end of this log before stopping the pod."
    return 1
}

log "═══════════════════════════════════════════════════════════════"
log "  FaceFlash — RunPod Complete Pipeline"
log "═══════════════════════════════════════════════════════════════"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: CLONE REPO + INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 1/7: Installing dependencies                           │"
log "└─────────────────────────────────────────────────────────────┘"
log ""

cd /workspace
if [ ! -d "faceflash" ]; then
    log "  Cloning repository from GitHub..."
    git clone https://github.com/***REMOVED***rudhanti/faceflash.git
    log "  ✓ Repository cloned"
else
    log "  Repository already exists, pulling latest..."
    cd faceflash && git pull && cd ..
    log "  ✓ Repository updated"
fi
cd faceflash

log "  Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate
log "  Installing Python packages..."
log "    - numpy, pillow, tqdm (core)"
log "    - onnxruntime-gpu (GPU-accelerated face embedding)"
log "    - faiss-cpu (baseline comparison)"
log "    - datasets, huggingface-hub (data streaming)"
log "    - maturin (Rust→Python bridge)"
log "    - matplotlib (figures)"
pip install -q numpy onnxruntime-gpu pillow tqdm faiss-cpu datasets huggingface-hub maturin matplotlib 2>&1 | tail -3
log "  ✓ All Python packages installed"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD RUST BACKEND
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 2/7: Building Rust POPCNT backend                      │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
log "  This compiles faceflash_core (Hamming distance via hardware POPCNT)."
log "  It makes binary search ~50x faster than NumPy."
log ""

if python -c "import faceflash_core" 2>/dev/null; then
    log "  ✓ Rust backend already built"
else
    log "  Compiling Rust code (takes ~60 seconds)..."
    # Install Rust if needed
    if ! command -v cargo &>/dev/null; then
        log "  Installing Rust toolchain..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    cd rust
    maturin develop --release 2>&1 | tail -5
    cd ..
    log "  ✓ Rust backend compiled and installed"
fi

python -c "import faceflash_core; print('  Verification: faceflash_core loaded successfully')"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: DOWNLOAD ARCFACE MODEL
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 3/7: Downloading ArcFace face embedding model          │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
log "  Model: InsightFace buffalo_l (ResNet50, trained on WebFace600K)"
log "  Output: 512-dimensional L2-normalized face embedding"
log "  Size: ~166MB"
log "  Source: HuggingFace (public-data/insightface)"
log ""

MODEL_DIR="$HOME/.faceflash/models"
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/arcface_r100.onnx" ]; then
    log "  ✓ Model already downloaded ($(du -h $MODEL_DIR/arcface_r100.onnx | cut -f1))"
else
    log "  Downloading..."
    curl -L --progress-bar -o "$MODEL_DIR/arcface_r100.onnx" \
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
    log "  ✓ Model downloaded ($(du -h $MODEL_DIR/arcface_r100.onnx | cut -f1))"
fi
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: EXTRACT 1M FACE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 4/7: Extracting 1M face embeddings from VGGFace2       │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
log "  Data source: VGGFace2 (3.3M images, 9,131 identities)"
log "  Streamed from: HuggingFace (logasja/VGGFace2, train split)"
log "  Target: 1,000,000 embeddings"
log "  Method: ArcFace ONNX inference (GPU if available)"
log "  Output: data/vggface2_1m_embeddings.npy + data/vggface2_1m_labels.npy"
log ""
log "  Each image goes through:"
log "    1. Decode JPEG from HuggingFace stream"
log "    2. Mild center-crop (remove 5% border)"
log "    3. Resize to 112x112"
log "    4. Normalize to [-1, 1]"
log "    5. ArcFace inference → 512-dim vector"
log "    6. L2 normalize → unit sphere"
log ""

mkdir -p data

python << 'EXTRACT_EOF'
import sys, time, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from faceflash.embed import FaceEmbedder

DATA_DIR = Path('data')
OUT = DATA_DIR / 'vggface2_1m_embeddings.npy'
OUT_L = DATA_DIR / 'vggface2_1m_labels.npy'
CHECKPOINT = DATA_DIR / 'vggface2_1m_checkpoint.npz'

TARGET = 1000000

if OUT.exists():
    embs = np.load(OUT)
    print(f'  ✓ Already extracted: {embs.shape[0]:,} embeddings')
    sys.exit(0)

# Resume from checkpoint if available
start_count = 0
embeddings_list = []
labels_list = []
if CHECKPOINT.exists():
    data = np.load(CHECKPOINT, allow_pickle=True)
    embeddings_list = list(data['embeddings'])
    labels_list = list(data['labels'])
    start_count = len(embeddings_list)
    print(f'  Resuming from checkpoint: {start_count:,} embeddings already done')

from datasets import load_dataset
print(f'  Connecting to HuggingFace (logasja/VGGFace2, test split)...')
ds = load_dataset('logasja/VGGFace2', split='train', streaming=True)

print(f'  Loading ArcFace model...')
embedder = FaceEmbedder()

# Check GPU
import onnxruntime as ort
provider = embedder.session.get_providers()[0]
print(f'  ONNX provider: {provider}')
if 'CUDA' in provider:
    print(f'  ★ GPU acceleration active')
else:
    print(f'  ⚠ Running on CPU (slower, ~3 img/s vs ~1000 img/s on GPU)')

# Fail fast if the label field is missing — otherwise every image gets a unique
# label and the recall benchmark silently breaks (no identity has >=2 images).
probe = next(iter(ds))
LABEL_FIELD = next((c for c in ('class_id', 'label', 'identity', 'person_id', 'id')
                    if c in probe), None)
if LABEL_FIELD is None:
    print(f'  ERROR: no known label field in dataset. Available keys: {list(probe.keys())}')
    print(f'  Edit the script to use the correct field, then re-run.')
    sys.exit(1)
if 'image' not in probe:
    print(f'  ERROR: no "image" field. Available keys: {list(probe.keys())}')
    sys.exit(1)
print(f'  Label field: {LABEL_FIELD}')

failed = 0
t_start = time.perf_counter()

for i, sample in enumerate(tqdm(ds, total=TARGET, initial=start_count, desc='  Extracting')):
    if i < start_count:
        continue
    if len(embeddings_list) >= TARGET:
        break
    try:
        img = sample['image'].convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        mh, mw = int(h * 0.05), int(w * 0.05)
        face = img_np[mh:h-mh, mw:w-mw]
        emb = embedder.embed(face)
        embeddings_list.append(emb)
        labels_list.append(sample.get(LABEL_FIELD, i))
    except Exception:
        failed += 1

    # Checkpoint every 50K
    if len(embeddings_list) % 50000 == 0 and len(embeddings_list) > start_count:
        elapsed = time.perf_counter() - t_start
        rate = (len(embeddings_list) - start_count) / elapsed
        remaining = (TARGET - len(embeddings_list)) / max(rate, 1) / 60
        print(f'\n  Checkpoint: {len(embeddings_list):,} done | '
              f'{rate:.0f} img/s | ETA {remaining:.0f} min | Failed: {failed}')
        np.savez(CHECKPOINT,
                 embeddings=np.array(embeddings_list, dtype=np.float32),
                 labels=np.array(labels_list))

# Save final
embeddings = np.array(embeddings_list, dtype=np.float32)
labels = np.array(labels_list)

# Remove zero-norm (failed extractions that slipped through)
valid = np.linalg.norm(embeddings, axis=1) > 0.5
embeddings = embeddings[valid]
labels = labels[valid]

elapsed = time.perf_counter() - t_start
n_ids = len(np.unique(labels))
print(f'\n  ═══════════════════════════════════════════════════')
print(f'  EXTRACTION COMPLETE')
print(f'  Embeddings: {len(embeddings):,}')
print(f'  Identities: {n_ids:,}')
print(f'  Failed: {failed}')
print(f'  Time: {elapsed/60:.1f} min ({len(embeddings)/elapsed:.0f} img/s)')
print(f'  ═══════════════════════════════════════════════════')

np.save(OUT, embeddings)
np.save(OUT_L, labels)
print(f'  ✓ Saved: {OUT} ({embeddings.nbytes/1024/1024:.0f} MB)')
print(f'  ✓ Saved: {OUT_L}')

# Clean checkpoint
if CHECKPOINT.exists():
    CHECKPOINT.unlink()
EXTRACT_EOF

log ""
log "  ✓ Embedding extraction complete"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: RUN BENCHMARKS AT MULTIPLE SCALES
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 5/7: Running benchmarks (100K / 500K / 1M)             │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
log "  For each scale, measures:"
log "    - FAISS-Flat (exact brute force, baseline)"
log "    - FaceFlash at 256/384/512 bits × 100/200/300 candidates"
log "    - Recall@1, latency (mean/P99), QPS, memory"
log ""
log "  All single-threaded for fair comparison."
log ""

python << 'BENCH_EOF'
import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss

print(f'  Backend: {backend_info()}')
faiss.omp_set_num_threads(1)
print(f'  FAISS threads: 1 (single-threaded for fair comparison)')

DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Load best available data
for name in ['vggface2_1m', 'vggface2_500k', 'vggface2_100k', 'vggface2']:
    p = DATA_DIR / f'{name}_embeddings.npy'
    if p.exists():
        embs = np.load(p).astype(np.float32)
        labels = np.load(DATA_DIR / f'{name}_labels.npy', allow_pickle=True)
        print(f'  Loaded: {p.name} ({embs.shape[0]:,} vectors, {len(np.unique(labels)):,} identities)')
        break
else:
    print('  ERROR: No embedding data found!')
    sys.exit(1)

# Split into database + queries (hold out last 3 per identity)
label_to_idx = {}
for i, l in enumerate(labels):
    label_to_idx.setdefault(str(l), []).append(i)
qi = []
for indices in label_to_idx.values():
    if len(indices) >= 4:
        qi.extend(indices[-3:])
    elif len(indices) >= 2:
        qi.append(indices[-1])
qi = np.array(qi[:1000])  # Up to 1000 queries for tight CIs
mask = np.ones(len(embs), dtype=bool)
mask[qi] = False
all_db = embs[np.where(mask)[0]]
queries = embs[qi]
print(f'  Total available: DB={len(all_db):,}, Queries={len(queries)}')

# Benchmark at multiple scales
scales = []
if len(all_db) >= 100000: scales.append(('100K', 100000))
if len(all_db) >= 500000: scales.append(('500K', 500000))
if len(all_db) >= 1000000: scales.append(('1M', 1000000))
if not scales: scales.append((f'{len(all_db)//1000}K', len(all_db)))

configs = [
    (256, 100), (256, 300),
    (384, 100), (384, 300),
    (512, 100), (512, 200), (512, 300),
]

all_results = {}

for scale_name, n_db in scales:
    print(f'\n  {"═"*60}')
    print(f'  SCALE: {scale_name} ({n_db:,} faces)')
    print(f'  {"═"*60}')

    database = all_db[:n_db]

    # Ground truth
    print(f'  Computing ground truth (brute force cosine)...')
    gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])

    # FAISS-Flat baseline
    idx = faiss.IndexFlatIP(512)
    idx.add(database)
    times_f = []
    for i in range(len(queries)):
        t0 = time.perf_counter()
        idx.search(queries[i:i+1], 1)
        times_f.append((time.perf_counter() - t0) * 1000)
    faiss_ms = np.mean(times_f)
    print(f'  FAISS-Flat: {faiss_ms:.2f}ms | Memory: {database.nbytes/1024/1024:.0f} MB')

    scale_results = [{'method': 'FAISS-Flat', 'recall_1': 1.0, 'latency_ms': faiss_ms,
                      'memory_mb': database.nbytes/1024/1024}]

    print(f'\n  {"Config":<16} {"Recall":<8} {"Latency":<10} {"Speedup":<10} {"Memory":<10} {"QPS"}')
    print(f'  {"─"*65}')

    for n_bits, n_cands in configs:
        pca = PCABinaryQuantizer(n_bits=n_bits)
        pca.fit(database[:5000])
        db_codes = pca.encode(database)
        q_codes = pca.encode(queries)

        # Warmup
        for i in range(5):
            faceflash_core.hamming_topk(q_codes[i], db_codes, n_cands)

        times, correct = [], 0
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(q_codes[i], db_codes, n_cands)
            cand = np.asarray(topk).astype(np.intp)
            best = cand[np.argmax(database[cand] @ queries[i])]
            times.append((time.perf_counter() - t0) * 1000)
            if best == gt[i]: correct += 1

        recall = correct / len(queries)
        lat = np.mean(times)
        mem = db_codes.nbytes / 1024 / 1024
        speedup = faiss_ms / lat
        qps = 1000 / lat

        print(f'  {n_bits}b/{n_cands}c        {recall*100:<6.1f}% {lat:<9.3f}ms {speedup:<9.1f}x {mem:<8.1f}MB {qps:<.0f}')

        scale_results.append({
            'method': f'{n_bits}b/{n_cands}c', 'n_bits': n_bits, 'n_candidates': n_cands,
            'recall_1': recall, 'latency_ms': lat, 'speedup': speedup,
            'memory_mb': mem, 'qps': qps,
        })

    all_results[scale_name] = {
        'n_database': n_db, 'n_queries': len(queries),
        'n_identities': int(len(np.unique(labels[:n_db]))),
        'faiss_flat_ms': faiss_ms, 'results': scale_results,
    }

# Save
import os
run_ts = os.environ.get('RUN_TS', 'latest')
(RESULTS_DIR / 'runpod').mkdir(parents=True, exist_ok=True)
for _p in (RESULTS_DIR / 'bench_runpod.json',
           RESULTS_DIR / 'runpod' / f'bench_scale_{run_ts}.json'):
    with open(_p, 'w') as f:
        json.dump(all_results, f, indent=2)
print(f'  ✓ Saved scale results: latest + runpod/bench_scale_{run_ts}.json')
BENCH_EOF

log ""
log "  ✓ Scale benchmark complete"
log ""

# ─────────────────────────────────────────────────────────────────────────────
# Granular, CI-backed benchmark (the trusted one). Auto-loads the largest data
# available (1M if extracted): per-config recall with 95% bootstrap CIs.
# ─────────────────────────────────────────────────────────────────────────────
log "  Running granular grid benchmark (per-config recall + 95% CIs)..."
rm -f results/bench_nbits_grid.json
python benchmarks/bench_nbits_grid.py --queries 1000 2>&1 | tail -45 || true
if [ -f results/bench_nbits_grid.json ]; then
    cp results/bench_nbits_grid.json "results/runpod/bench_grid_${RUN_TS}.json"
    log "  ✓ Granular results: latest + runpod/bench_grid_${RUN_TS}.json"
else
    log "  ⚠ Grid benchmark produced no output (scale results still saved)"
fi
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: COMMIT + PUSH RESULTS NOW (before figures — RunPod is ephemeral)
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 6/7: Commit + push results (doing this ASAP)           │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
commit_and_push "bench: RunPod results (VGGFace2 up to 1M, real data, run ${RUN_TS})

Scale sweep + granular grid (per-config recall with 95% bootstrap CIs).
Provider: $(python -c 'import onnxruntime as o; print(o.get_available_providers()[0])' 2>/dev/null || echo unknown)
Timestamped copies under results/runpod/ (no overwrite)." results/ || true
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: FIGURES (optional) + result safety dump
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 7/7: Figures (optional) + result safety dump           │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
if python scripts/plot_figures.py 2>/dev/null; then
    commit_and_push "docs: refresh figures from RunPod run ${RUN_TS}" docs/figures/ || true
else
    log "  ⚠ Figure generation skipped (non-critical)"
fi
log ""

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY — dump results to the log as a last-resort fallback
# ═══════════════════════════════════════════════════════════════════════════
log "═══════════════════════════════════════════════════════════════"
log "  PIPELINE COMPLETE"
log "═══════════════════════════════════════════════════════════════"
log ""
log "  Saved under results/ and results/runpod/ (timestamped, never overwritten)"
log "  Log: $LOG_FILE"
log ""
log "  If the push FAILED above, copy the JSON below before stopping the pod:"
echo "═══════════════ RESULTS (fallback copy) ═══════════════"
for f in results/bench_nbits_grid.json results/bench_runpod.json; do
    echo "───── $f ─────"
    cat "$f" 2>/dev/null || echo "(missing)"
done
echo "═══════════════════════════════════════════════════════"
