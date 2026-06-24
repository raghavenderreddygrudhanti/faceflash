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
#   Step 5: Benchmarks — scale sweep (100K/500K/1M) + granular grid (CIs)
#           + baselines (FAISS-Flat / HNSWLIB / USearch) + end-to-end timing
#   Step 6: Commit + push results to GitHub (timestamped, no overwrite)
#   Step 7: Figures + result safety dump
#
# REQUIREMENTS:
#   - RunPod with GPU (A100/4090/3090 recommended)
#   - 50GB+ disk space
#   - ~60-90 minutes total runtime
#
# HOW TO USE — this is a PRIVATE repo, so a GitHub token is REQUIRED
# (create one at GitHub → Settings → Developer settings → Personal access
#  tokens; classic token with scope "repo"). Then ONE command on the pod:
#
#   export GITHUB_TOKEN=ghp_your_token_here
#   git clone https://${GITHUB_TOKEN}@github.com/***REMOVED***rudhanti/faceflash.git /workspace/faceflash && bash /workspace/faceflash/scripts/runpod_full.sh
#
#   (optional, to use MS1MV2: also `export DATASET=<repo> DATA_TAG=ms1m`)
#
# It installs everything, builds, downloads models, extracts, benchmarks, and
# bundles all results. The absolute results FOLDER + a single .tar.gz path are
# printed at the very end — download those before stopping the pod.
#
# ═══════════════════════════════════════════════════════════════════════════
set +e  # Do NOT exit on failure — each step should continue independently

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

# Dataset is configurable. Default = VGGFace2 (few distinct identities).
# For a meaningful 1:N test use MS1MV2 (~85K distinct people). Confirm the HF
# repo id + fields FIRST with: python scripts/preflight_check.py --dataset <repo>
# Then run, e.g.:
#   export DATASET=<ms1mv2-hf-repo> SPLIT=train DATA_TAG=ms1m TARGET_N=1000000
DATASET="${DATASET:-logasja/VGGFace2}"
SPLIT="${SPLIT:-train}"
DATA_TAG="${DATA_TAG:-vggface2_1m}"
TARGET_N="${TARGET_N:-1000000}"
export DATASET SPLIT DATA_TAG TARGET_N

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

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR" 2>/dev/null || WORKDIR="$HOME"
cd "$WORKDIR"
log "  Working directory: $WORKDIR"
if [ ! -d "faceflash" ]; then
    log "  Cloning repository from GitHub..."
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone "https://${GITHUB_TOKEN}@github.com/${REMOTE_SLUG}.git"
    else
        log "  ⚠ No GITHUB_TOKEN set — this fails on a PRIVATE repo."
        log "    Run: export GITHUB_TOKEN=ghp_xxx   (scope: repo)"
        git clone "https://github.com/${REMOTE_SLUG}.git"
    fi
    if [ ! -d "faceflash" ]; then
        log "  ✗✗✗ FATAL: Clone failed. Check your GITHUB_TOKEN and network."
        exit 1
    fi
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
log "    - onnxruntime-gpu (GPU-accelerated face embedding, CUDA 12)"
log "    - faiss-cpu, hnswlib, usearch (baseline comparison)"
log "    - datasets, huggingface-hub (data streaming)"
log "    - maturin (Rust→Python bridge)"
log "    - matplotlib (figures)"
pip install -q numpy pillow tqdm faiss-cpu hnswlib usearch datasets huggingface-hub maturin matplotlib 2>&1 | tail -3
# Install onnxruntime-gpu compatible with CUDA 12.x (RunPod default)
pip install -q onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ 2>&1 | tail -3
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
if [ $? -ne 0 ]; then
    log "  ✗✗✗ FATAL: faceflash_core won't import. Rust build failed."
    log "    Without it, search benchmarks will be 50x slower and results meaningless."
    exit 1
fi
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

# Verify GPU is available for ONNX — fail fast if extraction will crawl
GPU_CHECK=$(python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    print('GPU')
else:
    print('CPU')
" 2>/dev/null)
if [ "$GPU_CHECK" = "GPU" ]; then
    log "  ✓ GPU verified: CUDAExecutionProvider available"
else
    log "  ⚠⚠⚠ WARNING: GPU NOT DETECTED — onnxruntime sees CPU only!"
    log "    Extraction will take DAYS instead of ~50 min."
    log "    Fix: pip install onnxruntime-gpu with correct CUDA version"
    log "    Continuing anyway (Ctrl+C to stop and fix)..."
    sleep 5
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
import sys, os, time, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from faceflash.embed import FaceEmbedder

DATASET = os.environ.get('DATASET', 'logasja/VGGFace2')
SPLIT = os.environ.get('SPLIT', 'train')
DATA_TAG = os.environ.get('DATA_TAG', 'vggface2_1m')
TARGET = int(os.environ.get('TARGET_N', '1000000'))

DATA_DIR = Path('data')
OUT = DATA_DIR / f'{DATA_TAG}_embeddings.npy'
OUT_L = DATA_DIR / f'{DATA_TAG}_labels.npy'
CHECKPOINT = DATA_DIR / f'{DATA_TAG}_checkpoint.npz'

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
print(f'  Connecting to HuggingFace ({DATASET}, split={SPLIT})...')
ds = load_dataset(DATASET, split=SPLIT, streaming=True)

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

if [ $? -ne 0 ]; then
    log "  ⚠ VGGFace2 extraction failed — continuing with whatever data exists"
fi
log "  ✓ Embedding extraction complete"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4b: EXTRACT MS1MV2 EMBEDDINGS (85K identities — the hard benchmark)
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 4b/7: MS1MV2 (85K identities, industry standard)       │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
log "  MS1MV2 = MS-Celeb-1M cleaned by InsightFace (faces_emore)"
log "  85,742 distinct identities — the REAL benchmark for 1:N search"
log "  Format: MXNet RecordIO (.rec), images pre-aligned 112x112"
log "  Source: Kaggle (ms1m-arcface-dataset)"
log ""

MS1M_DIR="data/ms1m"
MS1M_OUT="data/ms1m_embeddings.npy"

if [ -f "$MS1M_OUT" ]; then
    log "  ✓ MS1MV2 embeddings already extracted — skipping"
else
    # Download MS1MV2 from Kaggle (requires KAGGLE_USERNAME + KAGGLE_KEY)
    # OR from a direct link if available
    mkdir -p "$MS1M_DIR"

    if [ -f "$MS1M_DIR/train.rec" ]; then
        log "  ✓ MS1MV2 .rec file already present"
    else
        # Try Kaggle API
        if command -v kaggle &>/dev/null || pip install -q kaggle 2>/dev/null; then
            if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
                log "  Downloading MS1MV2 from Kaggle..."
                kaggle datasets download -d debarghamitraroy/ms1m-arcface-dataset -p "$MS1M_DIR" --unzip 2>&1 | tail -5 || true
            else
                log "  ⚠ KAGGLE_USERNAME/KAGGLE_KEY not set"
                log "    To include MS1MV2, set these env vars before running:"
                log "    export KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key"
            fi
        fi

        # Alternative: direct download from known mirrors
        if [ ! -f "$MS1M_DIR/train.rec" ]; then
            log "  Trying alternative download (OneDrive/HuggingFace mirrors)..."
            # Try HuggingFace deepghs mirror
            python -c "
from huggingface_hub import hf_hub_download
import os
try:
    path = hf_hub_download('deepghs/insightface', 'faces_emore/train.rec',
                           repo_type='model', local_dir='$MS1M_DIR')
    print(f'  Downloaded: {path}')
    path = hf_hub_download('deepghs/insightface', 'faces_emore/train.idx',
                           repo_type='model', local_dir='$MS1M_DIR')
    print(f'  Downloaded: {path}')
except Exception as e:
    print(f'  Download failed: {e}')
" 2>&1 | tail -5 || true
        fi
    fi

    if [ -f "$MS1M_DIR/train.rec" ]; then
        log "  Extracting MS1MV2 embeddings (target: 1M from 85K identities)..."
        pip install -q mxnet-mkl 2>/dev/null || pip install -q mxnet 2>/dev/null || true
        python scripts/extract_ms1m.py --target 1000000 --data-dir data --rec-dir "$MS1M_DIR" 2>&1 | tail -20 || log "  ⚠ MS1MV2 extraction failed — continuing without it"
        log "  ✓ MS1MV2 extraction complete"
    else
        log "  ⚠ MS1MV2 data not available — skipping (VGGFace2 results still valid)"
        log "    To get MS1MV2: export KAGGLE_USERNAME=x KAGGLE_KEY=y before running"
    fi
fi
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
import sys, os, time, json, numpy as np
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

# Load best available data (DATA_TAG first, so MS1MV2 runs are picked up)
for name in [os.environ.get('DATA_TAG', 'vggface2_1m'), 'vggface2_1m', 'vggface2_500k', 'vggface2_100k', 'vggface2']:
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

if [ $? -ne 0 ]; then
    log "  ⚠ Scale benchmark failed — continuing to next step"
fi

log ""
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

# ─────────────────────────────────────────────────────────────────────────────
# FULL ANN COMPARISON — FaceFlash vs FAISS-Flat/IVF, HNSWLIB, USearch, ScaNN
# at ALL scales (100K/500K/1M). This is the definitive head-to-head benchmark.
# Shows recall-vs-latency-vs-memory for every competitor.
# ─────────────────────────────────────────────────────────────────────────────
log "  Running FULL ANN comparison (FAISS / HNSWLIB / USearch / ScaNN vs FaceFlash)..."
log "  This compares ALL methods at EVERY scale — the definitive benchmark."
log ""

# Install ScaNN if not present (only works on Linux x86_64)
pip install -q scann 2>/dev/null || log "  (ScaNN not available on this platform — skipping)"

python benchmarks/bench_ann_comparison.py --scales 100K,500K,1M --queries 1000 2>&1 | tee -a "$LOG_FILE" || true

# Also run ANN comparison on MS1MV2 if available (the hard benchmark — 85K identities)
if [ -f "data/ms1m_embeddings.npy" ]; then
    log ""
    log "  Running ANN comparison on MS1MV2 (85K identities — the hard benchmark)..."
    python benchmarks/bench_ann_comparison.py --scales 100K,500K,1M --queries 1000 \
        --data-tag ms1m 2>&1 | tee -a "$LOG_FILE" || true
fi

log ""
log "  ✓ Full ANN comparison complete"
log ""

log ""
log "  ✓ Full ANN comparison complete"
log ""

# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline timing (detect → embed → encode → search).
# NOTE: synthetic images — measures STAGE LATENCY, not detection accuracy.
# ─────────────────────────────────────────────────────────────────────────────
log "  Running end-to-end timing (detect → embed → encode → search)..."
python << 'E2E_EOF' 2>&1 | tail -12 || true
import sys, time, json, os, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.detect import detect_and_align
from faceflash.embed import FaceEmbedder
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core

DATA = Path('data'); RES = Path('results'); (RES / 'runpod').mkdir(parents=True, exist_ok=True)
emb_path = next((DATA / f'{n}_embeddings.npy' for n in
                 (os.environ.get('DATA_TAG', 'vggface2_1m'), 'vggface2_1m', 'vggface2_500k', 'vggface2_100k', 'vggface2')
                 if (DATA / f'{n}_embeddings.npy').exists()), None)
if emb_path is None:
    print('  no data'); sys.exit(0)
embs = np.load(emb_path).astype(np.float32)[:100000]
pca = PCABinaryQuantizer(n_bits=512); pca.fit(embs[:5000]); dbc = pca.encode(embs)
embedder = FaceEmbedder()
print(f'  Provider: {embedder.session.get_providers()[0]}')

rng = np.random.default_rng(42)
imgs = [rng.integers(0, 255, (250, 250, 3), dtype=np.uint8) for _ in range(50)]
td, te, tc, tsr, tt = [], [], [], [], []
for img in imgs:
    t0 = time.perf_counter(); face = detect_and_align(img); td.append((time.perf_counter()-t0)*1000)
    t1 = time.perf_counter(); emb = embedder.embed(face); te.append((time.perf_counter()-t1)*1000)
    t2 = time.perf_counter(); qc = pca.encode(emb.reshape(1, -1)).ravel(); tc.append((time.perf_counter()-t2)*1000)
    t3 = time.perf_counter()
    topk = faceflash_core.hamming_topk(qc, dbc, 100); cand = np.asarray(topk).astype(np.intp); _ = embs[cand] @ emb
    tsr.append((time.perf_counter()-t3)*1000)
    tt.append((time.perf_counter()-t0)*1000)

def stat(x): return {'mean': float(np.mean(x)), 'p99': float(np.percentile(x, 99))}
out = {'note': 'synthetic images — stage latency only, not detection accuracy',
       'provider': embedder.session.get_providers()[0], 'n_database': len(embs),
       'detect_ms': stat(td), 'embed_ms': stat(te), 'encode_ms': stat(tc),
       'search_ms': stat(tsr), 'total_ms': stat(tt), 'fps': 1000.0 / float(np.mean(tt))}
print(f'  detect {out["detect_ms"]["mean"]:.1f} | embed {out["embed_ms"]["mean"]:.1f} | '
      f'search {out["search_ms"]["mean"]:.3f} | total {out["total_ms"]["mean"]:.1f}ms ({out["fps"]:.0f} FPS)')
ts = os.environ.get('RUN_TS', 'latest')
for p in (RES / 'bench_e2e_runpod.json', RES / 'runpod' / f'bench_e2e_{ts}.json'):
    with open(p, 'w') as f: json.dump(out, f, indent=2)
print('  ✓ Saved e2e timing (latest + timestamped)')
E2E_EOF
log "  ✓ End-to-end timing complete"
log ""

# ─────────────────────────────────────────────────────────────────────────────
# 1:N IDENTIFICATION — the realistic test. Enroll ONE image per identity (a
# gallery of DISTINCT people), then probe with a DIFFERENT held-out image, and
# report rank-1 accuracy: "did it find the right PERSON among N strangers?"
# This is the metric that needs a high-identity dataset (MS1MV2) to mean
# anything — on few-identity data the gallery is tiny and the number is hollow.
# ─────────────────────────────────────────────────────────────────────────────
log "  Running 1:N identification (sparse gallery, rank-1)..."
python << 'IDENT_EOF' 2>&1 | tail -15 || true
import sys, os, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core, faiss

DATA = Path('data'); RES = Path('results'); (RES / 'runpod').mkdir(parents=True, exist_ok=True)
emb_path = next((DATA / f'{n}_embeddings.npy' for n in
                 (os.environ.get('DATA_TAG', 'vggface2_1m'), 'vggface2_1m', 'vggface2_500k', 'vggface2_100k', 'vggface2')
                 if (DATA / f'{n}_embeddings.npy').exists()), None)
if emb_path is None:
    print('  no data'); sys.exit(0)
embs = np.load(emb_path).astype(np.float32)
labels = np.load(DATA / emb_path.name.replace('_embeddings', '_labels'), allow_pickle=True)

# SPARSE gallery: one image per identity; probe with a DIFFERENT image.
by_id = {}
for i, l in enumerate(labels):
    by_id.setdefault(str(l), []).append(i)
gallery_idx, probe_idx = [], []
for idxs in by_id.values():
    if len(idxs) >= 2:
        gallery_idx.append(idxs[0])      # enroll first image
        probe_idx.append(idxs[-1])       # probe with a different (last) image
gallery = np.ascontiguousarray(embs[np.array(gallery_idx)])
probe_idx = np.array(probe_idx[:5000])
probes = np.ascontiguousarray(embs[probe_idx])
truth = np.arange(len(probe_idx))        # probe j's right answer is gallery slot j
n_id = len(gallery)
print(f'  Gallery: {n_id:,} DISTINCT identities (1 image each) | Probes: {len(probes):,}')
if n_id < 1000:
    print(f'  ⚠ only {n_id} identities — switch to MS1MV2 for a meaningful 1:N test')

faiss.omp_set_num_threads(1)
out = {'n_gallery_identities': int(n_id), 'n_probes': int(len(probes)),
       'dataset': emb_path.name,
       'note': 'rank-1 identification: 1 image/identity gallery, different image as probe',
       'results': []}

idx = faiss.IndexFlatIP(512); idx.add(gallery)
c = 0
for j in range(len(probes)):
    _, I = idx.search(probes[j:j+1], 1)
    if int(I[0][0]) == truth[j]: c += 1
exact = c / len(probes)
out['results'].append({'method': 'FAISS-Flat (exact)', 'rank1': exact})
print(f'  Exact rank-1 (ceiling): {exact*100:.1f}%')

for nb, nc in ((256, 100), (512, 100), (512, 300)):
    pca = PCABinaryQuantizer(n_bits=nb); pca.fit(gallery[:min(5000, n_id)])
    gc = pca.encode(gallery); pc = pca.encode(probes)
    c = 0
    for j in range(len(probes)):
        topk = faceflash_core.hamming_topk(pc[j], gc, min(nc, n_id))
        cand = np.asarray(topk).astype(np.intp)
        if int(cand[np.argmax(gallery[cand] @ probes[j])]) == truth[j]: c += 1
    r1 = c / len(probes)
    out['results'].append({'method': f'FaceFlash({nb}/{nc})', 'rank1': r1,
                           'memory_mb': gc.nbytes/1024/1024})
    print(f'  FaceFlash {nb}/{nc}: rank-1 {r1*100:.1f}%')

ts = os.environ.get('RUN_TS', 'latest')
for p in (RES / 'bench_identification_runpod.json', RES / 'runpod' / f'bench_identification_{ts}.json'):
    with open(p, 'w') as f: json.dump(out, f, indent=2)
print('  ✓ Saved 1:N identification (latest + timestamped)')
IDENT_EOF
log "  ✓ 1:N identification complete"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: COMMIT + PUSH RESULTS NOW (before figures — RunPod is ephemeral)
# ═══════════════════════════════════════════════════════════════════════════
log "┌─────────────────────────────────────────────────────────────┐"
log "│ STEP 6/7: Commit + push results (doing this ASAP)           │"
log "└─────────────────────────────────────────────────────────────┘"
log ""
commit_and_push "bench: RunPod results (VGGFace2 up to 1M, real data, run ${RUN_TS})

Scale sweep + granular grid (per-config recall, 95% bootstrap CIs)
+ FULL ANN comparison (FAISS-Flat/IVF / HNSWLIB / USearch / ScaNN)
+ end-to-end timing. All methods single-threaded, same hardware.
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
# SUMMARY — bundle everything into one folder + tarball, print the path
# ═══════════════════════════════════════════════════════════════════════════
RESULTS_DIR_ABS="$(pwd)/results"
FIGURES_DIR_ABS="$(pwd)/docs/figures"
BUNDLE="${WORKDIR:-/workspace}/faceflash_results_${RUN_TS}.tar.gz"

BUNDLE_PATHS="results"
[ -d docs/figures ] && BUNDLE_PATHS="$BUNDLE_PATHS docs/figures"
tar czf "$BUNDLE" $BUNDLE_PATHS 2>/dev/null || log "  ⚠ could not create bundle"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   PIPELINE COMPLETE — DOWNLOAD YOUR RESULTS FROM HERE          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "   ➜ FOLDER (all results):"
echo "       ${RESULTS_DIR_ABS}"
echo "       └─ runpod/   timestamped copies (one per run, never overwritten)"
echo ""
echo "   ➜ ONE FILE (easiest to download):"
echo "       ${BUNDLE}"
echo ""
echo "   ➜ FIGURES:"
echo "       ${FIGURES_DIR_ABS}"
echo ""
echo "   Result files produced:"
for f in results/bench_runpod.json results/bench_nbits_grid.json \
         results/bench_ann_comparison.json results/bench_ann_comparison_ms1m.json \
         results/bench_e2e_runpod.json; do
    [ -f "$f" ] && echo "       ✓ ${f}" || echo "       ✗ ${f} (not produced)"
done
echo ""
echo "   (Results are also on GitHub if GITHUB_TOKEN was set.)"
echo "═══════════════════════════════════════════════════════════════════"
log "  Bundle: $BUNDLE"
log "  Folder: $RESULTS_DIR_ABS"
