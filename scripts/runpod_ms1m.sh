#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — MS1MV2 Reproducible Benchmark (85K identities, 1M+ embeddings)
# ═══════════════════════════════════════════════════════════════════════════
#
# SELF-CONTAINED: clones the public repo, installs deps, builds the Rust
# backend, pulls pre-extracted embeddings from HuggingFace (public), and
# runs the full benchmark suite — all from this ONE file. No tokens needed.
#
# ONE COMMAND on a fresh RunPod / cloud VM / Linux box:
#
#   git clone https://github.com/raghavenderreddygrudhanti/faceflash.git /workspace/faceflash \
#     && bash /workspace/faceflash/scripts/runpod_ms1m.sh
#
# That's it. No tokens, no API keys, no GPU required.
# Embeddings are pulled from a PUBLIC HuggingFace dataset (~800 MB download).
# Time: ~5-15 min on a CPU pod.
#
# ─── OPTIONAL (for maintainers / re-extraction) ──────────────────────────
#
# FORCE_EXTRACT=1 — re-extract ALL ~85K identities from raw MS1MV2 images.
#   Needs a GPU pod + either HF_IMG_REPO or Kaggle credentials:
#   export FORCE_EXTRACT=1
#   export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>   # image source
#   export HF_TOKEN=hf_xxx                           # to upload new embeddings
#   (takes ~60-90 min)
#
# GITHUB_TOKEN — only needed if you want the script to push results back to
#   the repo (maintainer workflow). Public users just get results locally.
#
# ═══════════════════════════════════════════════════════════════════════════
set +e  # keep going on errors; critical setup steps exit explicitly

REMOTE_SLUG="raghavenderreddygrudhanti/faceflash"
LOG_FILE="/workspace/faceflash_ms1m.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export RUN_TS

log "═══════════════════════════════════════════════════════════════"
log "  FaceFlash — MS1MV2 Standalone Pipeline (85K identities)"
log "═══════════════════════════════════════════════════════════════"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 0: Self-contained setup (clone, venv, deps, Rust, model)
# ─────────────────────────────────────────────────────────────────────────
log "┌─── STEP 0: Setup (clone / venv / deps / Rust / model) ───┐"

WORKDIR="${WORKDIR:-/workspace}"
mkdir -p "$WORKDIR" 2>/dev/null || WORKDIR="$HOME"
cd "$WORKDIR"

if [ ! -d "faceflash" ]; then
    log "  Cloning public repo..."
    git clone "https://github.com/${REMOTE_SLUG}.git" || { log "  ✗ clone failed"; exit 1; }
else
    log "  Repo present — pulling latest..."
    (cd faceflash && git pull 2>/dev/null)
fi
cd faceflash || { log "  ✗ cannot enter repo dir"; exit 1; }

# venv
if [ ! -d ".venv" ]; then
    log "  Creating virtual environment..."
    python -m venv .venv
fi
source .venv/bin/activate || { log "  ✗ venv activation failed"; exit 1; }

# python deps (idempotent; fast if already installed)
log "  Installing Python packages..."
pip install -q numpy pillow tqdm opencv-python-headless faiss-cpu hnswlib usearch datasets huggingface-hub maturin matplotlib kaggle 2>&1 | tail -3
pip install -q scann 2>/dev/null || log "  (ScaNN unavailable on this platform — ANN comparison will skip it)"
# onnxruntime: GPU build if a GPU is present, else CPU (works on cheap CPU pods
# when embeddings come from HuggingFace and no 1M extraction is needed).
if command -v nvidia-smi &>/dev/null; then
    log "  GPU detected → installing onnxruntime-gpu (CUDA 12)..."
    pip install -q onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ 2>&1 | tail -3
else
    log "  No GPU detected → installing onnxruntime (CPU)..."
    pip install -q onnxruntime 2>&1 | tail -3
fi

# Rust backend (POPCNT) — maturin builds faceflash._core into the package.
# Run from repo root (where pyproject.toml lives), NOT from rust/.
# ALWAYS rebuild: a reused pod may carry a stale _core from an earlier commit,
# and skipping the build would silently benchmark the old kernel. Cargo's
# incremental cache makes a no-op rebuild fast.
log "  Building Rust backend (maturin develop --release)..."
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env" 2>/dev/null
maturin develop --release 2>&1 | tail -3
python -c "from faceflash import _core" 2>/dev/null || { log "  ✗ Rust backend failed to build — cannot benchmark"; exit 1; }
log "  ✓ Rust backend ready (freshly built)"

# ArcFace model
MODEL_DIR="$HOME/.faceflash/models"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/w600k_r50.onnx" ]; then
    log "  Downloading ArcFace model (~166MB)..."
    curl -sL -o "$MODEL_DIR/w600k_r50.onnx" \
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
fi
log "  ✓ Setup complete"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Get MS1MV2 embeddings — prefer HuggingFace, fall back to Kaggle+extract
# ─────────────────────────────────────────────────────────────────────────
MS1M_DIR="data/ms1m"
MS1M_OUT="data/ms1m_embeddings.npy"
EXTRACTED_FRESH=0

# FORCE_EXTRACT=1 → ignore any local/HF embeddings and re-extract ALL ~85K
# identities from MS1MV2 (needs a GPU pod + Kaggle keys). The freshly extracted
# 85K embeddings are uploaded back to HF, replacing the old 13.7K subset.
if [ "${FORCE_EXTRACT:-0}" = "1" ]; then
    log "  FORCE_EXTRACT=1 → re-extracting full 85K (skipping local + HF embeddings)"
    rm -f "$MS1M_OUT" 2>/dev/null
fi

if [ "${FORCE_EXTRACT:-0}" != "1" ] && [ -f "$MS1M_OUT" ]; then
    log "  ✓ MS1MV2 embeddings already present locally"
elif [ "${FORCE_EXTRACT:-0}" != "1" ] && { python scripts/hf_sync.py download 2>&1 | tee -a "$LOG_FILE"; [ -f "$MS1M_OUT" ]; }; then
    log "  ✓ Pulled embeddings from HuggingFace — skipped Kaggle download + GPU extraction"
else
    log "  Need raw MS1MV2 images to extract embeddings (full 85K)..."
    mkdir -p "$MS1M_DIR"
    FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)

    # ── Image source 1 (PREFERRED): YOUR HuggingFace dataset ──────────────
    # Pull images from HF_IMG_REPO (or HF_EMB_REPO) and lay them out as
    # data/ms1m/ms1m-arcface/<id>/*.jpg — avoids the 16GB Kaggle download.
    if [ "$FOLDER_COUNT" -lt 70000 ] && [ -n "${HF_IMG_REPO:-${HF_EMB_REPO:-}}" ]; then
        log "  Pulling MS1MV2 images from HuggingFace (${HF_IMG_REPO:-$HF_EMB_REPO})..."
        pip install -q huggingface-hub 2>/dev/null
        python scripts/hf_images.py 2>&1 | tee -a "$LOG_FILE" || log "  (HF image pull failed — falling back to Kaggle)"
        FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
    fi

    # ── Image source 2 (FALLBACK): Kaggle ─────────────────────────────────
    if [ "$FOLDER_COUNT" -lt 70000 ]; then
        if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
            log "  ✗ Images not available from HuggingFace and Kaggle keys not set."
            log "    Either: export HF_IMG_REPO=<user>/<image-dataset>   (preferred)"
            log "    Or:     export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>"
            exit 1
        fi
        if [ ! -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
            log "  Downloading MS1MV2 from Kaggle (~16GB, 5-10 min)..."
            kaggle datasets download -d yakhyokhuja/ms1m-arcface-dataset -p "$MS1M_DIR" 2>&1 | tail -5
            [ -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ] || { log "  ✗ Kaggle download failed!"; exit 1; }
            log "  ✓ Download complete"
        fi
        FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
        if [ "$FOLDER_COUNT" -lt 70000 ] && [ -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
            log "  Extracting MS1MV2 zip (~85K folders, progress every 30s)..."
            unzip -o "$MS1M_DIR/ms1m-arcface-dataset.zip" -d "$MS1M_DIR/" > /tmp/unzip_ms1m.log 2>&1 &
            UNZIP_PID=$!
            while kill -0 $UNZIP_PID 2>/dev/null; do
                log "    extracting... $(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l) / ~85,742 folders"
                sleep 30
            done
            wait $UNZIP_PID
            rm -f "$MS1M_DIR/ms1m-arcface-dataset.zip" 2>/dev/null
        fi
    fi
    FINAL_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
    log "  ✓ Images ready: $FINAL_COUNT identity folders"

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Extract embeddings (scripts/extract_ms1m.py — single source of
    # truth; labels by folder position so any folder naming works).
    # ─────────────────────────────────────────────────────────────────────
    log ""
    # 15 imgs/identity over ~85K identities ≈ 1.1-1.2M embeddings — comfortably
    # above 1M so the 500K/1M ANN/clustering scales are GENUINELY DISTINCT
    # (real subsets, not tiled duplicates). Override with MAX_PER_IDENTITY.
    log "  Extracting ArcFace embeddings — ALL ~85K identities (up to ${MAX_PER_IDENTITY:-15} imgs each ≈ 1.1M+)..."
    python scripts/extract_ms1m.py --ms1m-dir data/ms1m \
        --max-per-identity "${MAX_PER_IDENTITY:-15}" 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "$MS1M_OUT" ]; then
        log "  ✗ MS1MV2 extraction failed!"
        exit 1
    fi
    EXTRACTED_FRESH=1
fi

# Push freshly-extracted embeddings to HuggingFace so future runs skip extraction
if [ "$EXTRACTED_FRESH" = "1" ]; then
    log "  Uploading new embeddings to HuggingFace for future runs..."
    python scripts/hf_sync.py upload 2>&1 | tee -a "$LOG_FILE" || log "  (HF upload skipped — set HF_TOKEN with write access)"
fi

log ""
log "  ✓ MS1MV2 embeddings ready"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 3: ANN comparison on MS1MV2
# ─────────────────────────────────────────────────────────────────────────
log "  Running ANN comparison on MS1MV2 (85K identities)..."
python benchmarks/bench_ann_comparison.py --scales 100K,200K,300K,500K,1M --queries 1000 \
    --data-tag ms1m 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4: 1:N identification on MS1MV2 (the meaningful test).
# Uses benchmarks/bench_identification.py — single source of truth.
# ─────────────────────────────────────────────────────────────────────────
log "  Running 1:N identification on MS1MV2 (sparse gallery)..."
DATA_TAG=ms1m python benchmarks/bench_identification.py 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4b: Alignment benchmark — Haar vs RetinaFace on RAW LFW images.
# (The embedding benchmarks above use pre-aligned data; this is the only one
#  that measures the alignment upgrade.)
# ─────────────────────────────────────────────────────────────────────────
log "  Running alignment benchmark (LFW: Haar vs RetinaFace)..."
# SCRFD det_10g detector for RetinaFace alignment
mkdir -p "$HOME/.faceflash/models"
[ -f "$HOME/.faceflash/models/det_10g.onnx" ] || curl -sL \
    -o "$HOME/.faceflash/models/det_10g.onnx" \
    "https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx"
# LFW is fetched by bench_alignment.py via scikit-learn (download + cache),
# so no fragile manual tarball download is needed.
pip install -q scikit-learn 2>/dev/null
python benchmarks/bench_alignment.py 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4c: Coarse clustering (IVF) — validate sub-linear scan on REAL faces.
# Full scan vs clustered recall@1 + latency, sweeping n_probe.
# ─────────────────────────────────────────────────────────────────────────
log "  Running coarse-clustering benchmark (full scan vs IVF, real MS1MV2)..."
DATA_TAG=ms1m python benchmarks/bench_clustering.py --scales 100K,200K,300K,500K,1M \
    --queries 1000 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4d: SIMD Hamming kernel — raw scan speed (AVX2 on x86, NEON on ARM).
# Self-validates correctness vs NumPy, then times single + parallel scans.
# Synthetic codes (no embeddings needed) → writes results/bench_simd.json.
# ─────────────────────────────────────────────────────────────────────────
log "  Running SIMD Hamming benchmark (AVX2/NEON raw scan speed)..."
python benchmarks/bench_simd.py --scales 100K,500K,1M 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4d-bis: Batch QPS — single-query vs cache-blocked batched throughput.
# Runs on REAL MS1MV2 codes (PCA+ITQ quantized), not synthetic. The batched
# scan tiles the DB into cache-sized blocks and reuses each block across all
# queries in a chunk, so the DB streams from RAM ~once per thread instead of
# once per query. This is where the x86 win shows (DB > LLC at 500K-1M),
# unlike single-query parallel which is bandwidth-bound (~1.3x). Self-validates
# correctness, reports per-query vs batched QPS + the AVX-512 VPOPCNTDQ flag
# (the next compute-bound lever). → bench_batch_qps.json
# ─────────────────────────────────────────────────────────────────────────
log "  Running batch-QPS benchmark (per-query vs cache-blocked batched, real MS1MV2)..."
DATA_TAG=ms1m python benchmarks/bench_batch_qps.py --scales 100K,200K,300K,500K,1M \
    --queries 1000 --data-tag ms1m 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4e: n_bits × candidates grid (recall vs candidates, 95% bootstrap CI).
# Characterises the memory↔recall trade on real MS1MV2 → bench_nbits_grid.json
# (feeds the "recall vs candidates" chart). Uses the full embedding set.
# ─────────────────────────────────────────────────────────────────────────
log "  Running n_bits × candidates grid (recall/CI on real MS1MV2)..."
DATA_TAG=ms1m python benchmarks/bench_nbits_grid.py --queries 1000 \
    2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4f: On-device memory measurement (actual RSS, not modeled).
# Measures committed RAM at each stage — the ground truth for the
# "48–96× less memory" claim. Writes results/bench_memory.json.
# ─────────────────────────────────────────────────────────────────────────
log "  Running on-device memory measurement (actual RSS)..."
DATA_TAG=ms1m python benchmarks/bench_memory.py --scale 100K 2>&1 | tee -a "$LOG_FILE" || true

log ""
log "  ✓ MS1MV2 benchmarks complete"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4e: Regenerate README charts from the fresh result JSONs.
# (Resilient — a missing JSON skips just that chart, not the whole run.)
# ─────────────────────────────────────────────────────────────────────────
log "  Regenerating charts from fresh results..."
pip install -q matplotlib 2>/dev/null
python scripts/plot_charts.py 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Commit + push (optional) + print results folder
# ─────────────────────────────────────────────────────────────────────────
# Commit + push is OPTIONAL — only if GITHUB_TOKEN is set (maintainer workflow).
# Public users just get the results locally in results/ and the tarball.
if [ -n "${GITHUB_TOKEN:-}" ]; then
    log "  GITHUB_TOKEN set → committing and pushing results..."
    git config user.email "bench-bot@faceflash" 2>/dev/null
    git config user.name "FaceFlash Benchmark Bot" 2>/dev/null
    git add results/ docs/figures/ 2>/dev/null || true
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -q -m "bench: MS1MV2 results + charts — ANN + 1:N + clustering + SIMD (${RUN_TS})" || true
        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REMOTE_SLUG}.git"
        CUR_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        git push origin "HEAD:${CUR_BRANCH}" && log "  ✓ Pushed to GitHub (${CUR_BRANCH})" || log "  ✗ Push failed (results still saved locally)"
    fi
else
    log "  Results saved locally (set GITHUB_TOKEN to auto-push to the repo)"
fi

BUNDLE="${WORKDIR:-/workspace}/faceflash_ms1m_results_${RUN_TS}.tar.gz"
tar czf "$BUNDLE" results docs/figures 2>/dev/null || true

log ""
log "═══════════════════════════════════════════════════════════════"
log "  MS1MV2 BENCHMARK COMPLETE"
log "═══════════════════════════════════════════════════════════════"
echo ""
echo "   ➜ RESULTS FOLDER: $(pwd)/results"
echo "       results/bench_ann_comparison_ms1m.json   (ANN, full 85K identities)"
echo "       results/bench_identification_ms1m.json   (1:N rank-1, 85K gallery)"
echo "       results/bench_alignment.json             (Haar vs RetinaFace, LFW)"
echo "       results/bench_clustering.json            (full scan vs IVF clustering)"
echo "       results/bench_simd.json                  (AVX2/NEON raw scan speed)"
echo "       results/bench_batch_qps.json             (batched throughput vs single-query)"
echo "       results/bench_memory.json                (actual RSS, on-device memory)"
echo "       docs/figures/*.png                       (6 regenerated charts)"
echo "   ➜ BUNDLE (download this): ${BUNDLE}"
echo "═══════════════════════════════════════════════════════════════"
