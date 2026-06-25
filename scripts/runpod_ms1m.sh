#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — MS1MV2 Standalone Pipeline (85K identities)
# ═══════════════════════════════════════════════════════════════════════════
#
# SELF-CONTAINED: clones, installs, builds, downloads the model, gets MS1MV2
# from Kaggle, extracts embeddings, and benchmarks — all from this ONE file.
# (Every setup step is idempotent: it skips whatever is already done, so it's
#  safe to run after runpod_full.sh too.)
#
# REQUIRES (set in your terminal — NEVER commit these):
#   export GITHUB_TOKEN=<your-github-token>     # private repo needs this
#   export KAGGLE_USERNAME=<your-kaggle-username>
#   export KAGGLE_KEY=<your-kaggle-key>
#
# ONE command on a fresh RunPod terminal:
#   export GITHUB_TOKEN=ghp_xxx KAGGLE_USERNAME=you KAGGLE_KEY=key
#   git clone https://${GITHUB_TOKEN}@github.com/raghavenderreddygrudhanti/faceflash.git /workspace/faceflash && bash /workspace/faceflash/scripts/runpod_ms1m.sh
#
# Time: ~60-90 min (download ~16GB + extract 1M embeddings + benchmarks)
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
    if [ -n "$GITHUB_TOKEN" ]; then
        log "  Cloning private repo..."
        git clone "https://${GITHUB_TOKEN}@github.com/${REMOTE_SLUG}.git" || { log "  ✗ clone failed"; exit 1; }
    else
        log "  ✗ Private repo needs a token. Run: export GITHUB_TOKEN=ghp_xxx (scope: repo)"
        exit 1
    fi
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
pip install -q numpy pillow tqdm faiss-cpu hnswlib usearch datasets huggingface-hub maturin matplotlib kaggle 2>&1 | tail -3
log "  Installing onnxruntime-gpu (CUDA 12)..."
pip install -q onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ 2>&1 | tail -3

# Rust backend (POPCNT) — build only if not importable
if ! python -c "import faceflash_core" 2>/dev/null; then
    log "  Building Rust backend..."
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    (cd rust && maturin develop --release 2>&1 | tail -3)
fi
python -c "import faceflash_core" 2>/dev/null || { log "  ✗ Rust backend failed to build — cannot benchmark"; exit 1; }
log "  ✓ Rust backend ready"

# ArcFace model
MODEL_DIR="$HOME/.faceflash/models"
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_DIR/arcface_r100.onnx" ]; then
    log "  Downloading ArcFace model (~166MB)..."
    curl -sL -o "$MODEL_DIR/arcface_r100.onnx" \
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
fi
log "  ✓ Setup complete"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Download MS1MV2 from Kaggle
# ─────────────────────────────────────────────────────────────────────────
MS1M_DIR="data/ms1m"
MS1M_OUT="data/ms1m_embeddings.npy"

if [ -f "$MS1M_OUT" ]; then
    log "  ✓ MS1MV2 embeddings already exist — skipping download+extraction"
else
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        log "  ✗ KAGGLE_USERNAME and KAGGLE_KEY must be set!"
        log "    export KAGGLE_USERNAME=<your-kaggle-username>"
        log "    export KAGGLE_KEY=<your-kaggle-key>"
        exit 1
    fi

    # Download zip if not present
    if [ ! -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
        FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
        if [ "$FOLDER_COUNT" -ge 70000 ]; then
            log "  ✓ MS1MV2 already fully extracted ($FOLDER_COUNT folders)"
        else
            log "  Downloading MS1MV2 from Kaggle (~16GB, takes 5-10 min)..."
            mkdir -p "$MS1M_DIR"
            kaggle datasets download -d yakhyokhuja/ms1m-arcface-dataset -p "$MS1M_DIR" 2>&1 | tail -5
            if [ ! -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
                log "  ✗ Kaggle download failed!"
                exit 1
            fi
            log "  ✓ Download complete"
        fi
    else
        log "  ✓ MS1MV2 zip already present"
    fi

    # Extract zip if needed (check folder count)
    FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
    if [ "$FOLDER_COUNT" -lt 70000 ] && [ -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
        log "  Extracting MS1MV2 zip (~85K folders, progress every 30s)..."
        log "  Currently: $FOLDER_COUNT folders"
        unzip -o "$MS1M_DIR/ms1m-arcface-dataset.zip" -d "$MS1M_DIR/" > /tmp/unzip_ms1m.log 2>&1 &
        UNZIP_PID=$!
        while kill -0 $UNZIP_PID 2>/dev/null; do
            CURRENT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
            log "    extracting... $CURRENT / ~85,742 folders"
            sleep 30
        done
        wait $UNZIP_PID
        FINAL_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
        log "  ✓ Extraction done: $FINAL_COUNT identity folders"
        rm -f "$MS1M_DIR/ms1m-arcface-dataset.zip" 2>/dev/null
    else
        log "  ✓ MS1MV2 extraction complete: $FOLDER_COUNT identity folders"
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Extract embeddings (scripts/extract_ms1m.py — single source of
    # truth; labels by folder position so any folder naming works).
    # ─────────────────────────────────────────────────────────────────────
    log ""
    log "  Extracting ArcFace embeddings from MS1MV2 (target: 1M)..."
    python scripts/extract_ms1m.py --ms1m-dir data/ms1m --target 1000000 2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "$MS1M_OUT" ]; then
        log "  ✗ MS1MV2 extraction failed!"
        exit 1
    fi
fi

log ""
log "  ✓ MS1MV2 embeddings ready"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 3: ANN comparison on MS1MV2
# ─────────────────────────────────────────────────────────────────────────
log "  Running ANN comparison on MS1MV2 (85K identities)..."
python benchmarks/bench_ann_comparison.py --scales 100K,500K,1M --queries 1000 \
    --data-tag ms1m 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4: 1:N identification on MS1MV2 (the meaningful test).
# Uses benchmarks/bench_identification.py — single source of truth.
# ─────────────────────────────────────────────────────────────────────────
log "  Running 1:N identification on MS1MV2 (sparse gallery)..."
DATA_TAG=ms1m python benchmarks/bench_identification.py 2>&1 | tee -a "$LOG_FILE" || true

log ""
log "  ✓ MS1MV2 benchmarks complete"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Commit + push (optional) + print results folder
# ─────────────────────────────────────────────────────────────────────────
log "  Committing MS1MV2 results..."
git config user.email "raghavenderreddy1212@gmail.com"
git config user.name "Raghavender Grudhanti"
git add results/ 2>/dev/null || true

if ! git diff --cached --quiet 2>/dev/null; then
    git commit -q -m "bench: MS1MV2 results — 85K identities, ANN comparison + 1:N identification (${RUN_TS})" || true
    if [ -n "$GITHUB_TOKEN" ]; then
        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REMOTE_SLUG}.git"
    fi
    git push origin HEAD:main && log "  ✓ Pushed to GitHub" || log "  ✗ Push failed (results still saved locally)"
fi

BUNDLE="${WORKDIR:-/workspace}/faceflash_ms1m_results_${RUN_TS}.tar.gz"
tar czf "$BUNDLE" results 2>/dev/null || true

log ""
log "═══════════════════════════════════════════════════════════════"
log "  MS1MV2 BENCHMARK COMPLETE"
log "═══════════════════════════════════════════════════════════════"
echo ""
echo "   ➜ RESULTS FOLDER: $(pwd)/results"
echo "       results/bench_ann_comparison_ms1m.json"
echo "       results/bench_identification_ms1m.json"
echo "   ➜ BUNDLE (download this): ${BUNDLE}"
echo "═══════════════════════════════════════════════════════════════"
