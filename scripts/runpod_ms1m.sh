#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — MS1MV2 Only (85K identities benchmark)
# ═══════════════════════════════════════════════════════════════════════════
#
# Run this AFTER runpod_full.sh (environment already set up).
# Downloads MS1MV2 from Kaggle, extracts embeddings, runs benchmarks.
#
# REQUIRES (set these in your terminal — NEVER commit them to the repo):
#   export KAGGLE_USERNAME=<your-kaggle-username>
#   export KAGGLE_KEY=<your-kaggle-key>
#   export GITHUB_TOKEN=<your-github-token>
#
# Usage:
#   bash scripts/runpod_ms1m.sh
#
# Time: ~60-90 min (download ~16GB + extract 1M embeddings + benchmarks)
# ═══════════════════════════════════════════════════════════════════════════
set +e

cd /workspace/faceflash
source .venv/bin/activate 2>/dev/null || { echo "No .venv — run runpod_full.sh first"; exit 1; }

LOG_FILE="/workspace/faceflash_ms1m.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
RUN_TS="$(date +%Y%m%d_%H%M%S)"
export RUN_TS

REMOTE_SLUG="***REMOVED***rudhanti/faceflash"

log "═══════════════════════════════════════════════════════════════"
log "  FaceFlash — MS1MV2 Benchmark (85K identities)"
log "═══════════════════════════════════════════════════════════════"
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

    pip install -q kaggle 2>/dev/null

    # Step 1a: Download zip if not present
    if [ ! -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
        # Check if already fully extracted (85K+ folders)
        FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
        if [ "$FOLDER_COUNT" -ge 80000 ]; then
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

    # Step 1b: Extract zip if needed (check folder count)
    FOLDER_COUNT=$(ls "$MS1M_DIR/ms1m-arcface" 2>/dev/null | wc -l)
    if [ "$FOLDER_COUNT" -lt 80000 ] && [ -f "$MS1M_DIR/ms1m-arcface-dataset.zip" ]; then
        log "  Extracting MS1MV2 zip (~85K folders, showing progress every 30s)..."
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
        # Clean up zip to save disk
        rm -f "$MS1M_DIR/ms1m-arcface-dataset.zip" 2>/dev/null
    else
        log "  ✓ MS1MV2 extraction complete: $FOLDER_COUNT identity folders"
    fi

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Extract embeddings (uses scripts/extract_ms1m.py — single source
    # of truth; labels by folder position so any naming works).
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
# Step 3: Run ANN comparison on MS1MV2
# ─────────────────────────────────────────────────────────────────────────
log "  Running ANN comparison on MS1MV2 (85K identities)..."
python benchmarks/bench_ann_comparison.py --scales 100K,500K,1M --queries 1000 \
    --data-tag ms1m 2>&1 | tee -a "$LOG_FILE" || true

log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 4: Run 1:N identification on MS1MV2 (the meaningful test).
# Uses benchmarks/bench_identification.py — single source of truth.
# ─────────────────────────────────────────────────────────────────────────
log "  Running 1:N identification on MS1MV2 (sparse gallery)..."
DATA_TAG=ms1m python benchmarks/bench_identification.py 2>&1 | tee -a "$LOG_FILE" || true

log ""
log "  ✓ MS1MV2 benchmarks complete"
log ""

# ─────────────────────────────────────────────────────────────────────────
# Step 5: Commit + push
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
    git push origin HEAD:main && log "  ✓ Pushed to GitHub" || log "  ✗ Push failed"
fi

log ""
log "═══════════════════════════════════════════════════════════════"
log "  MS1MV2 BENCHMARK COMPLETE"
log "═══════════════════════════════════════════════════════════════"
log "  Results:"
log "    results/bench_ann_comparison_ms1m.json"
log "    results/bench_identification_ms1m.json"
log "═══════════════════════════════════════════════════════════════"
