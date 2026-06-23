#!/bin/bash
# FaceFlash — One-Command Benchmark Suite
# Reproduces all results from the paper.
#
# Usage:
#   cd ~/lang-chain/faceflash
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh
#
# Prerequisites:
#   - Python 3.9+ with .venv set up
#   - pip install -e . (or pip install faceflash)
#   - pip install faiss-cpu hnswlib nmslib usearch datasets
#   - Rust backend: cd rust && maturin develop --release

set -e

VENV=".venv/bin/python"
echo "============================================================"
echo "  FaceFlash — Full Benchmark Suite"
echo "============================================================"
echo ""
echo "  This will:"
echo "    1. Download LFW and extract embeddings (~5 min)"
echo "    2. Download VGGFace2 and extract 100K embeddings (~20 min)"
echo "    3. Run all benchmarks (~5 min)"
echo ""
echo "  Total time: ~30 minutes"
echo "  Results saved to: results/"
echo ""
echo "============================================================"
echo ""

# Step 1: Data
echo "[1/5] Extracting LFW embeddings..."
if [ -f "data/lfw_embeddings.npy" ]; then
    echo "  Already exists. Skipping."
else
    $VENV scripts/extract_lfw_embeddings.py
fi

echo ""
echo "[2/5] Extracting VGGFace2 embeddings (100K)..."
if [ -f "data/vggface2_100k_embeddings.npy" ]; then
    echo "  Already exists. Skipping."
else
    $VENV scripts/extract_vggface2_fast.py
fi

# Step 2: Benchmarks
echo ""
echo "[3/5] Running comprehensive benchmark (LFW + VGGFace2)..."
$VENV benchmarks/bench_accuracy.py

echo ""
echo "[4/5] Running Tier 1 baselines (FAISS, HNSWLIB, USearch)..."
$VENV benchmarks/bench_vector_search.py

echo ""
echo "[5/5] Running face library comparison..."
$VENV benchmarks/bench_face_libraries.py

echo ""
echo "============================================================"
echo "  DONE. All results in results/"
echo "============================================================"
echo ""
ls -la results/*.json
