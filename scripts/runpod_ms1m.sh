#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — MS1MV2 Only (85K identities benchmark)
# ═══════════════════════════════════════════════════════════════════════════
#
# Run this AFTER runpod_full.sh (environment already set up).
# Downloads MS1MV2 from Kaggle, extracts embeddings, runs benchmarks.
#
# REQUIRES:
#   export KAGGLE_USERNAME=***REMOVED***
#   export KAGGLE_KEY=***REMOVED***
#   export GITHUB_TOKEN=<your-token>
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
        log "    export KAGGLE_USERNAME=***REMOVED***"
        log "    export KAGGLE_KEY=<your-key>"
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
    # Step 2: Extract embeddings from folder-of-JPEGs
    # ─────────────────────────────────────────────────────────────────────
    log ""
    log "  Extracting ArcFace embeddings from MS1MV2 (target: 1M)..."
    log "  Format: folders of JPEGs (folder name = identity)"
    log ""

    python << 'MS1M_EXTRACT'
import sys, os, time, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from faceflash.embed import FaceEmbedder

MS1M_DIR = Path('data/ms1m')
OUT = Path('data/ms1m_embeddings.npy')
OUT_L = Path('data/ms1m_labels.npy')
TARGET = 1000000

if OUT.exists():
    embs = np.load(OUT)
    print(f'  ✓ Already extracted: {embs.shape[0]:,} embeddings')
    sys.exit(0)

# Find the root with identity folders
# Structure: ms1m/ms1m-arcface/0/, ms1m/ms1m-arcface/1/, ... OR ms1m/0/, ms1m/1/, ...
root = MS1M_DIR
candidates = [MS1M_DIR, MS1M_DIR / 'ms1m-arcface', MS1M_DIR / 'ms1m_arcface',
              MS1M_DIR / 'faces_emore', MS1M_DIR / 'train']
for c in candidates:
    if c.exists() and any(c.iterdir()):
        # Check if subfolders contain images
        sub = next(c.iterdir(), None)
        if sub and sub.is_dir():
            root = c
            break

# Gather all identity folders
id_folders = sorted([f for f in root.iterdir() if f.is_dir()])
print(f'  Root: {root}')
print(f'  Identity folders found: {len(id_folders):,}')

if len(id_folders) < 100:
    print(f'  ERROR: Expected 85K+ folders, found {len(id_folders)}')
    print(f'  Contents of {root}: {[x.name for x in list(root.iterdir())[:20]]}')
    sys.exit(1)

# Load embedder
embedder = FaceEmbedder()
import onnxruntime as ort
provider = embedder.session.get_providers()[0]
print(f'  ONNX provider: {provider}')
if 'CUDA' in provider:
    print(f'  ★ GPU acceleration active')

embeddings = []
labels = []
failed = 0
t_start = time.perf_counter()

for folder in tqdm(id_folders, desc='  Identities'):
    if len(embeddings) >= TARGET:
        break
    
    identity_id = folder.name
    images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
    
    for img_path in images:
        if len(embeddings) >= TARGET:
            break
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            # Images are already 112x112 aligned — embed directly
            emb = embedder.embed(img_np)
            embeddings.append(emb)
            labels.append(int(identity_id))
        except Exception:
            failed += 1
    
    # Checkpoint every 100K
    if len(embeddings) % 100000 < len(images) and len(embeddings) > 0:
        elapsed = time.perf_counter() - t_start
        rate = len(embeddings) / elapsed
        remaining = (TARGET - len(embeddings)) / max(rate, 1) / 60
        print(f'\n  Progress: {len(embeddings):,} embeddings | '
              f'{rate:.0f} img/s | ETA {remaining:.0f} min | Failed: {failed}')

# Save
embeddings = np.array(embeddings, dtype=np.float32)
labels = np.array(labels)
valid = np.linalg.norm(embeddings, axis=1) > 0.5
embeddings = embeddings[valid]
labels = labels[valid]

n_ids = len(np.unique(labels))
elapsed = time.perf_counter() - t_start
print(f'\n  ═══════════════════════════════════════════════════')
print(f'  MS1MV2 EXTRACTION COMPLETE')
print(f'  Embeddings: {len(embeddings):,}')
print(f'  Identities: {n_ids:,}')
print(f'  Failed: {failed}')
print(f'  Time: {elapsed/60:.1f} min ({len(embeddings)/elapsed:.0f} img/s)')
print(f'  ═══════════════════════════════════════════════════')

np.save(OUT, embeddings)
np.save(OUT_L, labels)
print(f'  ✓ Saved: {OUT} ({embeddings.nbytes/1024/1024:.0f} MB)')
MS1M_EXTRACT

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
# Step 4: Run 1:N identification on MS1MV2 (the meaningful test)
# ─────────────────────────────────────────────────────────────────────────
log "  Running 1:N identification on MS1MV2 (sparse gallery)..."
python << 'IDENT_MS1M' 2>&1 | tee -a "$LOG_FILE" || true
import sys, os, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core, faiss

DATA = Path('data'); RES = Path('results'); (RES / 'runpod').mkdir(parents=True, exist_ok=True)
embs = np.load(DATA / 'ms1m_embeddings.npy').astype(np.float32)
labels = np.load(DATA / 'ms1m_labels.npy', allow_pickle=True)

# SPARSE gallery: one image per identity; probe with DIFFERENT image.
by_id = {}
for i, l in enumerate(labels):
    by_id.setdefault(int(l), []).append(i)

gallery_idx, probe_idx = [], []
for idxs in by_id.values():
    if len(idxs) >= 2:
        gallery_idx.append(idxs[0])
        probe_idx.append(idxs[-1])

gallery = np.ascontiguousarray(embs[np.array(gallery_idx)])
probes = np.ascontiguousarray(embs[np.array(probe_idx[:5000])])
truth = np.arange(len(probes))
n_id = len(gallery)
print(f'  Gallery: {n_id:,} DISTINCT identities | Probes: {len(probes):,}')

faiss.omp_set_num_threads(1)
out = {'dataset': 'ms1m', 'n_gallery_identities': int(n_id), 'n_probes': int(len(probes)),
       'note': '1:N identification — 1 image/identity gallery, different image as probe',
       'results': []}

# Exact baseline
idx = faiss.IndexFlatIP(512); idx.add(gallery)
c = 0
for j in range(len(probes)):
    _, I = idx.search(probes[j:j+1], 1)
    if int(I[0][0]) == truth[j]: c += 1
exact = c / len(probes)
out['results'].append({'method': 'FAISS-Flat (exact)', 'rank1': exact})
print(f'  Exact rank-1 (ceiling): {exact*100:.1f}%')

# FaceFlash configs
for nb, nc in ((256, 100), (384, 300), (512, 100), (512, 300)):
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
for p in (RES / 'bench_identification_ms1m.json', RES / 'runpod' / f'bench_identification_ms1m_{ts}.json'):
    with open(p, 'w') as f: json.dump(out, f, indent=2)
print('  ✓ Saved MS1MV2 identification results')
IDENT_MS1M

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
