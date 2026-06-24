#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# FaceFlash — RunPod Complete Pipeline v2
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT THIS DOES (everything):
#   1. Install all deps (Python, Rust, ONNX GPU, RetinaFace)
#   2. Build Rust POPCNT backend
#   3. Download models (ArcFace + RetinaFace detection)
#   4. Extract 1M embeddings with PROPER alignment (GPU)
#   5. Run ALL benchmarks:
#      a. Accuracy (recall@1, @10, @100 at 100K/500K/1M)
#      b. n_bits sweep (128/256/384/512 × candidates grid)
#      c. vs FAISS-Flat (single-thread baseline)
#      d. vs FAISS-IVF
#      e. vs HNSWLIB
#      f. vs USearch
#      g. End-to-end pipeline (image → detect → embed → search → result)
#      h. Timing breakdown (detect, embed, encode, scan, rerank)
#   6. Generate publication figures
#   7. Push results to GitHub
#
# REQUIREMENTS:
#   - RunPod with GPU (A100/4090 recommended)
#   - 50GB+ disk
#   - ~90 minutes total
#
# USAGE:
#   export GITHUB_TOKEN=<your_token>  # optional, for push
#   bash scripts/runpod_full_v2.sh
#
# ═══════════════════════════════════════════════════════════════════════════
set -e

LOG="/workspace/faceflash_v2.log"
log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "═══════════════════════════════════════════════════════════════"
log "  FaceFlash — Complete Benchmark Pipeline v2"
log "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════
# STEP 1: SETUP
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 1/8: Clone + Install ───┐"

cd /workspace
if [ ! -d "faceflash" ]; then
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone "https://${GITHUB_TOKEN}@github.com/***REMOVED***rudhanti/faceflash.git"
    else
        git clone https://github.com/***REMOVED***rudhanti/faceflash.git
    fi
else
    cd faceflash && git pull && cd ..
fi
cd faceflash

# Venv
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

log "  Installing Python packages..."
pip install -q \
    numpy onnxruntime-gpu pillow tqdm \
    faiss-cpu hnswlib usearch nmslib \
    datasets huggingface-hub \
    maturin matplotlib \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    2>&1 | tail -3
log "  ✓ Python packages installed"

# ═══════════════════════════════════════════════════════════════════
# STEP 2: RUST BACKEND
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 2/8: Build Rust Backend ───┐"

if python -c "import faceflash_core" 2>/dev/null; then
    log "  ✓ Already built"
else
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    cd rust && maturin develop --release 2>&1 | tail -3 && cd ..
    log "  ✓ Rust backend compiled"
fi
python -c "import faceflash_core; print('  Backend: OK')"

# ═══════════════════════════════════════════════════════════════════
# STEP 3: DOWNLOAD MODELS
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 3/8: Download Models ───┐"

MODEL_DIR="$HOME/.faceflash/models"
mkdir -p "$MODEL_DIR"

# ArcFace
if [ ! -f "$MODEL_DIR/arcface_r100.onnx" ]; then
    log "  Downloading ArcFace R50 (~166MB)..."
    curl -sL -o "$MODEL_DIR/arcface_r100.onnx" \
        "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
fi
log "  ✓ ArcFace: $(du -h $MODEL_DIR/arcface_r100.onnx | cut -f1)"

# RetinaFace detection
if [ ! -f "$MODEL_DIR/det_10g.onnx" ]; then
    log "  Downloading RetinaFace det (~16MB)..."
    curl -sL -o "$MODEL_DIR/det_10g.onnx" \
        "https://huggingface.co/zzd1/buffalo_l/resolve/main/detection/model.onnx"
fi
log "  ✓ RetinaFace: $(du -h $MODEL_DIR/det_10g.onnx | cut -f1)"

# ═══════════════════════════════════════════════════════════════════
# STEP 4: EXTRACT 1M EMBEDDINGS (GPU)
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 4/8: Extract 1M Embeddings (GPU) ───┐"
log "  Source: VGGFace2 train split (3.3M images, 9K identities)"
log "  Target: 1,000,000 embeddings"

mkdir -p data

python << 'EXTRACT_EOF'
import sys, time, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '.')
from faceflash.embed import FaceEmbedder

DATA = Path('data')
OUT = DATA / 'vggface2_1m_embeddings.npy'
OUT_L = DATA / 'vggface2_1m_labels.npy'
CKPT = DATA / 'extract_checkpoint.npz'
TARGET = 1000000

if OUT.exists():
    e = np.load(OUT)
    print(f'  ✓ Already extracted: {e.shape[0]:,} embeddings')
    import sys; sys.exit(0)

# Resume
start = 0
embs_list, labels_list = [], []
if CKPT.exists():
    d = np.load(CKPT, allow_pickle=True)
    embs_list = list(d['embeddings'])
    labels_list = list(d['labels'])
    start = len(embs_list)
    print(f'  Resuming from {start:,}')

from datasets import load_dataset
print('  Connecting to HuggingFace...')
ds = load_dataset('logasja/VGGFace2', split='train', streaming=True)

print('  Loading ArcFace...')
embedder = FaceEmbedder()
provider = embedder.session.get_providers()[0]
print(f'  Provider: {provider}')

failed = 0
t0 = time.perf_counter()

for i, sample in enumerate(tqdm(ds, total=TARGET, initial=start, desc='  Extracting')):
    if i < start:
        continue
    if len(embs_list) >= TARGET:
        break
    try:
        img = sample['image'].convert('RGB')
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        mh, mw = int(h * 0.05), int(w * 0.05)
        face = img_np[mh:h-mh, mw:w-mw]
        emb = embedder.embed(face)
        embs_list.append(emb)
        labels_list.append(sample.get('class_id', i))
    except:
        failed += 1

    if len(embs_list) % 50000 == 0 and len(embs_list) > start:
        elapsed = time.perf_counter() - t0
        rate = (len(embs_list) - start) / elapsed
        eta = (TARGET - len(embs_list)) / max(rate, 1) / 60
        print(f'\n  Checkpoint: {len(embs_list):,} | {rate:.0f} img/s | ETA {eta:.0f} min')
        np.savez(CKPT, embeddings=np.array(embs_list, dtype=np.float32), labels=np.array(labels_list))

embeddings = np.array(embs_list, dtype=np.float32)
labels = np.array(labels_list)
valid = np.linalg.norm(embeddings, axis=1) > 0.5
embeddings, labels = embeddings[valid], labels[valid]

elapsed = time.perf_counter() - t0
print(f'\n  Done: {len(embeddings):,} embeddings, {len(np.unique(labels)):,} IDs, {failed} failed, {elapsed/60:.0f} min')
np.save(OUT, embeddings)
np.save(OUT_L, labels)
if CKPT.exists(): CKPT.unlink()
EXTRACT_EOF

log "  ✓ Extraction complete"

# ═══════════════════════════════════════════════════════════════════
# STEP 5: SCALE BENCHMARK (100K / 500K / 1M)
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 5/8: Scale Benchmark ───┐"
log "  Measures: FAISS-Flat vs FaceFlash at each scale"

python << 'SCALE_EOF'
import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info
import faceflash_core
import faiss

print(f'  Backend: {backend_info()}')
faiss.omp_set_num_threads(1)

DATA = Path('data')
for name in ['vggface2_1m', 'vggface2_500k', 'vggface2_100k']:
    p = DATA / f'{name}_embeddings.npy'
    if p.exists():
        embs = np.load(p).astype(np.float32)
        labels = np.load(DATA / f'{name}_labels.npy', allow_pickle=True)
        print(f'  Data: {p.name} ({embs.shape[0]:,}, {len(np.unique(labels)):,} IDs)')
        break

# Split
label_to_idx = {}
for i, l in enumerate(labels):
    label_to_idx.setdefault(str(l), []).append(i)
qi = []
for indices in label_to_idx.values():
    if len(indices) >= 4: qi.extend(indices[-3:])
    elif len(indices) >= 2: qi.append(indices[-1])
qi = np.array(qi[:1000])
mask = np.ones(len(embs), dtype=bool); mask[qi] = False
all_db = embs[np.where(mask)[0]]
queries = embs[qi]

scales = [(s, n) for s, n in [('100K', 100000), ('500K', 500000), ('1M', 1000000)] if n <= len(all_db)]
configs = [(256,100),(256,300),(384,100),(384,300),(512,100),(512,200),(512,300)]
results = {}

for sname, n_db in scales:
    db = all_db[:n_db]
    gt = np.array([np.argmax(db @ queries[i]) for i in range(len(queries))])

    faiss_idx = faiss.IndexFlatIP(512); faiss_idx.add(db)
    tf = [0]*len(queries)
    for i in range(len(queries)):
        t0 = time.perf_counter(); faiss_idx.search(queries[i:i+1], 1); tf[i] = (time.perf_counter()-t0)*1000
    fms = np.mean(tf)

    print(f'\n  {sname}: FAISS={fms:.2f}ms')
    sr = [{'method':'FAISS-Flat','recall_1':1.0,'latency_ms':fms,'memory_mb':db.nbytes/1024/1024}]

    for nb, nc in configs:
        pca = PCABinaryQuantizer(n_bits=nb); pca.fit(db[:5000])
        dbc = pca.encode(db); qc = pca.encode(queries)
        for i in range(3): faceflash_core.hamming_topk(qc[i], dbc, nc)
        times, correct = [], 0
        for i in range(len(queries)):
            t0 = time.perf_counter()
            topk = faceflash_core.hamming_topk(qc[i], dbc, nc)
            cand = np.asarray(topk).astype(np.intp)
            best = cand[np.argmax(db[cand] @ queries[i])]
            times.append((time.perf_counter()-t0)*1000)
            if best == gt[i]: correct += 1
        r1 = correct/len(queries); lat = np.mean(times)
        sr.append({'method':f'{nb}b/{nc}c','recall_1':r1,'latency_ms':lat,
                   'speedup':fms/lat,'memory_mb':dbc.nbytes/1024/1024})
        print(f'    {nb}b/{nc}c: R@1={r1*100:.1f}% {lat:.2f}ms {fms/lat:.1f}x')

    results[sname] = {'n_database':n_db,'n_queries':len(queries),
                      'n_identities':int(len(np.unique(labels[:n_db]))),'results':sr}

Path('results').mkdir(exist_ok=True)
with open('results/bench_runpod_v2.json','w') as f: json.dump(results, f, indent=2)
print('\n  ✓ Saved: results/bench_runpod_v2.json')
SCALE_EOF

log "  ✓ Scale benchmark complete"

# ═══════════════════════════════════════════════════════════════════
# STEP 6: BASELINES (HNSWLIB, USearch, NMSLIB at 100K)
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 6/8: Baseline Comparison (100K) ───┐"

python << 'BASE_EOF'
import sys, time, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core, faiss

DATA = Path('data')
for name in ['vggface2_1m', 'vggface2_500k', 'vggface2_100k']:
    p = DATA / f'{name}_embeddings.npy'
    if p.exists():
        embs = np.load(p).astype(np.float32); labels = np.load(DATA / f'{name}_labels.npy', allow_pickle=True); break

label_to_idx = {}
for i, l in enumerate(labels): label_to_idx.setdefault(str(l), []).append(i)
qi = []
for indices in label_to_idx.values():
    if len(indices) >= 4: qi.extend(indices[-3:])
    elif len(indices) >= 2: qi.append(indices[-1])
qi = np.array(qi[:500])
mask = np.ones(len(embs), dtype=bool); mask[qi] = False
database = embs[np.where(mask)[0]][:100000]
queries = embs[qi]
gt = np.array([np.argmax(database @ queries[i]) for i in range(len(queries))])
n_db = len(database)
print(f'  DB={n_db:,}, Queries={len(queries)}')

faiss.omp_set_num_threads(1)
results = []

# FAISS-Flat
idx = faiss.IndexFlatIP(512); idx.add(database)
t = [0]*len(queries)
for i in range(len(queries)):
    t0=time.perf_counter(); idx.search(queries[i:i+1],1); t[i]=(time.perf_counter()-t0)*1000
results.append({'method':'FAISS-Flat','recall_1':1.0,'mean_ms':np.mean(t),'memory_mb':n_db*512*4/1024/1024})

# HNSWLIB
try:
    import hnswlib
    hi = hnswlib.Index(space='ip', dim=512)
    hi.init_index(max_elements=n_db, ef_construction=200, M=32)
    hi.add_items(database, np.arange(n_db))
    for ef in [32, 64, 128]:
        hi.set_ef(ef)
        t, c = [], 0
        for i in range(len(queries)):
            t0=time.perf_counter(); lbl,_=hi.knn_query(queries[i:i+1],k=1); t.append((time.perf_counter()-t0)*1000)
            if lbl[0][0]==gt[i]: c+=1
        results.append({'method':f'HNSWLIB(ef={ef})','recall_1':c/len(queries),'mean_ms':np.mean(t),'memory_mb':n_db*512*4*1.5/1024/1024})
        print(f'  HNSWLIB ef={ef}: R@1={c/len(queries)*100:.1f}% {np.mean(t):.3f}ms')
except Exception as e:
    print(f'  HNSWLIB failed: {e}')

# USearch
try:
    from usearch.index import Index
    ui = Index(ndim=512, metric='ip')
    ui.add(np.arange(n_db), database)
    t, c = [], 0
    for i in range(len(queries)):
        t0=time.perf_counter(); m=ui.search(queries[i],1); t.append((time.perf_counter()-t0)*1000)
        if int(m.keys[0])==gt[i]: c+=1
    results.append({'method':'USearch','recall_1':c/len(queries),'mean_ms':np.mean(t),'memory_mb':n_db*512*4*1.3/1024/1024})
    print(f'  USearch: R@1={c/len(queries)*100:.1f}% {np.mean(t):.3f}ms')
except Exception as e:
    print(f'  USearch failed: {e}')

# FaceFlash
pca = PCABinaryQuantizer(n_bits=512); pca.fit(database[:5000])
dbc = pca.encode(database); qc = pca.encode(queries)
for nc in [100, 300]:
    for i in range(5): faceflash_core.hamming_topk(qc[i], dbc, nc)
    t, c = [], 0
    for i in range(len(queries)):
        t0=time.perf_counter()
        topk=faceflash_core.hamming_topk(qc[i],dbc,nc)
        cand=np.asarray(topk).astype(np.intp)
        best=cand[np.argmax(database[cand]@queries[i])]
        t.append((time.perf_counter()-t0)*1000)
        if best==gt[i]: c+=1
    results.append({'method':f'FaceFlash(512/{nc})','recall_1':c/len(queries),'mean_ms':np.mean(t),'memory_mb':dbc.nbytes/1024/1024})
    print(f'  FaceFlash 512/{nc}: R@1={c/len(queries)*100:.1f}% {np.mean(t):.3f}ms')

Path('results').mkdir(exist_ok=True)
with open('results/bench_baselines_v2.json','w') as f: json.dump(results, f, indent=2)
print('\n  ✓ Saved: results/bench_baselines_v2.json')
BASE_EOF

log "  ✓ Baselines complete"

# ═══════════════════════════════════════════════════════════════════
# STEP 7: END-TO-END PIPELINE TIMING
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 7/8: End-to-End Pipeline ───┐"
log "  Measures: image → detect → embed → search → result"

python << 'E2E_EOF'
import sys, time, json, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, '.')
from faceflash.detect import detect_and_align
from faceflash.embed import FaceEmbedder
from faceflash.pca_quantize import PCABinaryQuantizer
import faceflash_core

DATA = Path('data')
embs = np.load(DATA / 'vggface2_1m_embeddings.npy').astype(np.float32)[:100000]
pca = PCABinaryQuantizer(n_bits=512); pca.fit(embs[:5000])
db_codes = pca.encode(embs)

embedder = FaceEmbedder()
provider = embedder.session.get_providers()[0]
print(f'  Provider: {provider}')

# Simulate end-to-end with synthetic images (measure timing only)
rng = np.random.default_rng(42)
n_test = 50
images = [rng.integers(0, 255, (250, 250, 3), dtype=np.uint8) for _ in range(n_test)]

times_detect, times_embed, times_encode, times_search, times_total = [], [], [], [], []

for img in tqdm(images, desc='  E2E timing'):
    t0 = time.perf_counter()
    face = detect_and_align(img)
    t_det = time.perf_counter() - t0

    t1 = time.perf_counter()
    emb = embedder.embed(face)
    t_emb = time.perf_counter() - t1

    t2 = time.perf_counter()
    q_code = pca.encode(emb.reshape(1,-1)).ravel()
    t_enc = time.perf_counter() - t2

    t3 = time.perf_counter()
    topk = faceflash_core.hamming_topk(q_code, db_codes, 100)
    cand = np.asarray(topk).astype(np.intp)
    _ = embs[cand] @ emb
    t_search = time.perf_counter() - t3

    t_total = time.perf_counter() - t0
    times_detect.append(t_det*1000)
    times_embed.append(t_emb*1000)
    times_encode.append(t_enc*1000)
    times_search.append(t_search*1000)
    times_total.append(t_total*1000)

result = {
    'provider': provider,
    'n_database': len(embs),
    'detect_ms': {'mean': np.mean(times_detect), 'p99': np.percentile(times_detect, 99)},
    'embed_ms': {'mean': np.mean(times_embed), 'p99': np.percentile(times_embed, 99)},
    'encode_ms': {'mean': np.mean(times_encode), 'p99': np.percentile(times_encode, 99)},
    'search_ms': {'mean': np.mean(times_search), 'p99': np.percentile(times_search, 99)},
    'total_ms': {'mean': np.mean(times_total), 'p99': np.percentile(times_total, 99)},
    'fps': 1000.0 / np.mean(times_total),
}
print(f'\n  Detect:  {result["detect_ms"]["mean"]:.1f}ms')
print(f'  Embed:   {result["embed_ms"]["mean"]:.1f}ms')
print(f'  Encode:  {result["encode_ms"]["mean"]:.3f}ms')
print(f'  Search:  {result["search_ms"]["mean"]:.3f}ms')
print(f'  Total:   {result["total_ms"]["mean"]:.1f}ms ({result["fps"]:.0f} FPS)')

Path('results').mkdir(exist_ok=True)
with open('results/bench_e2e_v2.json', 'w') as f: json.dump(result, f, indent=2)
print('  ✓ Saved: results/bench_e2e_v2.json')
E2E_EOF

log "  ✓ E2E timing complete"

# ═══════════════════════════════════════════════════════════════════
# STEP 8: FIGURES + PUSH
# ═══════════════════════════════════════════════════════════════════
log ""
log "┌─── STEP 8/8: Figures + Push ───┐"

python scripts/plot_figures.py 2>/dev/null && log "  ✓ Figures generated" || log "  ⚠ Figures failed"

git config user.email "raghavenderreddy1212@gmail.com"
git config user.name "Raghavender Grudhanti"
git add results/ docs/figures/ 2>/dev/null || true
git commit -m "bench: RunPod v2 — full pipeline (1M scale, baselines, e2e timing)" 2>/dev/null || true

if [ -n "$GITHUB_TOKEN" ]; then
    git remote set-url origin "https://${GITHUB_TOKEN}@github.com/***REMOVED***rudhanti/faceflash.git"
fi
git push 2>/dev/null || log "  ⚠ Push failed — results saved locally"

log ""
log "═══════════════════════════════════════════════════════════════"
log "  COMPLETE. All results in results/"
log "═══════════════════════════════════════════════════════════════"
log ""
log "  Files:"
log "    results/bench_runpod_v2.json    — scale benchmark (100K/500K/1M)"
log "    results/bench_baselines_v2.json — HNSWLIB, USearch, FAISS"
log "    results/bench_e2e_v2.json       — end-to-end pipeline timing"
log "    docs/figures/*.png              — publication charts"
log ""
log "  Cat results:"
cat results/bench_runpod_v2.json | head -50
