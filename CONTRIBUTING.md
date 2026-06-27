# Contributing to FaceFlash

Thanks for your interest. Here's how to get started.

## Dev Setup (one command)

```bash
git clone https://github.com/raghavenderreddygrudhanti/faceflash.git
cd faceflash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,benchmark]"

# Build the Rust AVX-512/NEON backend (requires Rust toolchain: https://rustup.rs)
maturin develop --release

# Verify
python -c "from faceflash import FaceFlash; print('OK')"
python -c "from faceflash._core import simd_info; print(simd_info())"
pytest tests/ -v   # 36 tests, ~19s
```

## Running Benchmarks

```bash
# Download test data (LFW, ~200MB)
python scripts/extract_lfw_embeddings.py

# Quick local benchmark
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500

# Full suite on RunPod (MS1MV2, all scales)
bash scripts/runpod_ms1m.sh
```

## Code Style

- Python: type-hinted, docstrings on public methods, match existing patterns
- Rust: `cargo fmt` and `cargo clippy` clean
- No strict linter enforced — just keep it consistent with what's there

## What to Work On

| Area | Difficulty | Impact |
|------|-----------|--------|
| **DiskANN comparison** | Medium | High — the one competitor missing |
| **Mobile deployment** (ONNX + CoreML) | Medium | High — iOS/Android face search |
| **Streaming insertion** (no PCA refit) | Hard | High — online learning |
| **GPU batched search** (CUDA) | Hard | Medium — 10M+ galleries |
| **Raspberry Pi / Jetson benchmarks** | Easy | Medium — edge credibility |
| **REST API server** (FastAPI) | Medium | High — production deployment |
| **Liveness detection** | Medium | High — anti-spoofing |
| **WebAssembly build** | Medium | Medium — browser face search |

## Pull Requests

- One concern per PR
- Include benchmark numbers if your change affects performance
- Add tests for new functionality
- Keep the README honest — update claims if numbers change

## Questions?

Open an issue. No question is too basic.
