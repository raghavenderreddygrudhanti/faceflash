# Contributing to FaceFlash

Thanks for your interest. Here's how to get started.

## Dev Setup

```bash
# Clone
git clone https://github.com/***REMOVED***rudhanti/faceflash.git
cd faceflash

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmark]"

# Rust backend (optional but recommended — 50x faster)
# Requires Rust toolchain: https://rustup.rs
cd rust && maturin develop --release && cd ..

# Verify
python -c "import faceflash; print(faceflash.__version__)"
python -c "import faceflash_core; print('Rust backend OK')"
```

## Running Tests

```bash
pytest tests/ -v
```

## Running Benchmarks

```bash
# Download test data (LFW, ~200MB)
python scripts/extract_lfw_embeddings.py

# Quick local benchmark
python benchmarks/bench_search.py

# Full ANN comparison (needs faiss, hnswlib, usearch)
python benchmarks/bench_ann_comparison.py --scales 100K --queries 500
```

## Code Style

- Python: keep it readable, type-hinted, docstrings on public methods
- Rust: `cargo fmt` and `cargo clippy` clean
- No linter enforced yet — just match the existing style

## What to Work On

Check the [README contributing section](README.md#contributing) for the roadmap.
The highest-impact items:

1. **RetinaFace alignment** — replace Haar cascade with proper 5-point face alignment
2. **Prebuilt wheels** — set up maturin CI to ship Rust in the pip package
3. **Coarse clustering** — partition codes into buckets for sub-linear scan

## Pull Requests

- One concern per PR
- Include benchmark numbers if your change affects performance
- Add tests for new functionality
- Keep the README honest — update claims if numbers change

## Questions?

Open an issue. No question is too basic.
