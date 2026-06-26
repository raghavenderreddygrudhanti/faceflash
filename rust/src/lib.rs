//! FaceFlash Core — Rust binary search engine.
//! Hardware-accelerated Hamming distance via SIMD (AVX2/NEON) with scalar fallback.
//! Uses Rayon for optional parallelism.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// SIMD Hamming kernels — compile-time feature detection
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 Hamming distance: processes 32 bytes (256 bits) per iteration.
/// ~3-4x faster than scalar u64 POPCNT on x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let mut dist: u32 = 0;
    let len = a.len();
    let mut i = 0;

    // Process 32 bytes at a time with AVX2
    while i + 32 <= len {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let xor = _mm256_xor_si256(va, vb);

        // Count bits using the standard POPCNT-via-lookup approach for AVX2:
        // Split each byte into nibbles, lookup popcount in a table
        let low_mask = _mm256_set1_epi8(0x0f);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        );

        let lo = _mm256_and_si256(xor, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);
        let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
        let sum = _mm256_add_epi8(popcnt_lo, popcnt_hi);

        // Horizontal sum: sad against zero gives u64 sums of bytes
        let sad = _mm256_sad_epu8(sum, _mm256_setzero_si256());

        // Extract the 4 u64 values and sum them
        dist += _mm256_extract_epi64(sad, 0) as u32;
        dist += _mm256_extract_epi64(sad, 1) as u32;
        dist += _mm256_extract_epi64(sad, 2) as u32;
        dist += _mm256_extract_epi64(sad, 3) as u32;

        i += 32;
    }

    // Handle remaining bytes with scalar POPCNT
    while i + 8 <= len {
        let x = (a.as_ptr().add(i) as *const u64).read_unaligned();
        let y = (b.as_ptr().add(i) as *const u64).read_unaligned();
        dist += (x ^ y).count_ones();
        i += 8;
    }
    while i < len {
        dist += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    dist
}

/// NEON Hamming distance: processes 16 bytes (128 bits) per iteration.
/// ~2x faster than scalar u64 POPCNT on ARM (Apple M-series, Raspberry Pi 4+).
#[cfg(target_arch = "aarch64")]
unsafe fn hamming_neon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let mut dist: u32 = 0;
    let len = a.len();
    let mut i = 0;

    // Process 16 bytes at a time with NEON
    while i + 16 <= len {
        let va = vld1q_u8(a.as_ptr().add(i));
        let vb = vld1q_u8(b.as_ptr().add(i));
        let xor = veorq_u8(va, vb);

        // ARM has native vcntq_u8 (popcount per byte)
        let cnt = vcntq_u8(xor);

        // Sum all bytes: vaddlvq gives total across all lanes
        dist += vaddlvq_u8(cnt) as u32;

        i += 16;
    }

    // Handle remaining bytes with scalar
    while i + 8 <= len {
        let x = (a.as_ptr().add(i) as *const u64).read_unaligned();
        let y = (b.as_ptr().add(i) as *const u64).read_unaligned();
        dist += (x ^ y).count_ones();
        i += 8;
    }
    while i < len {
        dist += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    dist
}

/// Scalar fallback: processes 8 bytes at a time via u64 POPCNT.
/// Works on all platforms. The compiler lowers count_ones() to hardware POPCNT
/// when available.
#[inline(always)]
fn hamming_scalar(a: &[u8], b: &[u8]) -> u32 {
    let mut dist: u32 = 0;
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let x = unsafe { (a.as_ptr().add(i) as *const u64).read_unaligned() };
        let y = unsafe { (b.as_ptr().add(i) as *const u64).read_unaligned() };
        dist += (x ^ y).count_ones();
        i += 8;
    }
    while i < len {
        dist += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    dist
}

/// Dispatch to the best available SIMD kernel at runtime.
/// NOTE: On x86_64 with hardware POPCNT, the scalar path (count_ones → POPCNT)
/// is faster than AVX2 vpshufb lookup. AVX2 is kept for reference but NOT used.
#[inline(always)]
fn hamming(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON vcntq_u8 is a native byte popcount — always faster on ARM
        return unsafe { hamming_neon(a, b) };
    }

    // On x86_64: scalar count_ones() compiles to hardware POPCNT instruction,
    // which is faster than the AVX2 vpshufb nibble-lookup approach.
    #[allow(unreachable_code)]
    hamming_scalar(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Python-exposed functions
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn hamming_distances<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray1<u32>> {
    let q = query.as_slice().unwrap();
    let db = database.as_array();
    let n = db.shape()[0];

    let distances: Vec<u32> = (0..n)
        .map(|i| hamming(db.row(i).to_slice().unwrap(), q))
        .collect();

    PyArray1::from_vec_bound(py, distances)
}

#[pyfunction]
fn hamming_distances_parallel<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray1<u32>> {
    let q = query.as_slice().unwrap();
    let db = database.as_array();
    let n = db.shape()[0];

    let rows: Vec<&[u8]> = (0..n).map(|i| db.row(i).to_slice().unwrap()).collect();
    let distances: Vec<u32> = rows
        .par_iter()
        .map(|row| hamming(row, q))
        .collect();

    PyArray1::from_vec_bound(py, distances)
}

#[pyfunction]
fn hamming_topk<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let q = query.as_slice().unwrap();
    let db = database.as_array();
    let n = db.shape()[0];
    let k = k.min(n);

    let mut dist_idx: Vec<(u32, usize)> = (0..n)
        .map(|i| (hamming(db.row(i).to_slice().unwrap(), q), i))
        .collect();

    dist_idx.select_nth_unstable_by_key(k - 1, |&(d, _)| d);
    dist_idx[..k].sort_unstable_by_key(|&(d, _)| d);

    let indices: Vec<u64> = dist_idx[..k].iter().map(|&(_, i)| i as u64).collect();
    PyArray1::from_vec_bound(py, indices)
}

#[pyfunction]
fn hamming_topk_parallel<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
    k: usize,
) -> Bound<'py, PyArray1<u64>> {
    let q = query.as_slice().unwrap();
    let db = database.as_array();
    let n = db.shape()[0];
    let k = k.min(n);
    if k == 0 {
        return PyArray1::from_vec_bound(py, Vec::<u64>::new());
    }

    // Coarse-grained parallelism: split the database into a few contiguous
    // blocks per worker (not one task per row). Each block is scanned
    // sequentially — good cache/prefetch behaviour — and reduced to its own
    // local top-k. We then merge the (few) per-block top-k lists serially.
    // This keeps the parallel portion ~100% of the work; the old version did
    // an O(N) serial select over all N afterwards, which capped speedup.
    let n_threads = rayon::current_num_threads().max(1);
    // ~4 blocks per thread → good load balance without excess overhead.
    let n_blocks = (n_threads * 4).max(1);
    let block = n.div_ceil(n_blocks).max(1);

    let mut merged: Vec<(u32, u64)> = (0..n)
        .step_by(block)
        .collect::<Vec<_>>()
        .par_iter()
        .flat_map(|&start| {
            let end = (start + block).min(n);
            let mut local: Vec<(u32, u64)> = (start..end)
                .map(|i| (hamming(db.row(i).to_slice().unwrap(), q), i as u64))
                .collect();
            let kk = k.min(local.len());
            if kk > 0 {
                local.select_nth_unstable_by_key(kk - 1, |&(d, _)| d);
                local.truncate(kk);
            }
            local
        })
        .collect();

    // Final merge over the small set of per-block candidates (≈ n_blocks * k).
    let kk = k.min(merged.len());
    merged.select_nth_unstable_by_key(kk - 1, |&(d, _)| d);
    merged[..kk].sort_unstable_by_key(|&(d, _)| d);

    let indices: Vec<u64> = merged[..kk].iter().map(|&(_, i)| i).collect();
    PyArray1::from_vec_bound(py, indices)
}

/// Cache-block target in bytes. We tile the database so one block of codes
/// stays resident in L2 while every query in the chunk is scored against it —
/// the database is streamed from RAM ~once per thread instead of once per
/// query. This sidesteps the memory-bandwidth wall that caps the naive
/// "parallelize over queries" approach once the DB exceeds last-level cache.
const BLOCK_BYTES: usize = 256 * 1024;

/// Reduce a per-query candidate buffer to the k smallest (sorted ascending by
/// distance) and write their indices into `out` (len == k).
#[inline]
fn finalize(buf: &mut Vec<(u32, u64)>, k: usize, out: &mut [u64]) {
    let kk = k.min(buf.len());
    if kk == 0 {
        return;
    }
    buf.select_nth_unstable_by_key(kk - 1, |&(d, _)| d);
    buf[..kk].sort_unstable_by_key(|&(d, _)| d);
    for (slot, &(_, idx)) in out.iter_mut().zip(buf[..kk].iter()) {
        *slot = idx;
    }
}

/// Score a chunk of queries against the whole database with cache-blocking.
///
/// Loop order is (DB block) → (query) → (row within block): the block is
/// loaded once and reused across every query in the chunk while it's hot in
/// cache. Each query keeps its own bounded candidate buffer + running k-th-best
/// threshold so work stays ~O(N) per query without an O(N) materialization.
/// Writes query qi's k indices into out[qi*k .. qi*k + k].
fn process_chunk(
    q_rows: &[&[u8]],
    db: &numpy::ndarray::ArrayView2<u8>,
    k: usize,
    block_rows: usize,
    out: &mut [u64],
) {
    let n = db.shape()[0];
    let qn = q_rows.len();
    if qn == 0 || k == 0 {
        return;
    }
    let mut bufs: Vec<Vec<(u32, u64)>> =
        (0..qn).map(|_| Vec::with_capacity(4 * k)).collect();
    let mut worst: Vec<u32> = vec![u32::MAX; qn];

    let mut bstart = 0;
    while bstart < n {
        let bend = (bstart + block_rows).min(n);
        for (qi, q) in q_rows.iter().enumerate() {
            let buf = &mut bufs[qi];
            let w = worst[qi];
            for r in bstart..bend {
                let d = hamming(db.row(r).to_slice().unwrap(), q);
                // Always fill until we have k; after that only keep contenders.
                if buf.len() < k || d <= w {
                    buf.push((d, r as u64));
                }
            }
            // Trim once the buffer drifts past k; refresh the threshold to the
            // current k-th smallest distance.
            if buf.len() >= 2 * k {
                buf.select_nth_unstable_by_key(k - 1, |&(d, _)| d);
                buf.truncate(k);
                worst[qi] = buf.iter().map(|&(d, _)| d).max().unwrap_or(u32::MAX);
            }
        }
        bstart = bend;
    }

    for (qi, buf) in bufs.iter_mut().enumerate() {
        finalize(buf, k, &mut out[qi * k..qi * k + k]);
    }
}

/// Batched top-k: process many queries in a single FFI call.
///
/// Returns a flat (m * k) array of database indices, row-major: query 0's
/// top-k first, then query 1's, etc. Reshape to (m, k) on the Python side.
///
/// With `parallel=True` the query set is split into chunks (≈ a couple per
/// core); each thread cache-blocks the database for its chunk (see
/// `process_chunk`). This is the throughput path — measure it as QPS, not
/// single-query latency.
#[pyfunction]
#[pyo3(signature = (queries, database, k, parallel=true))]
fn hamming_topk_batch<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, u8>,
    database: PyReadonlyArray2<'py, u8>,
    k: usize,
    parallel: bool,
) -> Bound<'py, PyArray1<u64>> {
    let q = queries.as_array();
    let db = database.as_array();
    let m = q.shape()[0];
    let n = db.shape()[0];
    let code_len = db.shape()[1].max(1);
    let k = k.min(n);

    let mut out = vec![0u64; m * k];
    if k == 0 || m == 0 {
        return PyArray1::from_vec_bound(py, out);
    }

    let block_rows = (BLOCK_BYTES / code_len).max(1);
    let q_rows: Vec<&[u8]> = (0..m).map(|i| q.row(i).to_slice().unwrap()).collect();

    if parallel && m > 1 {
        let n_threads = rayon::current_num_threads().max(1);
        // ~2 chunks per thread → disjoint query ranges, no merge, no locks.
        let n_chunks = (n_threads * 2).min(m).max(1);
        let chunk = m.div_ceil(n_chunks);
        out.par_chunks_mut(chunk * k)
            .enumerate()
            .for_each(|(ci, out_slice)| {
                let qs = ci * chunk;
                let qe = (qs + chunk).min(m);
                if qs < qe {
                    process_chunk(&q_rows[qs..qe], &db, k, block_rows, out_slice);
                }
            });
    } else {
        process_chunk(&q_rows[..], &db, k, block_rows, &mut out);
    }

    PyArray1::from_vec_bound(py, out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_distances, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distances_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_batch, m)?)?;
    Ok(())
}
