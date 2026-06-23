//! FaceFlash Core — Rust binary search engine.
//! Hardware-accelerated Hamming distance via explicit u64 POPCNT + Rayon parallelism.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute Hamming distance between two packed binary vectors.
/// Uses u64 POPCNT for explicit hardware acceleration (not relying on autovectorization).
#[inline(always)]
fn hamming_u64(a: &[u8], b: &[u8]) -> u32 {
    let n_u64 = a.len() / 8;
    let a_u64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u64, n_u64) };
    let b_u64 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u64, n_u64) };

    let mut dist: u32 = 0;
    for i in 0..n_u64 {
        dist += (a_u64[i] ^ b_u64[i]).count_ones();
    }
    // Handle remaining bytes (if length not multiple of 8)
    for i in (n_u64 * 8)..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

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
        .map(|i| hamming_u64(db.row(i).to_slice().unwrap(), q))
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
        .map(|row| hamming_u64(row, q))
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
        .map(|i| (hamming_u64(db.row(i).to_slice().unwrap(), q), i))
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

    let rows: Vec<&[u8]> = (0..n).map(|i| db.row(i).to_slice().unwrap()).collect();
    let mut dist_idx: Vec<(u32, usize)> = rows
        .par_iter()
        .enumerate()
        .map(|(i, row)| (hamming_u64(row, q), i))
        .collect();

    dist_idx.select_nth_unstable_by_key(k - 1, |&(d, _)| d);
    dist_idx[..k].sort_unstable_by_key(|&(d, _)| d);

    let indices: Vec<u64> = dist_idx[..k].iter().map(|&(_, i)| i as u64).collect();
    PyArray1::from_vec_bound(py, indices)
}

#[pymodule]
fn faceflash_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_distances, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distances_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_topk_parallel, m)?)?;
    Ok(())
}
