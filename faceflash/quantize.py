"""
Embedding quantization — converts 512-dim float vectors to compact binary.
This is where TurboVec's quantization is applied.
The key insight: face embeddings have high redundancy and can be
compressed to binary with minimal accuracy loss.
"""

import numpy as np


def float_to_binary(embedding: np.ndarray, projection: np.ndarray = None) -> np.ndarray:
    """
    Convert a normalized float embedding to binary vector.

    Method: Random hyperplane projection (SimHash).
    Projects embedding onto random planes, takes sign.
    Preserves cosine similarity in Hamming space.

    512 floats (2048 bytes) → 512 bits (64 bytes) = 32x compression.
    """
    if projection is not None:
        projected = embedding @ projection.T
        return (projected > 0).astype(np.uint8)
    # Fallback: sign-based (works well for structured embeddings like ArcFace)
    return (embedding > 0).astype(np.uint8)


def float_to_binary_batch(embeddings: np.ndarray, projection: np.ndarray = None) -> np.ndarray:
    """Quantize a batch of embeddings to binary."""
    if projection is not None:
        projected = embeddings @ projection.T
        return (projected > 0).astype(np.uint8)
    return (embeddings > 0).astype(np.uint8)


def pack_binary(binary_vec: np.ndarray) -> np.ndarray:
    """Pack binary vector into uint8 bytes (8 bits per byte)."""
    # Pad to multiple of 8
    pad_len = (8 - len(binary_vec) % 8) % 8
    if pad_len > 0:
        binary_vec = np.concatenate([binary_vec, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(binary_vec)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Compute Hamming distance between two packed binary vectors."""
    xor = np.bitwise_xor(a, b)
    return sum(bin(byte).count('1') for byte in xor)


def hamming_distance_batch(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute Hamming distances between query and all database vectors.
    Fully vectorized with NumPy — no Python loops.
    """
    # XOR all rows with query
    xor = np.bitwise_xor(database, query)
    # Popcount via lookup table (fast, pure numpy)
    # Unpack bits and sum per row
    bits = np.unpackbits(xor, axis=1)
    distances = bits.sum(axis=1)
    return distances
