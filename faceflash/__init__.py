"""
FaceFlash — memory-efficient face retrieval via PCA+ITQ binary quantization.

100% Recall@1 (matching exact brute-force search) at 48-96x less memory than
HNSW, measured on MS1MV2 from 100K to 1M faces. CPU-only; AVX-512 VPOPCNTDQ
(x86) / NEON (ARM) accelerated.

Usage:
    from faceflash import FaceFlash

    ff = FaceFlash()
    ff.register("Alice", "alice.jpg")
    ff.register("Bob", "bob.jpg")
    result = ff.search("query.jpg")
"""

__version__ = "0.1.0"

# Library best practice (PEP/logging guide): attach a NullHandler so FaceFlash
# emits nothing unless the host application configures logging. Apps opt in via
# `logging.getLogger("faceflash").setLevel(logging.INFO)`.
import logging as _logging
_logging.getLogger("faceflash").addHandler(_logging.NullHandler())

from faceflash.engine import FaceFlash
from faceflash.index import FaceIndex
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info

__all__ = ["FaceFlash", "FaceIndex", "PCABinaryQuantizer", "backend_info"]
