"""
FaceFlash — Fast face retrieval via PCA+ITQ binary quantization.

~10x faster than DeepFace/InsightFace search. 48x less committed memory
than HNSW at equal recall. Best at 10K–100K scale.

Usage:
    from faceflash import FaceFlash

    ff = FaceFlash()
    ff.register("Alice", "alice.jpg")
    ff.register("Bob", "bob.jpg")
    result = ff.search("query.jpg")
"""

__version__ = "0.1.0"

from faceflash.engine import FaceFlash
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info

__all__ = ["FaceFlash", "PCABinaryQuantizer", "backend_info"]
