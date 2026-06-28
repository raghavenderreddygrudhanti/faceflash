"""FaceFlash — face search in 3MB. PCA+ITQ binary codes + SIMD Hamming scan."""

__version__ = "0.1.1"

import logging as _logging
_logging.getLogger("faceflash").addHandler(_logging.NullHandler())

from faceflash.engine import FaceFlash
from faceflash.index import FaceIndex
from faceflash.pca_quantize import PCABinaryQuantizer, backend_info

__all__ = ["FaceFlash", "FaceIndex", "PCABinaryQuantizer", "backend_info"]
