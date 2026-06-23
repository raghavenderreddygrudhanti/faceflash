"""
FaceFlash — Sub-linear face retrieval at million scale.

Search 1 million faces in <5ms using binary vector quantization.
Built on TurboVec for hardware-optimized binary search.

Usage:
    from faceflash import FaceFlash

    ff = FaceFlash()
    ff.index("photos/")              # Index a folder of faces
    result = ff.search("query.jpg")  # Find match in milliseconds
"""

__version__ = "0.1.0"

from faceflash.engine import FaceFlash

__all__ = ["FaceFlash"]
