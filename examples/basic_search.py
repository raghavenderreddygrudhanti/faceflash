"""Basic face search example."""
from faceflash import FaceFlash

ff = FaceFlash()

# Register faces
ff.register("Alice", "path/to/alice.jpg")
ff.register("Bob", "path/to/bob.jpg")

# Search
result = ff.search("path/to/query.jpg")
print(result)
