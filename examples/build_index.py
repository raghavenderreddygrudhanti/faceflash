"""Build an index from a folder of face images."""
from faceflash import FaceFlash

ff = FaceFlash()

# Register all faces from folder structure:
# employees/
#   Alice/photo1.jpg, photo2.jpg
#   Bob/photo1.jpg
result = ff.register_folder("employees/")
print(f"Registered: {result['registered']}, Errors: {result['errors']}")

# Save index for later use
ff.save("my_index/")
print("Index saved.")

# Load and search
ff2 = FaceFlash(index_path="my_index/")
result = ff2.search("query.jpg")
print(result)
