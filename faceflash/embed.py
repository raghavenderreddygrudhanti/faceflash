"""
Face embedding — converts a face image to a 512-dim vector using ArcFace (ONNX).
Downloads the model automatically on first use.
"""

import os
import urllib.request
from pathlib import Path
import numpy as np

MODEL_URL = "https://huggingface.co/onnx-community/arcface/resolve/main/arcface_r100.onnx"
MODEL_DIR = Path.home() / ".faceflash" / "models"
MODEL_PATH = MODEL_DIR / "arcface_r100.onnx"


def ensure_model():
    """Download ArcFace ONNX model if not present."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading ArcFace model (~250MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}")
    return MODEL_PATH


class FaceEmbedder:
    """ArcFace embedding model (ONNX, runs on CPU)."""

    def __init__(self):
        import onnxruntime as ort
        model_path = ensure_model()
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 112, 112]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess face image for ArcFace: resize to 112x112, normalize."""
        from PIL import Image

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        # Resize to 112x112
        img = img.resize((112, 112), Image.BILINEAR)
        img = np.array(img, dtype=np.float32)

        # Normalize to [-1, 1]
        img = (img - 127.5) / 127.5

        # HWC -> CHW -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def embed(self, face_img: np.ndarray) -> np.ndarray:
        """Get 512-dim embedding for an aligned face image."""
        inp = self.preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: inp})
        embedding = outputs[0][0]
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(self, face_images: list) -> np.ndarray:
        """Get embeddings for a batch of face images."""
        embeddings = []
        for img in face_images:
            embeddings.append(self.embed(img))
        return np.array(embeddings)
