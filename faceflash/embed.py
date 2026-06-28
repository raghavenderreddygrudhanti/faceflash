"""ArcFace ONNX embedding. Downloads model on first use (~250MB)."""

import logging
import urllib.request
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/onnx-community/arcface/resolve/main/arcface_r100.onnx"
MODEL_DIR = Path.home() / ".faceflash" / "models"
MODEL_PATH = MODEL_DIR / "arcface_r100.onnx"


def ensure_model():
    """Download model if missing."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ArcFace model (~250MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    return MODEL_PATH


class FaceEmbedder:
    """ArcFace ONNX inference. Picks GPU if available."""

    def __init__(self):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for embeddings. Install one of:\n"
                "  pip install faceflash[cpu]   # CPU\n"
                "  pip install faceflash[gpu]   # CUDA"
            ) from e
        model_path = ensure_model()
        # Prefer GPU → CPU fallback
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(model_path),
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 112, 112]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize to 112x112, normalize to [-1,1], HWC→NCHW."""
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
        """512-dim L2-normalized embedding from an aligned face."""
        inp = self.preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: inp})
        embedding = outputs[0][0]
        # L2 normalize (guard against degenerate zero-norm)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            raise ValueError("Face embedding has zero norm (degenerate face detection)")
        embedding = embedding / norm
        return embedding

    def embed_batch(self, face_images: list) -> np.ndarray:
        """Batch embed. Just loops for now — no real batching in ONNX CPU."""
        embeddings = []
        for img in face_images:
            embeddings.append(self.embed(img))
        return np.array(embeddings)
