"""ArcFace ONNX embedding. Downloads model on first use (~166MB)."""

import logging
import urllib.request
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# w600k_r50 — the exact model every benchmark number was measured with.
MODEL_URL = ("https://huggingface.co/public-data/insightface/resolve/main/"
             "models/buffalo_l/w600k_r50.onnx")
MODEL_DIR = Path.home() / ".faceflash" / "models"
MODEL_PATH = MODEL_DIR / "w600k_r50.onnx"
# Older installs (and the RunPod script) cached the same weights under a
# misleading name; reuse them instead of re-downloading 166MB.
_LEGACY_MODEL_PATH = MODEL_DIR / "arcface_r100.onnx"


def ensure_model():
    """Download model if missing."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    if _LEGACY_MODEL_PATH.exists():
        return _LEGACY_MODEL_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ArcFace model (~166MB)...")
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
        """Embed many aligned faces in one ONNX call.

        Stacks preprocessed images into one (N, 3, 112, 112) tensor. Falls
        back to per-image inference if the model's batch axis is fixed at 1.
        """
        if not face_images:
            return np.empty((0, 512), dtype=np.float32)

        batch_dim = self.input_shape[0]
        if isinstance(batch_dim, int) and batch_dim == 1:
            return np.array([self.embed(img) for img in face_images])

        # CoreML compiles per input shape, so varying batch sizes trigger
        # recompilation and stacked inference comes out ~5x SLOWER than the
        # loop (measured on M-series). Batch only on CPU/CUDA providers.
        if self.session.get_providers()[0] == "CoreMLExecutionProvider":
            return np.array([self.embed(img) for img in face_images])

        inp = np.concatenate([self.preprocess(img) for img in face_images], axis=0)
        try:
            out = self.session.run(None, {self.input_name: inp})[0]
        except Exception:
            # Some exports declare a symbolic batch dim but only accept N=1
            return np.array([self.embed(img) for img in face_images])

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if np.any(norms < 1e-10):
            raise ValueError("Face embedding has zero norm (degenerate face detection)")
        return out / norms
