"""Age, gender, and emotion estimation from aligned face crops.

Uses lightweight ONNX models — no training required. Downloads on first use.
All models run on the same 112x112 (or 224x224) aligned face crop that
FaceFlash already produces, so there's no extra detection cost.

Models:
  - Age/Gender: InsightFace genderage (~1.3MB)
  - Emotion: FER+ from ONNX model zoo (~8MB), 8 classes
"""

import logging
import urllib.request
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".faceflash" / "models"

# InsightFace genderage model (predicts age as float + gender as binary)
GENDERAGE_URL = "https://huggingface.co/deepghs/insightface/resolve/main/models/buffalo_l/genderage.onnx"
GENDERAGE_PATH = MODEL_DIR / "genderage.onnx"

# Emotion model (FER+ / ONNX zoo variant)
EMOTION_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-2/resolve/main/model.onnx"
EMOTION_PATH = MODEL_DIR / "emotion_ferplus.onnx"

EMOTIONS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


def _download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        logger.info("Downloading %s ...", path.name)
        urllib.request.urlretrieve(url, path)


class AgeGenderEstimator:
    """Age + gender from a face crop. InsightFace genderage model."""

    def __init__(self):
        import onnxruntime as ort
        _download(GENDERAGE_URL, GENDERAGE_PATH)
        self.session = ort.InferenceSession(
            str(GENDERAGE_PATH), providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = inp.shape[2]  # usually 96 or 112

    def predict(self, face_crop: np.ndarray) -> dict:
        """
        face_crop: RGB uint8 or float, any size (will be resized).
        Returns {"age": int, "gender": "M" or "F"}
        """
        from PIL import Image

        img = Image.fromarray(face_crop.astype(np.uint8)) if face_crop.dtype != np.uint8 else Image.fromarray(face_crop)
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        # normalize same as insightface expects
        arr = (arr - 127.5) / 127.5
        arr = arr.transpose(2, 0, 1)[None]  # NCHW

        out = self.session.run(None, {self.input_name: arr})[0][0]
        # InsightFace genderage output: [gender_logit, age_value]
        # gender: out[0] > 0 means female in some versions, varies
        # age: out[2] or out[-1] scaled
        # The buffalo_l genderage model outputs shape (3,): [gender, age_raw...]
        # Handling both common formats:
        if len(out) == 3:
            gender = "F" if out[0] > 0 else "M"
            age = int(np.clip(out[1] * 100, 0, 100))
        elif len(out) >= 2:
            gender = "F" if out[0] > 0 else "M"
            age = int(np.clip(out[1], 0, 100))
        else:
            gender = "unknown"
            age = -1

        return {"age": age, "gender": gender}


class EmotionEstimator:
    """Facial emotion recognition. 8 classes (FER+)."""

    def __init__(self):
        import onnxruntime as ort
        _download(EMOTION_URL, EMOTION_PATH)
        self.session = ort.InferenceSession(
            str(EMOTION_PATH), providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # FER+ expects 64x64 grayscale typically, or 1x1x64x64
        self.input_shape = inp.shape

    def predict(self, face_crop: np.ndarray) -> dict:
        """
        Returns {"emotion": "happy", "scores": {"happy": 0.8, "sad": 0.1, ...}}
        """
        from PIL import Image

        img = Image.fromarray(face_crop.astype(np.uint8) if face_crop.dtype != np.uint8 else face_crop)
        # FER+ model typically expects 64x64 grayscale
        img = img.convert("L").resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Reshape for model input — usually (1, 1, 64, 64)
        arr = arr.reshape(1, 1, 64, 64)

        out = self.session.run(None, {self.input_name: arr})[0][0]
        # Softmax
        exp = np.exp(out - out.max())
        probs = exp / exp.sum()

        top_idx = int(probs.argmax())
        scores = {EMOTIONS[i]: round(float(probs[i]), 3) for i in range(min(len(EMOTIONS), len(probs)))}

        return {
            "emotion": EMOTIONS[top_idx] if top_idx < len(EMOTIONS) else "unknown",
            "confidence": round(float(probs[top_idx]), 3),
            "scores": scores,
        }


class FaceAttributes:
    """Combined age + gender + emotion. Lazy-loads models on first call."""

    def __init__(self):
        self._age_gender = None
        self._emotion = None

    @property
    def age_gender(self):
        if self._age_gender is None:
            self._age_gender = AgeGenderEstimator()
        return self._age_gender

    @property
    def emotion(self):
        if self._emotion is None:
            self._emotion = EmotionEstimator()
        return self._emotion

    def analyze(self, face_crop: np.ndarray) -> dict:
        """Run all attribute models on a face crop.

        Returns: {"age": 28, "gender": "F", "emotion": "happy", "emotion_confidence": 0.82}
        """
        ag = self.age_gender.predict(face_crop)
        em = self.emotion.predict(face_crop)
        return {
            "age": ag["age"],
            "gender": ag["gender"],
            "emotion": em["emotion"],
            "emotion_confidence": em["confidence"],
        }

    def is_child(self, face_crop: np.ndarray, threshold: int = 16) -> bool:
        """Quick check: is this face likely a child (under threshold age)?"""
        ag = self.age_gender.predict(face_crop)
        return ag["age"] < threshold
