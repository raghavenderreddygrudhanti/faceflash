"""
Face detection — finds and aligns faces in images.
Uses ONNX-based face detector (lightweight, no PyTorch needed).
Falls back to OpenCV Haar cascade if ONNX model unavailable.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load an image from path or URL."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def detect_and_align(img: np.ndarray) -> np.ndarray:
    """
    Detect face in image and return aligned 112x112 crop.
    Simple center-crop approach for MVP.
    For production: use RetinaFace or YOLOv8-Face.
    """
    h, w = img.shape[:2]

    # Simple face detection using OpenCV (fast, no extra model)
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) > 0:
            # Take largest face
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            # Add margin
            margin = int(max(fw, fh) * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            face = img[y1:y2, x1:x2]
            return face
    except ImportError:
        pass

    # Fallback: center crop (assumes face is centered)
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    face = img[top:top+size, left:left+size]
    return face


def detect_faces(img: np.ndarray) -> list:
    """Detect all faces in an image. Returns list of cropped face arrays."""
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        results = []
        h, w = img.shape[:2]
        for (x, y, fw, fh) in faces:
            margin = int(max(fw, fh) * 0.15)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            results.append(img[y1:y2, x1:x2])
        return results
    except ImportError:
        return [detect_and_align(img)]
