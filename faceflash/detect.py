"""Face detection. SCRFD 5-point alignment preferred, Haar cascade fallback."""
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def detect_and_align(img: np.ndarray) -> np.ndarray:
    """Largest face → 112x112 aligned crop. SCRFD if available, else Haar."""
    try:
        from faceflash.align import detect_and_align as _rf_align
        aligned = _rf_align(img)
        if aligned is not None:
            return aligned
    except Exception:
        pass
    return _detect_haar_fallback(img)


def detect_faces(img: np.ndarray) -> list:
    """All faces in image → list of 112x112 aligned crops."""
    try:
        from faceflash.align import get_detector, align_face
        det = get_detector()
        if det is not None:
            boxes, kps = det.detect(img)
            if len(boxes):
                return [align_face(img, kps[i]) for i in range(len(boxes))]
    except Exception:
        pass
    return [_detect_haar_fallback(img)]


def _detect_haar_fallback(img: np.ndarray) -> np.ndarray:
    """Haar + center crop when SCRFD isn't available."""
    try:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            margin = int(max(fw, fh) * 0.2)
            h, w = img.shape[:2]
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            return img[y1:y2, x1:x2]
    except ImportError:
        pass

    # Final fallback: center crop
    h, w = img.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]
