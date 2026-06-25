"""
Face detection + 5-point alignment (SCRFD / "RetinaFace" det_10g, ONNX).

ArcFace expects a tightly aligned 112x112 crop. This module detects faces with
the lightweight SCRFD det_10g model and warps each face to the canonical ArcFace
template using a similarity transform — matching how the benchmark embeddings
were produced. Needs only onnxruntime + opencv (already dependencies); no
insightface package required.

Falls back gracefully: if the detector or its model is unavailable, callers use
the OpenCV Haar + center-crop path in detect.py.
"""
import urllib.request
from pathlib import Path

import numpy as np

MODEL_DIR = Path.home() / ".faceflash" / "models"
DET_PATH = MODEL_DIR / "det_10g.onnx"
DET_URL = "https://huggingface.co/immich-app/buffalo_l/resolve/main/detection/model.onnx"

# Canonical 5-point template ArcFace was trained on (112x112).
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _ensure_det_model():
    if DET_PATH.exists():
        return DET_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("  Downloading SCRFD face detector (~17MB)...")
    urllib.request.urlretrieve(DET_URL, DET_PATH)
    return DET_PATH


def _distance2points(anchor_centers, distance):
    """Decode SCRFD distance predictions (bbox or kps) to absolute points."""
    pts = []
    for i in range(0, distance.shape[1], 2):
        px = anchor_centers[:, 0] + distance[:, i]
        py = anchor_centers[:, 1] + distance[:, i + 1]
        pts.append(px)
        pts.append(py)
    return np.stack(pts, axis=-1)


def _distance2bbox(anchor_centers, distance):
    x1 = anchor_centers[:, 0] - distance[:, 0]
    y1 = anchor_centers[:, 1] - distance[:, 1]
    x2 = anchor_centers[:, 0] + distance[:, 2]
    y2 = anchor_centers[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


class FaceDetector:
    """SCRFD det_10g detector → bounding boxes + 5 landmarks."""

    _STRIDES = [8, 16, 32]
    _NUM_ANCHORS = 2

    def __init__(self, det_thresh: float = 0.5, nms_thresh: float = 0.4,
                 input_size: int = 640):
        import onnxruntime as ort
        model_path = _ensure_det_model()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if "CUDAExecutionProvider" in ort.get_available_providers() \
            else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self._anchor_cache = {}

    def _anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key not in self._anchor_cache:
            ay, ax = np.mgrid[0:height, 0:width]
            centers = np.stack([ax, ay], axis=-1).astype(np.float32) * stride
            centers = centers.reshape(-1, 2)
            if self._NUM_ANCHORS > 1:
                centers = np.repeat(centers, self._NUM_ANCHORS, axis=0)
            self._anchor_cache[key] = centers
        return self._anchor_cache[key]

    def detect(self, img_rgb: np.ndarray):
        """Return (boxes Nx5 [x1,y1,x2,y2,score], landmarks Nx5x2) in image coords."""
        import cv2
        h0, w0 = img_rgb.shape[:2]
        size = self.input_size
        # resize keeping aspect ratio, pad to size x size
        scale = min(size / w0, size / h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(img_rgb, (nw, nh))
        det_img = np.zeros((size, size, 3), dtype=np.uint8)
        det_img[:nh, :nw] = resized

        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128.0, (size, size),
                                     (127.5, 127.5, 127.5), swapRB=False)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        fmc = len(self._STRIDES)

        scores_list, boxes_list, kps_list = [], [], []
        for i, stride in enumerate(self._STRIDES):
            scores = outs[i].reshape(-1)
            bbox_preds = outs[i + fmc].reshape(-1, 4) * stride
            kps_preds = outs[i + fmc * 2].reshape(-1, 10) * stride
            feat = size // stride
            centers = self._anchor_centers(feat, feat, stride)
            keep = scores >= self.det_thresh
            if not keep.any():
                continue
            boxes = _distance2bbox(centers, bbox_preds)[keep]
            kps = _distance2points(centers, kps_preds)[keep].reshape(-1, 5, 2)
            scores_list.append(scores[keep])
            boxes_list.append(boxes)
            kps_list.append(kps)

        if not scores_list:
            return np.empty((0, 5)), np.empty((0, 5, 2))

        scores = np.concatenate(scores_list)
        boxes = np.concatenate(boxes_list) / scale
        kps = np.concatenate(kps_list) / scale
        order = scores.argsort()[::-1]
        boxes, kps, scores = boxes[order], kps[order], scores[order]
        keep = self._nms(np.hstack([boxes, scores[:, None]]))
        boxes = np.hstack([boxes[keep], scores[keep, None]])
        return boxes, kps[keep]

    def _nms(self, dets):
        x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= self.nms_thresh]
        return keep


def align_face(img_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Warp a face to the 112x112 ArcFace template from its 5 landmarks.

    Uses a similarity transform (rotation + uniform scale + translation) via
    OpenCV — no scikit-image dependency.
    """
    import cv2
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0.0)


_DETECTOR = None


def get_detector():
    """Lazily build (and cache) the detector; None if it can't be created."""
    global _DETECTOR
    if _DETECTOR is None:
        try:
            _DETECTOR = FaceDetector()
        except Exception:
            _DETECTOR = False  # sentinel: tried and failed
    return _DETECTOR or None


def detect_and_align(img_rgb: np.ndarray):
    """Detect the largest face and return an aligned 112x112 RGB crop, or None."""
    det = get_detector()
    if det is None:
        return None
    boxes, kps = det.detect(img_rgb)
    if len(boxes) == 0:
        return None
    # Pick the most prominent face: large AND central (insightface heuristic).
    # Pure "largest area" can lose to a spurious off-center detection.
    h, w = img_rgb.shape[:2]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    dist2 = (cx - w / 2.0) ** 2 + (cy - h / 2.0) ** 2
    best = int((areas - 2.0 * dist2).argmax())
    return align_face(img_rgb, kps[best])
