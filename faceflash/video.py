"""Scan a video for registered faces and report when each person appears.

Samples frames at a fixed rate, detects every face per frame, embeds them,
and searches the gallery. Consecutive hits for the same person are merged into
appearance intervals so the output is "Alice: 0:12-0:18, 1:45-2:03" rather
than a raw list of frame timestamps.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from faceflash.detect import detect_faces

logger = logging.getLogger(__name__)


@dataclass
class Appearance:
    """One continuous stretch where a person is on screen."""
    name: str
    start_s: float
    end_s: float
    best_confidence: float
    hits: int  # number of sampled frames they matched in

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "start_timestamp": _fmt(self.start_s),
            "end_timestamp": _fmt(self.end_s),
            "best_confidence": round(self.best_confidence, 4),
            "hits": self.hits,
        }


def _fmt(seconds: float) -> str:
    """Seconds -> H:MM:SS or M:SS."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _open_video(path: str):
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "Video scanning needs OpenCV: pip install opencv-python"
        ) from e
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap, cv2


def scan_video(
    engine,
    video_path: str,
    sample_fps: float = 1.0,
    threshold: float = 0.4,
    merge_gap_s: float = 2.0,
    names: Optional[List[str]] = None,
    progress: bool = True,
) -> Dict[str, List[Appearance]]:
    """
    Find every registered person in a video.

    Args:
        engine: a FaceFlash instance with faces already registered
        video_path: path to the video file
        sample_fps: how many frames per second to check (1.0 = one frame/sec).
                    Lower = faster scan, coarser timeline.
        threshold: min cosine similarity to count as a match
        merge_gap_s: hits within this gap are merged into one appearance.
                     Bridges short detection dropouts (blink, turn, occlusion).
        names: optionally restrict to specific people (None = all registered)
        progress: show a tqdm bar

    Returns:
        {name: [Appearance, ...]} — appearances sorted by start time per person.
    """
    cap, cv2 = _open_video(video_path)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_stride = max(1, int(round(src_fps / sample_fps)))

    name_filter = set(names) if names else None

    # Per-person list of (timestamp, confidence) hits, built as we scan.
    raw_hits: Dict[str, List[tuple]] = {}

    sampled = (total_frames // frame_stride) if total_frames else None
    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=sampled, desc="Scanning video", unit="frame")
        except ImportError:
            pass

    frame_idx = 0
    try:
        while True:
            ok = cap.grab()  # cheap: advance without decoding
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    timestamp = frame_idx / src_fps
                    # OpenCV gives BGR; FaceFlash/ArcFace expect RGB
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _process_frame(engine, rgb, timestamp, threshold,
                                   name_filter, raw_hits)
                if bar:
                    bar.update(1)
            frame_idx += 1
    finally:
        cap.release()
        if bar:
            bar.close()

    # Merge consecutive hits into appearance intervals.
    return {
        name: _merge_hits(name, hits, merge_gap_s, sample_fps)
        for name, hits in raw_hits.items()
    }


def _process_frame(engine, rgb, timestamp, threshold, name_filter, raw_hits):
    """Detect + identify every face in one frame, record matches."""
    crops = detect_faces(rgb)
    if not crops:
        return

    # Embed all faces in the frame, then one batched search.
    embeddings = []
    for crop in crops:
        try:
            embeddings.append(engine.embedder.embed(crop))
        except Exception:
            continue  # skip degenerate crops (blurry, no face)
    if not embeddings:
        return

    emb_array = np.array(embeddings, dtype=np.float32)
    results = engine.index.search_batch(emb_array, k=1,
                                        n_candidates=engine.n_candidates)

    for matches in results:
        if not matches:
            continue
        name, sim, _ = matches[0]
        if sim < threshold:
            continue
        if name_filter and name not in name_filter:
            continue
        raw_hits.setdefault(name, []).append((timestamp, float(sim)))


def _merge_hits(name, hits, merge_gap_s, sample_fps) -> List[Appearance]:
    """Group (timestamp, conf) hits into continuous appearances.

    A new appearance starts when the gap to the previous hit exceeds
    merge_gap_s. The end is extended by one sample interval so a single
    isolated hit still has a sensible (non-zero) duration.
    """
    if not hits:
        return []
    hits.sort(key=lambda h: h[0])
    sample_interval = 1.0 / sample_fps

    appearances: List[Appearance] = []
    start, last, best, count = hits[0][0], hits[0][0], hits[0][1], 1

    for ts, conf in hits[1:]:
        if ts - last <= merge_gap_s:
            last = ts
            best = max(best, conf)
            count += 1
        else:
            appearances.append(Appearance(name, start, last + sample_interval, best, count))
            start, last, best, count = ts, ts, conf, 1

    appearances.append(Appearance(name, start, last + sample_interval, best, count))
    return appearances
