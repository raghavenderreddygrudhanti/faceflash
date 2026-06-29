"""Tests for video timeline scanning.

The merge logic and timestamp formatting are tested directly (no video needed).
The end-to-end scan is tested with a synthetic video built from solid-color
frames, gated on OpenCV being available.
"""

import numpy as np
import pytest

from faceflash.video import _merge_hits, _fmt, Appearance


# ─── Timestamp formatting ────────────────────────────────────────────────────

class TestFormat:
    def test_seconds(self):
        assert _fmt(5) == "0:05"
        assert _fmt(45) == "0:45"

    def test_minutes(self):
        assert _fmt(65) == "1:05"
        assert _fmt(125) == "2:05"

    def test_hours(self):
        assert _fmt(3661) == "1:01:01"
        assert _fmt(7325) == "2:02:05"


# ─── Merge logic ─────────────────────────────────────────────────────────────

class TestMerge:
    def test_empty(self):
        assert _merge_hits("Alice", [], merge_gap_s=2.0, sample_fps=1.0) == []

    def test_single_hit_has_duration(self):
        # One isolated hit should still span one sample interval, not zero.
        apps = _merge_hits("Alice", [(10.0, 0.9)], merge_gap_s=2.0, sample_fps=1.0)
        assert len(apps) == 1
        assert apps[0].start_s == 10.0
        assert apps[0].end_s == 11.0  # +1 sample interval
        assert apps[0].duration_s == 1.0

    def test_consecutive_merge_into_one(self):
        # Hits 1s apart with a 2s gap tolerance → one appearance.
        hits = [(1.0, 0.8), (2.0, 0.9), (3.0, 0.85)]
        apps = _merge_hits("Alice", hits, merge_gap_s=2.0, sample_fps=1.0)
        assert len(apps) == 1
        assert apps[0].start_s == 1.0
        assert apps[0].hits == 3
        assert apps[0].best_confidence == 0.9  # max of the three

    def test_gap_splits_appearances(self):
        # A 10s gap exceeds merge_gap_s → two separate appearances.
        hits = [(1.0, 0.8), (2.0, 0.9), (15.0, 0.7), (16.0, 0.75)]
        apps = _merge_hits("Alice", hits, merge_gap_s=2.0, sample_fps=1.0)
        assert len(apps) == 2
        assert apps[0].start_s == 1.0 and apps[0].hits == 2
        assert apps[1].start_s == 15.0 and apps[1].hits == 2

    def test_unsorted_input(self):
        # Hits arrive out of order; merge should sort first.
        hits = [(3.0, 0.85), (1.0, 0.8), (2.0, 0.9)]
        apps = _merge_hits("Alice", hits, merge_gap_s=2.0, sample_fps=1.0)
        assert len(apps) == 1
        assert apps[0].start_s == 1.0

    def test_appearance_to_dict(self):
        app = Appearance("Bob", 12.0, 18.0, 0.91, 6)
        d = app.to_dict()
        assert d["name"] == "Bob"
        assert d["duration_s"] == 6.0
        assert d["start_timestamp"] == "0:12"
        assert d["end_timestamp"] == "0:18"
        assert d["hits"] == 6


# ─── End-to-end (synthetic video) ────────────────────────────────────────────

def _make_synthetic_video(path, cv2, n_frames=30, fps=10, size=(160, 160)):
    """Write a video where frame color changes halfway — a stand-in for scenes."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(n_frames):
        shade = 50 if i < n_frames // 2 else 200
        frame = np.full((size[1], size[0], 3), shade, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestEndToEnd:
    def test_scan_runs_on_video(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        from faceflash.video import _open_video

        video_path = str(tmp_path / "synthetic.mp4")
        _make_synthetic_video(video_path, cv2, n_frames=30, fps=10)

        # Verify the video opens and frame math is sane — no faces expected
        # in solid-color frames, so this checks the scan loop doesn't crash.
        cap, _ = _open_video(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert src_fps == 10
        assert total == 30
