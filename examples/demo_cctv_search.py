"""
CCTV-style demo: build a multi-person video, search for known people.

Simulates surveillance footage where several people appear at different times.
We register each person using ONE photo, then build a video from DIFFERENT
photos of the same people (so it's real recognition, not image matching).
Then we scan and print who appeared when.

Run from repo root:
    PYTHONPATH=. .venv/bin/python examples/demo_cctv_search.py
"""

import os
import glob
import tempfile

import cv2
import numpy as np

from faceflash import FaceFlash

LFW = "data/lfw_funneled"
FRAME_SIZE = (320, 320)
FPS = 10


def photos(name, n):
    """Return up to n photo paths for a person."""
    files = sorted(glob.glob(os.path.join(LFW, name, "*.jpg")))
    return files[:n]


def build_cctv_video(out_path, scenes):
    """Build a video from a list of (photo_path, seconds_on_screen) scenes.

    Each scene shows one person's photo for the given duration. Between
    scenes we insert 1s of 'empty corridor' (gray frames) to create gaps.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, FRAME_SIZE)
    gray = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 110, dtype=np.uint8)

    for photo_path, seconds in scenes:
        if photo_path is None:  # empty corridor
            frame = gray
        else:
            img = cv2.imread(photo_path)
            frame = cv2.resize(img, FRAME_SIZE)
        for _ in range(int(seconds * FPS)):
            writer.write(frame)
    writer.release()


def main():
    if not os.path.isdir(LFW):
        print(f"LFW data not found at {LFW} — this demo needs the LFW dataset.")
        return

    print("=" * 64)
    print("FaceFlash CCTV Demo — find known people in surveillance footage")
    print("=" * 64)

    # People we'll look for. Register with photo #1, put them in the video
    # using LATER photos (different angle/lighting) — so it's real recognition.
    people = {
        "Colin_Powell": photos("Colin_Powell", 6),
        "George_W_Bush": photos("George_W_Bush", 6),
        "Tony_Blair": photos("Tony_Blair", 6),
    }

    ff = FaceFlash()
    print("\nRegistering known people (1 enrollment photo each):")
    for name, files in people.items():
        ff.register(name, files[0])   # enroll with first photo
        print(f"  ✓ {name}")

    # Build a 'CCTV timeline' using DIFFERENT photos of each person.
    # Story: Powell enters early, leaves. Corridor empty. Bush appears twice.
    # Blair appears once near the end. (using photos 2-5, not the enrolled #1)
    scenes = [
        (people["Colin_Powell"][1], 3),   # 0:00-0:03 Powell
        (None, 2),                         # 0:03-0:05 empty
        (people["George_W_Bush"][1], 3),   # 0:05-0:08 Bush
        (None, 1),                         # gap
        (people["Tony_Blair"][1], 2),      # 0:09-0:11 Blair
        (people["George_W_Bush"][2], 3),   # 0:11-0:14 Bush again
        (None, 1),
        (people["Colin_Powell"][2], 2),    # 0:15-0:17 Powell again
    ]

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "cctv.mp4")
        build_cctv_video(video_path, scenes)
        total_s = sum(s for _, s in scenes)
        print(f"\nBuilt {total_s}s of CCTV footage (3 people, multiple appearances)")
        print("Scanning at 2 fps...\n")

        timeline = ff.scan_video(video_path, sample_fps=2.0, threshold=0.4,
                                 merge_gap_s=1.5, progress=False)

    print("─" * 64)
    print("RESULTS — who appeared when")
    print("─" * 64)
    if not timeline:
        print("No known people detected.")
        return

    for name in sorted(timeline):
        apps = timeline[name]
        total = sum(a["duration_s"] for a in apps)
        print(f"\n{name}  ({len(apps)} appearance(s), {total:.0f}s on screen)")
        for a in apps:
            print(f"    {a['start_timestamp']} → {a['end_timestamp']}  "
                  f"(conf {a['best_confidence']:.2f})")

    print("\n" + "=" * 64)
    print("Ground truth: Powell 2x, Bush 2x, Blair 1x")
    print("Enrolled with photo #1, detected using DIFFERENT photos = real recognition")
    print("=" * 64)


if __name__ == "__main__":
    main()
