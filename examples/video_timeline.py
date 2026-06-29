"""
Video Timeline Scan — find when a person appears in a video.

Register one or more people, point at a video, get a timeline of every
appearance with timestamps.

Usage:
    python examples/video_timeline.py --person Alice alice.jpg --video meeting.mp4
    python examples/video_timeline.py --gallery people/ --video cctv.mp4 --fps 2
"""

import argparse
from faceflash import FaceFlash


def main():
    ap = argparse.ArgumentParser(description="Find people in a video by face")
    ap.add_argument("--video", required=True, help="Path to the video file")
    ap.add_argument("--person", nargs=2, metavar=("NAME", "PHOTO"),
                    action="append", help="Register a person: --person Alice alice.jpg")
    ap.add_argument("--gallery", help="Folder of people (folder/name/photo.jpg)")
    ap.add_argument("--fps", type=float, default=1.0,
                    help="Frames per second to sample (default 1.0)")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="Match confidence threshold (default 0.4)")
    args = ap.parse_args()

    ff = FaceFlash()

    # Register people
    if args.person:
        for name, photo in args.person:
            ff.register(name, photo)
            print(f"Registered {name}")
    if args.gallery:
        ff.register_folder(args.gallery)
        print(f"Registered gallery: {ff.names()}")

    if len(ff) == 0:
        print("No people registered. Use --person or --gallery.")
        return

    print(f"\nScanning {args.video} at {args.fps} fps...\n")
    timeline = ff.scan_video(args.video, sample_fps=args.fps, threshold=args.threshold)

    if not timeline:
        print("No registered people found in the video.")
        return

    # Print a readable timeline per person
    for name in sorted(timeline):
        appearances = timeline[name]
        total = sum(a["duration_s"] for a in appearances)
        print(f"{name} — {len(appearances)} appearance(s), {total:.0f}s total on screen")
        for a in appearances:
            print(f"    {a['start_timestamp']} → {a['end_timestamp']}  "
                  f"({a['duration_s']:.0f}s, conf {a['best_confidence']:.2f})")
        print()


if __name__ == "__main__":
    main()
