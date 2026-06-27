"""
Video Face Search — Find when a person appears in a video.

Usage:
    python examples/video_search.py --video meeting.mp4 --gallery faces/

It scans the video, identifies faces against your gallery,
and outputs a timeline of who appeared when.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from faceflash import FaceFlash


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Find people in a video")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--gallery", required=True, help="Gallery folder (person_name/photo.jpg)")
    parser.add_argument("--output", default="video_results.json", help="Output JSON")
    parser.add_argument("--every-n-frames", type=int, default=15,
                        help="Check every N frames (default: 15 = ~2x/sec at 30fps)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Confidence threshold")
    parser.add_argument("--annotate", action="store_true",
                        help="Save annotated video with names drawn on faces")
    args = parser.parse_args()

    # Load gallery
    ff = FaceFlash(n_bits=512, n_candidates=100)
    result = ff.register_folder(args.gallery)
    print(f"Gallery: {result['registered']} faces registered")
    print(f"People: {ff.names()}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video: {args.video} ({format_time(duration)}, {fps:.0f} fps, {total_frames} frames)")
    print(f"Checking every {args.every_n_frames} frames ({fps/args.every_n_frames:.1f} checks/sec)")

    # Optional: annotated output video
    out_video = None
    if args.annotate:
        out_path = args.video.replace(".mp4", "_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"Annotated video: {out_path}")

    # Process
    detections = []  # list of {time, name, confidence}
    frame_count = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % args.every_n_frames == 0:
            tmp = "/tmp/_video_frame.jpg"
            cv2.imwrite(tmp, frame)

            try:
                result = ff.search(tmp, k=3, threshold=args.threshold)
                current_time = frame_count / fps

                for match in result["matches"]:
                    detections.append({
                        "time": format_time(current_time),
                        "seconds": round(current_time, 1),
                        "name": match["name"],
                        "confidence": round(match["confidence"], 3),
                        "frame": frame_count,
                    })

                # Draw on frame for annotated video
                if out_video and result["matches"]:
                    for i, match in enumerate(result["matches"]):
                        y = 40 + i * 35
                        color = (0, 255, 0) if match["confidence"] > 0.5 else (0, 255, 255)
                        cv2.putText(frame, f"{match['name']} ({match['confidence']:.2f})",
                                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception:
                pass

        if out_video:
            out_video.write(frame)

        # Progress
        if frame_count % (args.every_n_frames * 30) == 0:
            elapsed = time.perf_counter() - t0
            progress = frame_count / total_frames * 100
            print(f"  {progress:.0f}% ({format_time(frame_count/fps)}) — "
                  f"{len(detections)} detections so far")

    cap.release()
    if out_video:
        out_video.release()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s ({frame_count/elapsed:.0f} fps processing speed)")

    # Build timeline
    timeline = {}
    for d in detections:
        name = d["name"]
        if name not in timeline:
            timeline[name] = {"first_seen": d["time"], "last_seen": d["time"], "appearances": 0}
        timeline[name]["last_seen"] = d["time"]
        timeline[name]["appearances"] += 1

    # Output
    output = {
        "video": args.video,
        "duration": format_time(duration),
        "frames_checked": frame_count // args.every_n_frames,
        "people_found": list(timeline.keys()),
        "timeline": timeline,
        "detections": detections,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"\nTimeline:")
    for name, info in timeline.items():
        print(f"  {name}: {info['first_seen']} - {info['last_seen']} "
              f"({info['appearances']} appearances)")


if __name__ == "__main__":
    main()
