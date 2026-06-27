"""
FaceFlash Live Camera Demo
===========================

1. Place your photos in a folder: demo_faces/<your_name>/photo1.jpg, photo2.jpg, ...
2. Run: python demo_live.py
3. The webcam opens and identifies you in real-time.

Controls:
  q - quit
  s - save a screenshot
  r - re-register faces (if you add new photos)

Usage:
  # Register from a folder structure
  python demo_live.py --faces demo_faces/

  # Or register individual images
  python demo_live.py --register "Raghavender:path/to/photo.jpg"
"""

import argparse
import time
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from faceflash import FaceFlash


def register_from_folder(ff, folder_path):
    """Register all faces from folder/person_name/photo.jpg structure."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        print(f"  Create it with: mkdir -p {folder}/<your_name>/")
        print(f"  Then add your photos as JPG/PNG files.")
        return 0

    count = 0
    for person_dir in sorted(folder.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        photos = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
        for photo in photos:
            try:
                ff.register(name, str(photo))
                count += 1
                print(f"  Registered: {name} ({photo.name})")
            except Exception as e:
                print(f"  Skip: {name}/{photo.name} — {e}")
    return count


def register_individual(ff, pairs):
    """Register from 'Name:path' pairs."""
    count = 0
    for pair in pairs:
        if ":" not in pair:
            print(f"  Invalid format: {pair} (use 'Name:path/to/photo.jpg')")
            continue
        name, path = pair.split(":", 1)
        try:
            ff.register(name.strip(), path.strip())
            count += 1
            print(f"  Registered: {name}")
        except Exception as e:
            print(f"  Skip: {name} — {e}")
    return count


def run_camera(ff, camera_id=0):
    """Run live camera identification."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check permissions (System Preferences > Privacy > Camera)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("  LIVE CAMERA — press 'q' to quit")
    print("=" * 50 + "\n")

    frame_count = 0
    last_result = None
    last_search_time = 0
    search_every_n = 5  # search every N frames (saves CPU)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        # Run face search every N frames
        if frame_count % search_every_n == 0:
            # Save frame temporarily for FaceFlash
            tmp_path = "/tmp/_faceflash_frame.jpg"
            cv2.imwrite(tmp_path, frame)

            try:
                t0 = time.perf_counter()
                result = ff.search(tmp_path, k=1, threshold=0.3)
                last_search_time = (time.perf_counter() - t0) * 1000
                last_result = result
            except Exception:
                last_result = None

        # Draw result on frame
        if last_result and last_result["matches"]:
            match = last_result["matches"][0]
            name = match["name"]
            conf = match["confidence"]
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)

            # Draw name + confidence
            label = f"{name} ({conf:.2f})"
            cv2.putText(display, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(display, f"{last_search_time:.0f}ms", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        else:
            cv2.putText(display, "Unknown", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            if last_search_time > 0:
                cv2.putText(display, f"{last_search_time:.0f}ms", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("FaceFlash Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("faceflash_screenshot.jpg", display)
            print("  Screenshot saved: faceflash_screenshot.jpg")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FaceFlash Live Camera Demo")
    parser.add_argument("--faces", default="demo_faces",
                        help="Folder with photos: faces/<name>/*.jpg (default: demo_faces/)")
    parser.add_argument("--register", nargs="*", default=[],
                        help="Register individual photos as 'Name:path.jpg'")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera ID (default: 0)")
    parser.add_argument("--index", default=None,
                        help="Load a saved index instead of registering")
    args = parser.parse_args()

    print("=" * 50)
    print("  FaceFlash Live Camera Demo")
    print("=" * 50)

    # Initialize
    ff = FaceFlash(n_bits=512, n_candidates=100)

    if args.index:
        print(f"\n  Loading index from: {args.index}")
        ff.load(args.index)
        print(f"  Loaded {ff.index.count} faces")
    else:
        # Register faces
        print(f"\n  Registering faces from: {args.faces}/")
        count = register_from_folder(ff, args.faces)

        if args.register:
            count += register_individual(ff, args.register)

        if count == 0:
            print(f"\n  No faces registered!")
            print(f"  Create the folder structure:")
            print(f"    mkdir -p {args.faces}/<your_name>/")
            print(f"    cp your_photo.jpg {args.faces}/<your_name>/")
            print(f"  Then run again.")
            return

        print(f"\n  Total registered: {count} faces")
        print(f"  Index stats: {ff.index.stats()}")

    # Run camera
    run_camera(ff, args.camera)


if __name__ == "__main__":
    main()
