"""
Attendance System — Track who arrives and when.

Setup:
    1. Create a folder with employee photos: employees/John_Smith/photo.jpg
    2. Run this script: python examples/attendance_system.py

It opens your webcam and logs when each person is first seen.
Output: attendance_log.csv with timestamps.
"""
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
from faceflash import FaceFlash

# ─── Configuration ───────────────────────────────────────────────────────
GALLERY_FOLDER = "employees/"    # folder/person_name/photo.jpg
LOG_FILE = "attendance_log.csv"
CONFIDENCE_THRESHOLD = 0.5       # minimum confidence to mark as present
COOLDOWN_SECONDS = 60            # don't log same person twice within this window
CAMERA_ID = 0

# ─── Initialize ──────────────────────────────────────────────────────────
ff = FaceFlash(n_bits=512, n_candidates=100)

# Register employees
if Path(GALLERY_FOLDER).exists():
    result = ff.register_folder(GALLERY_FOLDER)
    print(f"Registered {result['registered']} faces from {GALLERY_FOLDER}")
else:
    print(f"Create {GALLERY_FOLDER} with employee photos first!")
    print(f"  mkdir -p {GALLERY_FOLDER}/John_Smith/")
    print(f"  cp photo.jpg {GALLERY_FOLDER}/John_Smith/")
    exit(1)

# ─── Attendance tracking ─────────────────────────────────────────────────
seen = {}  # name -> last_seen_timestamp
log_entries = []

# Initialize CSV
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "timestamp", "confidence"])

print(f"\nAttendance system running... (press 'q' to quit)")
print(f"Logging to: {LOG_FILE}\n")

cap = cv2.VideoCapture(CAMERA_ID)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()

    # Check every 10 frames (3x per second at 30fps)
    if frame_count % 10 == 0:
        tmp = "/tmp/_attendance_frame.jpg"
        cv2.imwrite(tmp, frame)

        try:
            result = ff.search(tmp, k=1, threshold=CONFIDENCE_THRESHOLD)
            if result["matches"]:
                match = result["matches"][0]
                name = match["name"]
                conf = match["confidence"]
                now = time.time()

                # Check cooldown
                if name not in seen or (now - seen[name]) > COOLDOWN_SECONDS:
                    seen[name] = now
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entries.append((name, timestamp, conf))

                    # Append to CSV
                    with open(LOG_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([name, timestamp, f"{conf:.3f}"])

                    print(f"  ARRIVED: {name} at {timestamp} (confidence: {conf:.2f})")

                # Draw on frame
                cv2.putText(display, f"{name} ({conf:.2f})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass

    cv2.imshow("Attendance System (q to quit)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nSession complete. {len(log_entries)} arrivals logged to {LOG_FILE}")
