"""
Security Watchlist — Alert when a known person appears on camera.

Usage:
    python examples/security_watchlist.py --watchlist watchlist/ --camera 0

Monitors a live camera feed and alerts (prints + optional webhook)
whenever someone from the watchlist is detected.
"""
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
from faceflash import FaceFlash

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def send_webhook(url, name, confidence, frame_path):
    """Send alert to a webhook (Slack, Discord, custom)."""
    if not HAS_REQUESTS or not url:
        return
    payload = {
        "text": f"ALERT: {name} detected (confidence: {confidence:.2f})",
        "name": name,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Security watchlist monitoring")
    parser.add_argument("--watchlist", required=True,
                        help="Folder with watchlist photos (person_name/photo.jpg)")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--rtsp", default=None, help="RTSP URL instead of local camera")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Alert threshold (default: 0.5)")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="Seconds between repeated alerts for same person")
    parser.add_argument("--webhook", default=None,
                        help="Webhook URL for alerts (Slack/Discord/custom)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save frames that trigger alerts")
    args = parser.parse_args()

    # Initialize
    ff = FaceFlash(n_bits=512, n_candidates=100)
    result = ff.register_folder(args.watchlist)
    print(f"Watchlist loaded: {result['registered']} photos of {len(ff.names())} people")
    print(f"Monitoring: {'RTSP ' + args.rtsp if args.rtsp else f'Camera {args.camera}'}")
    print(f"Threshold: {args.threshold}, Cooldown: {args.cooldown}s")
    if args.webhook:
        print(f"Webhook: {args.webhook}")
    print(f"\nMonitoring... (Ctrl+C to stop)\n")

    # Open camera
    source = args.rtsp if args.rtsp else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open camera/stream")
        return

    last_alert = {}  # name -> timestamp
    alert_count = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.rtsp:
                    time.sleep(1)
                    cap = cv2.VideoCapture(source)  # reconnect
                    continue
                break

            frame_count += 1

            # Check every 15 frames (~2x per second)
            if frame_count % 15 != 0:
                continue

            tmp = "/tmp/_watchlist_frame.jpg"
            cv2.imwrite(tmp, frame)

            try:
                result = ff.search(tmp, k=1, threshold=args.threshold)
                if result["matches"]:
                    match = result["matches"][0]
                    name = match["name"]
                    conf = match["confidence"]
                    now = time.time()

                    # Cooldown check
                    if name not in last_alert or (now - last_alert[name]) > args.cooldown:
                        last_alert[name] = now
                        alert_count += 1
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        print(f"  ⚠ ALERT #{alert_count}: {name} detected "
                              f"(confidence: {conf:.2f}) at {timestamp}")

                        # Save frame
                        if args.save_frames:
                            save_dir = Path("alert_frames")
                            save_dir.mkdir(exist_ok=True)
                            frame_path = save_dir / f"alert_{alert_count}_{name}_{timestamp.replace(' ','_')}.jpg"
                            cv2.imwrite(str(frame_path), frame)
                        else:
                            frame_path = None

                        # Webhook
                        send_webhook(args.webhook, name, conf, frame_path)
            except Exception:
                pass

    except KeyboardInterrupt:
        pass

    cap.release()
    print(f"\nStopped. Total alerts: {alert_count}")


if __name__ == "__main__":
    main()
