# FaceFlash Examples

Practical examples showing real-world use cases.

## Examples

| Example | Description | Use case |
|---------|------------|----------|
| [basic_search.py](basic_search.py) | Register + search in 5 lines | Getting started |
| [build_index.py](build_index.py) | Bulk register from folder + save/load | Building a gallery |
| [attendance_system.py](attendance_system.py) | Live camera attendance with CSV logging | Office / school |
| [photo_organizer.py](photo_organizer.py) | Group photos by person (like Google Photos) | Personal photo management |
| [video_search.py](video_search.py) | Find when a person appears in a video | Meeting review / surveillance |
| [security_watchlist.py](security_watchlist.py) | Real-time alerts when a known person appears | Security / access control |
| [face_dedup.py](face_dedup.py) | Find duplicate faces in a dataset | Dataset cleaning |

## Quick Start

```bash
# Install
pip install "faceflash[cpu] @ git+https://github.com/raghavenderreddygrudhanti/faceflash.git"

# Run any example
python examples/attendance_system.py
python examples/photo_organizer.py --input ~/Photos --output ~/Organized
python examples/video_search.py --video meeting.mp4 --gallery faces/
python examples/security_watchlist.py --watchlist watchlist/ --camera 0
python examples/face_dedup.py --input dataset/
```

## Folder Structure for Photos

All examples expect photos organized as:
```
gallery_folder/
├── Person_Name/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Another_Person/
│   └── photo.jpg
```

Multiple photos per person improve matching accuracy.
