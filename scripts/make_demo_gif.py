"""
Generate a terminal-style demo GIF for the README (docs/figures/demo.gif).

Stylized animation (not a screen recording) — but every benchmark-relevant
number is real: 100K faces, 3 MB index, ~0.5 ms search, CPU, Rust POPCNT.
Names/confidences are illustrative example output.

Run:  python scripts/make_demo_gif.py
"""
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent.parent / "docs" / "figures" / "demo.gif"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Colors (GitHub dark theme)
BG = (13, 17, 23)
FG = (230, 237, 243)
GRAY = (139, 148, 158)
DIM = (88, 96, 105)
GREEN = (14, 159, 110)        # accent (matches charts)
GREEN_HI = (63, 185, 80)
ORANGE = (210, 153, 34)

W, H = 900, 470
PAD_X, PAD_Y = 28, 26
LINE_H = 33
FS = 21

mono_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
bold_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono-Bold.ttf"
FONT = ImageFont.truetype(str(mono_path), FS)
BOLD = ImageFont.truetype(str(bold_path), FS)

# Each line = list of (text, color, font) segments
def seg(text, color=FG, bold=False):
    return (text, color, BOLD if bold else FONT)

LINES = [
    [seg("$ ", DIM), seg("python demo.py", FG)],
    [],
    [seg("  Loading index of 100,000 faces...", GRAY)],
    [seg("  ✓ ", GREEN), seg("100,000 faces", FG, True),
     seg("  ·  3 MB on disk  ·  Rust POPCNT backend", GRAY)],
    [],
    [seg("  ff.search(", FG), seg('"stranger.jpg"', ORANGE), seg(")", FG)],
    [seg("  → ", GREEN), seg("Alice Chen", FG, True),
     seg("        conf 0.94      0.5 ms", GRAY)],
    [],
    [seg("  ff.search(", FG), seg('"group_photo.jpg"', ORANGE), seg(")", FG)],
    [seg("  → ", GREEN), seg("Bob Martinez", FG, True),
     seg("      conf 0.91      0.4 ms", GRAY)],
    [],
    [seg("  100,000 faces.  3 MB.  half a millisecond.  on a CPU.", GREEN_HI, True)],
]

# How long each line holds once it appears (ms)
HOLD = [600, 120, 450, 800, 120, 500, 800, 120, 500, 800, 200, 1600]


def render(n_visible, cursor_on):
    """Draw the first n_visible lines; blink a cursor after the last one."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i in range(n_visible):
        x = PAD_X
        y = PAD_Y + i * LINE_H
        for text, color, font in LINES[i]:
            d.text((x, y), text, font=font, fill=color)
            x += d.textlength(text, font=font)
        if i == n_visible - 1 and cursor_on and LINES[i]:
            d.rectangle([x + 2, y + 3, x + 12, y + FS + 2], fill=GREEN)
    return img


frames, durations = [], []
for k in range(1, len(LINES) + 1):
    # blink cursor twice while this line holds
    half = max(HOLD[k - 1] // 2, 60)
    frames.append(render(k, True)); durations.append(half)
    frames.append(render(k, False)); durations.append(half)
# final hold with steady cursor
frames.append(render(len(LINES), True)); durations.append(1500)

frames[0].save(OUT, save_all=True, append_images=frames[1:],
               duration=durations, loop=0, disposal=2, optimize=True)
size_kb = OUT.stat().st_size / 1024
print(f"  ✓ {OUT.relative_to(OUT.parents[2])}  ({len(frames)} frames, {size_kb:.0f} KB)")
