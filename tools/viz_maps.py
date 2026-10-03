# before/after class maps for the 20 consecutive frames with the most changes; changed pixels drawn white
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

before = np.fromfile(sys.argv[1], dtype=np.uint8).reshape(600, 384, 512)
after = np.fromfile(sys.argv[2], dtype=np.uint8).reshape(600, 384, 512)
out = Path(sys.argv[3])
diff = before != after
per = diff.reshape(600, -1).sum(1)
win = np.convolve(per, np.ones(20, dtype=int), mode="valid")
f0 = int(win.argmax())
print(f"frames {f0}-{f0+19}: {int(per[f0:f0+20].sum())} changed pixels (whole video {int(per.sum())})")
palette = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [220, 20, 60], [0, 0, 142]], dtype=np.uint8)
cols, rows = 4, 10                     # each row: (before, after) for two frames
W, H, pad, lab = 512, 384, 6, 22
canvas = Image.new("RGB", (cols * (W + pad) + pad, rows * (H + pad + lab) + pad), (20, 20, 20))
draw = ImageDraw.Draw(canvas)
for k in range(20):
    f = f0 + k
    b = palette[before[f]]
    a = palette[after[f]].copy()
    a[diff[f]] = 255
    r, c = divmod(k, 2)
    for j, (img, name) in enumerate(((b, "before #135"), (a, "after training (white = changed)"))):
        x = pad + (c * 2 + j) * (W + pad)
        y = pad + r * (H + pad + lab)
        draw.text((x, y), f"frame {f}  {name}  ({int(diff[f].sum())} px changed)", fill=(230, 230, 230))
        canvas.paste(Image.fromarray(img), (x, y + lab))
canvas.save(out)
print("saved", out)
