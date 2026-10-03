# zoomed before/after on the horizon band where the edits concentrate; changed pixels white, 3x nearest upscale
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

before = np.fromfile(sys.argv[1], dtype=np.uint8).reshape(600, 384, 512)
after = np.fromfile(sys.argv[2], dtype=np.uint8).reshape(600, 384, 512)
out = Path(sys.argv[3]); f0 = int(sys.argv[4]); nfr = int(sys.argv[5])
diff = before != after
ys, xs = np.nonzero(diff[f0:f0 + nfr].any(0))
y0, y1 = max(0, ys.min() - 12), min(384, ys.max() + 13)
x0, x1 = max(0, xs.min() - 12), min(512, xs.max() + 13)
palette = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [220, 20, 60], [0, 0, 142]], dtype=np.uint8)
S = 3
w, h = (x1 - x0) * S, (y1 - y0) * S
lab, pad = 18, 6
canvas = Image.new("RGB", (2 * w + 3 * pad, nfr * (h + lab + pad) + pad), (20, 20, 20))
d = ImageDraw.Draw(canvas)
for k in range(nfr):
    f = f0 + k
    b = palette[before[f, y0:y1, x0:x1]]
    a = palette[after[f, y0:y1, x0:x1]].copy()
    a[diff[f, y0:y1, x0:x1]] = 255
    y = pad + k * (h + lab + pad)
    for j, (img, name) in enumerate(((b, "before #135"), (a, "after training, white = changed"))):
        x = pad + j * (w + pad)
        d.text((x, y), f"frame {f}  {name}  ({int(diff[f].sum())} px)  crop y{y0}-{y1} x{x0}-{x1}", fill=(230, 230, 230))
        canvas.paste(Image.fromarray(img).resize((w, h), Image.NEAREST), (x, y + lab))
canvas.save(out)
print("saved", out, "crop", (y0, y1, x0, x1))
