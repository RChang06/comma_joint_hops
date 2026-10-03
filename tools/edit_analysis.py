# how the trained map edits are shaped: clusters vs isolated flips, and distance to the nearest segnet error of the
# starting map (#140's search only ever moves a wrong pixel or one of its 8 neighbours, i.e. distance <= 1)
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
start = np.fromfile(sys.argv[1], dtype=np.uint8).reshape(600, 384, 512)
end = np.fromfile(sys.argv[2], dtype=np.uint8).reshape(600, 384, 512)
wrong0 = torch.load(sys.argv[3]) if len(sys.argv) > 3 else None
ch = torch.from_numpy(start != end).float()
n = int(ch.sum())
nb = F.conv2d(ch[:, None], torch.ones(1, 1, 3, 3), padding=1)[:, 0] - ch      # changed 8-neighbours of each changed px
isolated = int(((nb == 0) & (ch > 0)).sum())
print(f"changed pixels {n:,}; isolated (no changed 8-neighbour) {isolated:,} ({isolated/n*100:.1f}%); "
      f"in clusters {n-isolated:,} ({(n-isolated)/n*100:.1f}%)")
if wrong0 is not None:
    w = wrong0.float()[:, None]
    d1 = F.max_pool2d(w, 3, 1, 1)[:, 0] > 0
    d3 = F.max_pool2d(w, 7, 1, 3)[:, 0] > 0
    near1 = int((ch.bool() & d1).sum()); near3 = int((ch.bool() & d3).sum())
    print(f"within 1 px of a starting segnet error (where #140's search can reach): {near1:,} ({near1/n*100:.1f}%)")
    print(f"farther than 1 px: {n-near1:,} ({(n-near1)/n*100:.1f}%); farther than 3 px: {n-near3:,} ({(n-near3)/n*100:.1f}%)")
