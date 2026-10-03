# score an inflated submission against the dali targets from make_targets.py.
# same math as evaluate.py: 100*seg + sqrt(10*pose) + 25*rate, pose mse over the first 6 outputs.
import os, sys, math, time
from pathlib import Path
import numpy as np
import torch

ROOT, TARGETS, SUB = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(ROOT))
from modules import DistortionNet
from frame_utils import camera_size, seq_len

device = torch.device("cuda", 0)
# match the official numeric policy: ieee fp32, no tf32 anywhere
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
net = DistortionNet().eval().to(device)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", device)
gt = torch.load(TARGETS)
gt_pose, gt_seg = gt["pose"].to(device), gt["seg"].to(device)

w, h = camera_size
raw = np.memmap(SUB / "inflated/0.raw", dtype=np.uint8, mode="r").reshape(-1, seq_len, h, w, 3)
pairs = raw.shape[0]
pose_sum = seg_sum = 0.0
started = time.time()
with torch.inference_mode():
    bs = int(os.environ.get("SCORE_BATCH", "16"))  # 4 fits a 4 GB laptop gpu
    for i in range(0, pairs, bs):
        batch = torch.from_numpy(np.ascontiguousarray(raw[i:i + bs])).to(device)
        pose_out, seg_out = net(batch)
        p = pose_out["pose"][..., :6] - gt_pose[i:i + bs, :6]
        pose_sum += p.pow(2).mean(dim=1).sum().item()
        seg_sum += (seg_out.argmax(dim=1) != gt_seg[i:i + bs]).float().mean(dim=(1, 2)).sum().item()
pose, seg = pose_sum / pairs, seg_sum / pairs
size = (SUB / "archive.zip").stat().st_size
rate = size / 37545489
score = 100 * seg + math.sqrt(10 * pose) + 25 * rate
print(f"pairs {pairs}  pose {pose:.8e}  seg {seg:.8e}  bytes {size}  score {score:.6f}"
      f"  (seg {100*seg:.5f} pose {math.sqrt(10*pose):.5f} rate {25*rate:.5f})  {time.time()-started:.0f}s")
