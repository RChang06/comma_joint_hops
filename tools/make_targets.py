# decode the original video with the official dali reader and store what the scorer needs
# from it: posenet's pose output and segnet's argmax map per pair. the original never changes,
# so this runs once on a gpu with a working nvdec and every later score compares against it.
import sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from frame_utils import DaliVideoDataset
from modules import DistortionNet

device = torch.device("cuda", 0)
net = DistortionNet().eval().to(device)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", device)
ds = DaliVideoDataset(["0.mkv"], data_dir=ROOT / "videos", batch_size=4, device=device)
ds.prepare_data()

poses, segs = [], []
started = time.time()
with torch.inference_mode():
    for _, idx, batch in ds:
        pose_out, seg_out = net(batch.to(device))
        poses.append(pose_out["pose"].float().cpu())
        segs.append(seg_out.argmax(dim=1).to(torch.uint8).cpu())
        if idx % 25 == 0:
            print(f"batch {idx} pairs {sum(p.shape[0] for p in poses)} {time.time() - started:.0f}s", flush=True)
pose = torch.cat(poses)
seg = torch.cat(segs)
print("pose", tuple(pose.shape), "seg", tuple(seg.shape))
torch.save({"pose": pose, "seg": seg}, OUT)
print("saved", OUT)
