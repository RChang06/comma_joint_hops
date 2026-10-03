# how many class-map errors survive #141's renderer + segnet?
#
# renders the second frame of every pair from a given class map with #141's renderer, through the
# exact inflate path (384x512 render -> bilinear up to 874x1164 -> clamp -> round -> uint8), and
# scores segnet's argmax against the dali ground truth. segnet only reads frame[-1], so frame 0 is a
# copy of frame 1 and the pose output is ignored.
#
# maps tested:
#   pr141   #141's decoded (edited) tokens              -> should reproduce seg 2.0136e-4
#   gt      segnet's true argmax on the original video  -> what the renderer does with the exact answer
#   mapnet  argmax of the mapnet checkpoint              -> the experiment
#
# usage (repo root, wsl venv): python work/render_test.py <mapnet best.pt> [pairs]
import sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submissions/semantic_blocks/cpr1"))
sys.path.insert(0, "/mnt/d/projects/mapnet")
import inflate as R
from modules import DistortionNet
from mapnet import MapNet

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
pairs = int(sys.argv[2]) if len(sys.argv) > 2 else R.N
BS = 4

comp = torch.load(ROOT / "work/extracted/pr141_components.pt")
gt_seg = torch.load(ROOT / "work/targets/gt_dali_t2000.pt")["seg"][:pairs]
renderer = R.SemanticTokenRenderer(96)
renderer.load_state_dict(comp["renderer"])
renderer = renderer.eval().to(dev)
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)

ck = torch.load(sys.argv[1], map_location="cpu")
a = ck["args"]
mapnet = MapNet(R.N, a["latent_dim"], a["c0"]).to(dev)
mapnet.load_state_dict(ck["model"])
mapnet.eval()
with torch.no_grad():
    mapnet_maps = torch.cat([mapnet(torch.arange(s, min(s + 8, pairs), device=dev)).argmax(1).to(torch.uint8).cpu()
                             for s in range(0, pairs, 8)])

maps = {"pr141": comp["tokens"][:pairs], "gt": gt_seg, "mapnet": mapnet_maps}
total_px = pairs * R.EVAL_H * R.EVAL_W

for name, m in maps.items():
    started = time.time()
    map_wrong = seg_wrong = absorbed = introduced = 0
    with torch.inference_mode():
        for s in range(0, pairs, BS):
            idx = torch.arange(s, min(s + BS, pairs), device=dev)
            tok = m[s:s + BS].to(dev)
            frame = F.interpolate(renderer(tok.long(), idx), size=(R.CAMERA_H, R.CAMERA_W),
                                  mode="bilinear", align_corners=False).clamp(0, 255).round().to(torch.uint8)
            frame = frame.permute(0, 2, 3, 1)
            batch = torch.stack([frame, frame], dim=1)
            _, seg_out = net(batch)
            pred = seg_out.argmax(1)
            truth = gt_seg[s:s + BS].to(dev)
            mw = tok != truth
            sw = pred != truth
            map_wrong += int(mw.sum())
            seg_wrong += int(sw.sum())
            absorbed += int((mw & ~sw).sum())
            introduced += int((~mw & sw).sum())
    print(f"{name:7s} map errors {map_wrong:>9,} ({map_wrong / total_px:.3e})  "
          f"segnet errors {seg_wrong:>9,} (seg {seg_wrong / total_px:.4e}, term {100 * seg_wrong / total_px:.4f})  "
          f"absorbed {absorbed:,} ({absorbed / max(map_wrong, 1):.1%} of map errors)  "
          f"new {introduced:,}  {time.time() - started:.0f}s", flush=True)
