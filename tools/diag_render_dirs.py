# is the renderer at a sharp local minimum for exact seg on the current map? perturb all w4 row scales by a random
# direction of about one fp16 step (+ and -), re-render all 600 frames exactly as inflate does, count segnet errors.
import sys, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from modules import DistortionNet
from runtime import residual_archive as RA
from runtime.entropy.renderer_weight_codec import decode_wans1

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
maps = torch.from_numpy(np.fromfile(sys.argv[1], dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy())
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
EPS = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
seg_t = torch.load(HERE / "targets/gt_dali_t2000.pt")["seg"].to(dev).long()
parts = RA.read_residual_archive(SUB / "archive.zip")
recs = [r for r in decode_wans1(parts.semantic_blob)]
rend = R.SemanticTokenRenderer(96).to(dev).eval()
base_sd = {k: v.clone() for k, v in rend.state_dict().items()}


def load(delta):
    # delta: dict name -> per-row log-scale offsets (or None)
    sd = dict(base_sd)
    for r in recs:
        if r.codes is None:
            sd[r.schema.name] = torch.from_numpy(np.asarray(r.values, dtype=np.float32)).reshape(r.schema.shape)
            continue
        sc = np.asarray(r.scales, dtype=np.float32)
        if delta is not None:
            sc = (sc * np.exp(delta[r.schema.name])).astype(np.float16).astype(np.float32)
        shp = [1] * len(r.schema.shape)
        shp[-1 if r.schema.name.endswith("embed.weight") else 0] = r.schema.scale_count
        sd[r.schema.name] = torch.from_numpy(r.codes.astype(np.float32) * sc.reshape(shp))
    rend.load_state_dict({k: v.to(dev) for k, v in sd.items()})


@torch.inference_mode()
def seg_err():
    wrong = 0.0
    for s in range(0, R.N, 8):
        e = min(s + 8, R.N)
        idx = torch.arange(s, e, device=dev)
        f1 = F.interpolate(rend(maps[s:e].long().to(dev), idx), size=(R.CAMERA_H, R.CAMERA_W),
                           mode="bilinear", align_corners=False).clamp(0, 255).round()
        x = torch.stack([torch.zeros_like(f1), f1], 1).permute(0, 1, 3, 4, 2)
        so = net.segnet(net.preprocess_input(x)[1])
        wrong += (so.argmax(1) != seg_t[s:e]).float().mean(dim=(1, 2)).sum().item()
    return wrong / R.N


load(None)
base = seg_err()
print(f"base seg {base:.8e}", flush=True)
rng = np.random.default_rng(0)
better = 0
for k in range(K):
    d = {r.schema.name: rng.standard_normal(r.schema.scale_count).astype(np.float32) * EPS for r in recs if r.codes is not None}
    out = []
    for sgn in (1, -1):
        load({n: sgn * v for n, v in d.items()})
        out.append(seg_err())
    better += sum(o < base for o in out)
    print(f"dir {k}: +{(out[0]-base)*100:+.6f}  -{(out[1]-base)*100:+.6f}  (seg-term change)", flush=True)
print(f"directions better than base: {better} of {2*K}")
