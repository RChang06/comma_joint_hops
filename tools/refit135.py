# re-solve #135's pose-carrier strengths (600x12) for the even frames a given map produces.
#
# a map edit changes frame 1, and the stale carrier then makes pose explode (387-467x, per #140's notes;
# 2.3e-3 vs 6.9e-6 measured here on the #141-maps transplant). the basis stays fixed; only the per-pair
# strengths move. loss is plain pose mse (the scored metric), not #130's range-normalised loss.
# frame 1 is rendered once and cached (the renderer is frozen). rounding to uint8 is straight-through, and the
# result is snapped to #135's int12 grid (per-dimension scales from the shipped carrier) before evaluation.
import argparse, sys, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
import modules, frame_utils
from modules import DistortionNet
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True)
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--lr", type=float, default=0.02, help="in units of each dimension's int12 step")
ap.add_argument("--batch", type=int, default=20)
ap.add_argument("--out", required=True, help="output .npy of re-solved float strengths, already on the int12 grid")
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96); rend.load_state_dict(comp["renderer"]); rend = rend.to(dev).eval()
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in list(rend.parameters()) + list(net.parameters()):
    p.requires_grad_(False)
pose_t = torch.load(args.targets)["pose"][:, :6].to(dev).float()
maps = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy())

masters = torch.empty((R.N, 3, R.CAMERA_H, R.CAMERA_W), dtype=torch.uint8, device=dev)
with torch.no_grad():
    for s in range(0, R.N, 8):
        idx = torch.arange(s, min(s + 8, R.N), device=dev)
        masters[s:s + 8] = F.interpolate(rend(maps[s:s + 8].long().to(dev), idx), size=(R.CAMERA_H, R.CAMERA_W),
                                         mode="bilinear", align_corners=False).clamp(0, 255).round().to(torch.uint8)
basis = R.normalized_basis(comp["basis_raw"].to(dev))
coef0 = comp["coefficients"].to(dev).float()
# the shipped strengths are int12 codes * per-dimension scale; recover the scale as the smallest code step
step = torch.stack([(coef0[:, k][coef0[:, k] != 0].abs().min() if (coef0[:, k] != 0).any() else torch.tensor(1.0, device=dev))
                    for k in range(R.CARRIER_DIM)])
codes = (coef0 / step).round()
print("int12 code step per dim", step.tolist()[:4], "... max |code|", int(codes.abs().max()))


def slave(c):
    car = torch.einsum("bk,kchw->bchw", c, basis) / math.sqrt(R.CARRIER_DIM)
    x = (127.5 + R.CARRIER_AMPLITUDE * car).clamp(0, 255)
    x = x + (x.round() - x).detach()
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def pose_mse(c, s, e):
    x = torch.stack([slave(c), masters[s:e].float()], 1)          # (b, t=2, c, h, w)
    return (net.posenet(net.posenet.preprocess_input(x))["pose"][:, :6] - pose_t[s:e]).pow(2).mean(1)


def evaluate(cd):
    tot = 0.0
    with torch.no_grad():
        for s in range(0, R.N, args.batch):
            e = min(s + args.batch, R.N)
            tot += pose_mse(cd[s:e] * step, s, e).sum().item()
    return tot / R.N


z = codes.clone().requires_grad_(True)          # optimise in code units
opt = torch.optim.Adam([z], lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
p0 = evaluate(codes)
print(f"start pose {p0:.4e} (term {math.sqrt(10*p0):.5f})", flush=True)
best, best_codes = p0, codes.clone()
t0 = time.time()
for ep in range(1, args.epochs + 1):
    for s in range(0, R.N, args.batch):
        e = min(s + args.batch, R.N)
        loss = pose_mse(z[s:e] * step, s, e).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sched.step()
    if ep % 10 == 0 or ep == args.epochs:
        q = z.detach().round().clamp(-2048, 2047)
        pq = evaluate(q)
        if pq < best:
            best, best_codes = pq, q.clone()
        print(f"epoch {ep}: pose (int12) {pq:.4e} term {math.sqrt(10*pq):.5f}  best {best:.4e}  {time.time()-t0:.0f}s", flush=True)
np.save(args.out, (best_codes * step).cpu().numpy())
print(f"saved {args.out}: pose {p0:.4e} -> {best:.4e}  (term {math.sqrt(10*p0):.5f} -> {math.sqrt(10*best):.5f})")
