# damped gauss-newton re-solve of #135's carrier strengths (600x12) for the even frames a map produces.
#
# per pair: 12 unknowns, 6 scored pose outputs. linearise posenet around the current strengths (6x12 jacobian
# from 6 backward passes), solve (J^T J + lam*D) d = -J^T r, accept the step per pair only if the EXACT
# (int12-snapped, uint8-rounded) pose mse falls, otherwise raise lam (levenberg-marquardt). this is the method
# #140's author used (damped gn), which recovered pose to ~1.07x base after token edits; the adam refit in
# refit135.py was far too slow (pose 2.3e-3 -> 6.7e-4 where ~6.4e-6 is reachable).
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
from runtime import residual_archive as RA
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True)
ap.add_argument("--init", default="135", help="135 | 141 | path to a .npy of strengths")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--polish", type=int, default=0, help="passes of exact +-1/+-2 integer polish per dimension after gn")
ap.add_argument("--out", required=True)
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
c135 = comp["coefficients"].to(dev).float()
def _true_coef_scales():
    # the int12 grid's per-dimension scales, read from the carrier itself (not inferred from the values)
    from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
    _parts = RA.read_residual_archive(SUB / "archive.zip")
    _car, _ = split_frame0_selector_carrier(_parts.carrier_blob)
    _canon = materialize_cpr1(_car, R)
    _, _, cs, _ = R.decode_compact_carrier(_canon, basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W,
                                           frames=R.N, dimensions=R.CARRIER_DIM)
    return torch.from_numpy(np.asarray(cs, dtype=np.float32)).to(dev)


step = _true_coef_scales()
if args.init == "135":
    coef = c135.clone()
elif args.init == "141":
    coef = torch.load(HERE / "extracted/pr141_components.pt", weights_only=False)["coefficients"].to(dev).float()
else:
    coef = torch.from_numpy(np.load(args.init)).to(dev).float()
coef = (coef / step).round().clamp(-2048, 2047) * step


def slave(c, hard):
    car = torch.einsum("bk,kchw->bchw", c, basis) / math.sqrt(R.CARRIER_DIM)
    x = (127.5 + R.CARRIER_AMPLITUDE * car).clamp(0, 255)
    x = x.round() if hard else x
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
    return x.round() if hard else x


def outputs(c, s, e, hard):
    x = torch.stack([slave(c, hard), masters[s:e].float()], 1)
    return net.posenet(net.posenet.preprocess_input(x))["pose"][:, :6]


def exact_mse(c):
    out = torch.empty(R.N, device=dev)
    with torch.no_grad():
        for s in range(0, R.N, args.batch):
            e = min(s + args.batch, R.N)
            q = (c[s:e] / step).round().clamp(-2048, 2047) * step
            out[s:e] = (outputs(q, s, e, True) - pose_t[s:e]).pow(2).mean(1)
    return out


cur = exact_mse(coef)
print(f"start pose {cur.mean():.4e} (term {math.sqrt(10*cur.mean().item()):.5f})", flush=True)
lam = torch.full((R.N,), 1e-3, device=dev)
t0 = time.time()
for it in range(1, args.iters + 1):
    delta = torch.zeros_like(coef)
    for s in range(0, R.N, args.batch):
        e = min(s + args.batch, R.N)
        c = coef[s:e].clone().requires_grad_(True)
        out = outputs(c, s, e, False)
        r = (out - pose_t[s:e]).detach()                                     # (b,6)
        J = torch.zeros(e - s, 6, R.CARRIER_DIM, device=dev)
        for k in range(6):
            g, = torch.autograd.grad(out[:, k].sum(), c, retain_graph=k < 5)
            J[:, k] = g
        JtJ = J.transpose(1, 2) @ J
        D = torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2).clamp_min(1e-12))
        A = JtJ + lam[s:e, None, None] * D
        rhs = -(J.transpose(1, 2) @ r[:, :, None])[:, :, 0]
        delta[s:e] = torch.linalg.solve(A, rhs)
    trial = exact_mse(coef + delta)
    better = trial < cur
    coef = torch.where(better[:, None], coef + delta, coef)
    cur = torch.where(better, trial, cur)
    lam = torch.where(better, (lam / 3).clamp_min(1e-6), (lam * 4).clamp_max(1e6))
    print(f"iter {it}: pose {cur.mean():.4e} (term {math.sqrt(10*cur.mean().item()):.5f})  accepted {int(better.sum())}/600  "
          f"{time.time()-t0:.0f}s", flush=True)
coef = (coef / step).round().clamp(-2048, 2047) * step
# integer polish (#140 alternates gn with a +-2 search): try each dimension +-1, +-2 for every pair at once,
# keep per pair whatever lowers the exact pose mse
cur = exact_mse(coef)
for ps in range(args.polish):
    moved = 0
    for k in range(R.CARRIER_DIM):
        for m in (-2, -1, 1, 2):
            trial_c = coef.clone(); trial_c[:, k] += m * step[k]
            tr = exact_mse(trial_c)
            better = tr < cur
            coef = torch.where(better[:, None], trial_c, coef); cur = torch.where(better, tr, cur)
            moved += int(better.sum())
    print(f"polish pass {ps+1}: pose {cur.mean():.4e} (term {math.sqrt(10*cur.mean().item()):.5f})  moves {moved}", flush=True)
    if moved == 0:
        break
np.save(args.out, coef.cpu().numpy())
print(f"saved {args.out}: final pose {exact_mse(coef).mean():.4e}")
