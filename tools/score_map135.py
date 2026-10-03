# exact score of a class map under #135's renderer, carrier and selector, without the full inflate.
# frame 1 = renderer(map) at 384x512, bilinear to camera, clamp, round (as cpr1/inflate.py render_video).
# frame 0 = carrier (normalised basis, bicubic to camera) + the 14-byte selector (as f26_inflate).
# judges vs the dali targets, tf32 off. bytes = size of the archive given with --archive.
# gate: #135's own map + archive must give seg 2.9639e-4, pose 6.884e-6, score 0.16226842.
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
from modules import DistortionNet
from runtime.frame0_selector import apply_pixel_mode, decode_selector

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--archive", default=str(SUB / "archive.zip"))
ap.add_argument("--coefficients", default="", help="optional .npy [600,12] re-solved carrier strengths")
ap.add_argument("--state", default="", help="joint_state.pt: use its basis codes and gray/amp (pose-stage check)")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96); rend.load_state_dict(comp["renderer"]); rend = rend.to(dev).eval()
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
tg = torch.load(args.targets)
seg_t, pose_t = tg["seg"].to(dev).long(), tg["pose"][:, :6].to(dev).float()
maps = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy())
basis = R.normalized_basis(comp["basis_raw"].to(dev))
GRAY, AMP = 127.5, R.CARRIER_AMPLITUDE
if args.state:
    # decoder-side rebuild of the trained carrier: basis codes * #135's stored per-pattern scales, as cpr1 decodes it
    from runtime import residual_archive as RA
    from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
    parts = RA.read_residual_archive(SUB / "archive.zip")
    canon = materialize_cpr1(split_frame0_selector_carrier(parts.carrier_blob)[0], R)
    bs, _, _, _ = R.decode_compact_carrier(canon, basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W,
                                           frames=R.N, dimensions=R.CARRIER_DIM)
    st = torch.load(args.state, weights_only=False)
    bcodes = st["basis_codes"].to(dev).float()
    basis = R.normalized_basis(bcodes * torch.as_tensor(np.asarray(bs, dtype=np.float32), device=dev)[:, None, None, None])
    GRAY, AMP = float(np.float32(st["gray"])), float(np.float32(st["amp"]))
    print(f"state basis codes changed {int((bcodes != torch.from_numpy(np.asarray(_bc0 := R.decode_compact_carrier(canon, basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W, frames=R.N, dimensions=R.CARRIER_DIM)[1], dtype=np.float32).reshape(bcodes.shape)).to(dev)).sum())}  gray {GRAY}  amp {AMP}")
coef = torch.from_numpy(np.load(args.coefficients)).to(dev).float() if args.coefficients else comp["coefficients"].to(dev)
modes, sel = decode_selector(comp["selector"])

seg_sum = pose_sum = 0.0
t0 = time.time()
with torch.inference_mode():
    for s in range(0, R.N, args.batch):
        e = min(s + args.batch, R.N)
        idx = torch.arange(s, e, device=dev)
        f1 = F.interpolate(rend(maps[s:e].long().to(dev), idx), size=(R.CAMERA_H, R.CAMERA_W),
                           mode="bilinear", align_corners=False).clamp(0, 255).round()
        car = torch.einsum("bk,kchw->bchw", coef[s:e], basis) / math.sqrt(R.CARRIER_DIM)
        f0 = F.interpolate((GRAY + AMP * car).clamp(0, 255).round(),
                           size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255).round()
        f0 = f0.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for j in range(e - s):
            f0[j] = apply_pixel_mode(f0[j:j + 1].copy(), modes[sel[s + j]])[0]
        f0 = torch.from_numpy(f0).to(dev).float()
        x = torch.stack([f0, f1.permute(0, 2, 3, 1)], 1)
        po, so = net(x)
        seg_sum += (so.argmax(1) != seg_t[s:e]).float().mean(dim=(1, 2)).sum().item()
        pose_sum += (po["pose"][:, :6] - pose_t[s:e]).pow(2).mean(1).sum().item()
seg, pose = seg_sum / R.N, pose_sum / R.N
size = Path(args.archive).stat().st_size
score = 100 * seg + math.sqrt(10 * pose) + 25 * size / 37_545_489
print(f"seg {seg:.8e}  pose {pose:.8e}  bytes {size:,}  score {score:.8f}  "
      f"(seg {100*seg:.5f} pose {math.sqrt(10*pose):.5f} rate {25*size/37_545_489:.5f})  {time.time()-t0:.0f}s")
