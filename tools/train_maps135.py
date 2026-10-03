# train #135's class maps directly against the score: seg + pose + exact token bytes, renderer frozen.
#
# variables: 5-way logits for every pixel within --band px of a class boundary in the starting map (others stay
# hard). the forward pass uses the hard argmax map (straight-through), so the renderer always sees a real map.
# loss, in score units per pair:
#   seg   100/600 * expected flips (sigmoid(-margin/tau) against the dali segnet targets)
#   pose  dscore/dpose * mse, with the slope of sqrt(10*P) taken at the current pose level P0
#   rate  25/ORIG * sum(q * bits)/8, bits = exact per-pixel class costs from exact_cost135.py --dump-costs
# the price list is the exact #135 code length per class given the starting map as context; neighbour effects
# are first order only, so the real check is exact_cost135.py --tokens on the result, never this loss.
import argparse, sys, math, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
import modules, frame_utils
from modules import DistortionNet
from runtime.frame0_selector import apply_pixel_mode, decode_selector
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__   # the official one is no_grad

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0:20", help="a:b pair range to train")
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--costs", default="", help="costs_a_b.npy from exact_cost135.py --dump-costs (same range)")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--init-logit", type=float, default=3.0)
ap.add_argument("--rate-w", type=float, default=1.0)
ap.add_argument("--pose0", type=float, default=6.884e-6, help="pose level for the sqrt slope (#135 official)")
ap.add_argument("--out", default=str(HERE / "maps_run"))
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
ORIG = 37_545_489
a, b = map(int, args.frames.split(":"))
n = b - a
W_SEG = 100.0 / 600
W_POSE = (10.0 / (2.0 * math.sqrt(10.0 * args.pose0))) / 600
W_RATE = 25.0 / ORIG / 8.0 * args.rate_w

comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96)
rend.load_state_dict(comp["renderer"])
rend = rend.to(dev).eval()
for p in rend.parameters():
    p.requires_grad_(False)
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in net.parameters():
    p.requires_grad_(False)
tg = torch.load(args.targets)
seg_t = tg["seg"][a:b].to(dev).long()
pose_t = tg["pose"][a:b, :6].to(dev).float()

# frame 0 of every pair: #135's carrier exactly as inflate renders it, plus the 14-byte selector
basis = R.normalized_basis(comp["basis_raw"].to(dev))
coef = comp["coefficients"].to(dev)
with torch.no_grad():
    car = torch.einsum("bk,kchw->bchw", coef[a:b], basis) / math.sqrt(R.CARRIER_DIM)
    f0 = F.interpolate((127.5 + R.CARRIER_AMPLITUDE * car).clamp(0, 255).round(),
                       size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255).round()
f0 = f0.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
modes, sel = decode_selector(comp["selector"])
for i in range(n):
    m = sel[a + i]
    f0[i] = apply_pixel_mode(f0[i:i + 1].copy(), modes[m])[0]
f0 = torch.from_numpy(f0).to(dev).permute(0, 3, 1, 2).float()      # (n,3,H,W) camera res

maps0 = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)[a:b].copy()
m0 = torch.from_numpy(maps0).to(dev).long()
edge = torch.zeros_like(m0, dtype=torch.bool)
edge[:, 1:] |= m0[:, 1:] != m0[:, :-1]; edge[:, :-1] |= m0[:, :-1] != m0[:, 1:]
edge[:, :, 1:] |= m0[:, :, 1:] != m0[:, :, :-1]; edge[:, :, :-1] |= m0[:, :, :-1] != m0[:, :, 1:]
band = F.max_pool2d(edge.float()[:, None], 2 * args.band + 1, 1, args.band)[:, 0] > 0
print(f"pairs {a}:{b}  band pixels {int(band.sum()):,} of {band.numel():,} ({band.float().mean()*100:.2f}%)")

costs = None
if args.costs:
    costs = torch.from_numpy(np.load(args.costs).astype(np.float32)).to(dev)       # (n,5,H,W) bits
    assert costs.shape[0] == n

logits = torch.full((n, 5, R.EVAL_H, R.EVAL_W), 0.0, device=dev)
logits.scatter_(1, m0[:, None], args.init_logit)
logits = logits.requires_grad_(True)
opt = torch.optim.Adam([logits], lr=args.lr)
onehot0 = F.one_hot(m0, 5).permute(0, 3, 1, 2).float()


def soft_map(i):
    q = logits[i:i + 1].softmax(1)
    hard = F.one_hot(q.argmax(1), 5).permute(0, 3, 1, 2).float()
    st = hard + q - q.detach()
    bm = band[i:i + 1, None]
    return torch.where(bm, st, onehot0[i:i + 1]), torch.where(bm, q, onehot0[i:i + 1])


def render_frame1(onehot, i):
    value = torch.einsum("bchw,cd->bdhw", onehot, rend.token_embed.weight)
    value = rend.coord_mix(torch.cat([value, rend.coordinates(1, dev, value.dtype)], 1))
    frame = rend.frame_embed(torch.tensor([a + i], device=dev))
    for blk in rend.blocks:
        value = blk(value, frame)
    x = torch.sigmoid(rend.head(F.gelu(value))) * 255.0
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def pair_terms(i, train=True):
    s, q = soft_map(i)
    f1 = render_frame1(s, i)
    x = torch.stack([f0[i:i + 1], f1], 1).permute(0, 1, 3, 4, 2)         # b t h w c
    pin, sin_ = net.preprocess_input(x)
    seg_logits = net.segnet(sin_)
    tgt = seg_t[i:i + 1]
    true = seg_logits.gather(1, tgt[:, None])[:, 0]
    other = seg_logits.scatter(1, tgt[:, None], -1e9).amax(1)
    margin = true - other
    seg_soft = torch.sigmoid(-margin / args.tau).mean()
    seg_hard = (margin <= 0).float().mean()
    pose = (net.posenet(pin)["pose"][:, :6] - pose_t[i:i + 1]).pow(2).mean()
    bits = (q * costs[i:i + 1]).sum() if costs is not None else torch.zeros((), device=dev)
    return seg_soft, seg_hard, pose, bits


def evaluate():
    tot = {"seg": 0.0, "pose": 0.0, "bits": 0.0, "changed": 0}
    with torch.no_grad():
        for i in range(n):
            _, sh, po, bi = pair_terms(i)
            tot["seg"] += sh.item(); tot["pose"] += po.item(); tot["bits"] += bi.item()
        tot["changed"] = int((logits.argmax(1) != m0).sum())
    score = W_SEG * tot["seg"] + W_POSE * tot["pose"] + 25.0 / ORIG * tot["bits"] / 8.0
    return tot, score


out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
base, base_score = evaluate()
print(f"start: seg {base['seg']/n:.6f}  pose {base['pose']/n:.3e}  est bits {base['bits']:,.0f}  "
      f"local score {base_score:.6f}", flush=True)
t0 = time.time()
for step in range(1, args.steps + 1):
    order = torch.randperm(n).tolist()
    for i in order:
        ss, _, po, bi = pair_terms(i)
        loss = W_SEG * ss + W_POSE * po + W_RATE * bi
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if step % 10 == 0 or step == args.steps:
        cur, cur_score = evaluate()
        print(f"step {step}: seg {cur['seg']/n:.6f}  pose {cur['pose']/n:.3e}  est bits {cur['bits']:,.0f}  "
              f"changed px {cur['changed']:,}  local score {cur_score:.6f} (delta {cur_score-base_score:+.6f})  "
              f"{time.time()-t0:.0f}s", flush=True)
        hard = logits.argmax(1).to(torch.uint8).cpu().numpy()
        full = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)
        full[a:b] = hard
        full.tofile(out / "tokens_trained.u8")
json.dump({"frames": args.frames, "start": base, "start_score": base_score}, open(out / "run.json", "w"))
