# the CONNECTED joint trainer on #135: token maps + renderer + grey basis + 600x12 coefficients in ONE backward.
#
# built on train_maps135.py (the other session's map trainer: band logits, straight-through hard map, exact
# per-pixel byte prices) with the three things it freezes made trainable:
#   renderer  4-bit weights on #135's STORED grid (pr135_grid.pt), STE, lr in quantization-code units
#   basis     the 12 grey carrier images, shared by every pair -> its own optimizer, ONE step per epoch
#   coef      600x12, one row per pair -> one parameter per row so Adam never steps an unsampled row
# the slave (frame 0) is rendered differentiably with STE rounding and the 14-byte selector in torch, so the
# training forward is the scored function, not a smooth stand-in.
#
# monotone by construction: every epoch the whole range is scored on the hard path; the best state (maps,
# renderer, basis, coef) is kept and a worse epoch is reverted, so the reported result can never be worse
# than #135. --train-ren / --train-carrier off reproduces the maps-only method for an A/B on the same frames.
#
# loss per pair, in score units (same weights as train_maps135.py):
#   seg  100/600 * expected flips     pose  dscore/dpose / 600 * mse     rate  25/ORIG/8 * sum(q * bits)
# rate is first-order (neighbour effects ignored); the real byte check is exact_cost135.py --tokens.
import argparse, sys, math, time, json, copy
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
from runtime.frame0_selector import decode_selector
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0:20")
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--costs", default=str(HERE / "pr135_cost/costs_0_20.npy"))
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr-map", type=float, default=0.05)
ap.add_argument("--init-logit", type=float, default=3.0)
ap.add_argument("--train-ren", action="store_true")
ap.add_argument("--lr-ren", type=float, default=0.003, help="quantization codes per Adam step")
ap.add_argument("--train-carrier", action="store_true")
ap.add_argument("--lr-coef", type=float, default=1e-4)
ap.add_argument("--lr-basis", type=float, default=3e-5)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--pose0", type=float, default=6.884e-6)
ap.add_argument("--out", default=str(HERE / "joint4_run"))
ap.add_argument("--tag", default="J4")
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
ORIG = 37_545_489
fa, fb = map(int, args.frames.split(":"))
n = fb - fa
W_SEG = 100.0 / 600
W_POSE = (10.0 / (2.0 * math.sqrt(10.0 * args.pose0))) / 600
W_RATE = 25.0 / ORIG / 8.0

comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
grid = torch.load(HERE / "extracted/pr135_grid.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96)
rend.load_state_dict(comp["renderer"])
rend = rend.to(dev).eval()
for p in rend.parameters():
    p.requires_grad_(args.train_ren)
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in net.parameters():
    p.requires_grad_(False)
tg = torch.load(args.targets)
seg_t = tg["seg"][fa:fb].to(dev).long()
pose_t = tg["pose"][fa:fb, :6].to(dev).float()

# ---------------- renderer on #135's stored 4-bit grid
GRID = {}
for name, p in rend.named_parameters():
    e = grid.get(name)
    if e is None:
        GRID[name] = None
        continue
    shp = [1] * p.ndim
    shp[e["axis"]] = e["scales"].numel()
    GRID[name] = (e["scales"].reshape(shp).to(dev), int(e["limit"]))


def q(name, v):
    ent = GRID[name]
    if ent is None:
        r = v.to(torch.float16).float()
        return v + (r - v).detach()
    s, lim = ent
    c = (v / s).clamp(-lim, lim)
    return (c + (c.round() - c).detach()) * s


def hard_ste(x):
    y = x.clamp(0.0, 255.0)
    return y + (y.round() - y).detach()


def render_frame1(onehot, i):
    P = {k: q(k, v) for k, v in rend.named_parameters()}
    value = torch.einsum("bchw,cd->bdhw", onehot, P["token_embed.weight"])
    value = torch.func.functional_call(rend.coord_mix, {"weight": P["coord_mix.weight"], "bias": P["coord_mix.bias"]},
                                       (torch.cat([value, rend.coordinates(1, dev, value.dtype)], 1),))
    frame = F.embedding(torch.tensor([fa + i], device=dev), P["frame_embed.weight"])
    for bi, blk in enumerate(rend.blocks):
        sub = {k.split(".", 2)[2]: P[k] for k in P if k.startswith("blocks.%d." % bi)}
        value = torch.func.functional_call(blk, sub, (value, frame))
    x = torch.sigmoid(torch.func.functional_call(rend.head, {"weight": P["head.weight"], "bias": P["head.bias"]},
                                                 (F.gelu(value),))) * 255.0
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False)
    return hard_ste(x)


# ---------------- carrier: shared grey basis + one coefficient row per pair
modes, sel = decode_selector(comp["selector"])
basis_raw = comp["basis_raw"].to(dev).float().clone().requires_grad_(args.train_carrier)
coef_rows = [comp["coefficients"][fa + i].to(dev).float().clone().requires_grad_(args.train_carrier)
             for i in range(n)]


def selector_one(x, fid):
    m = modes[int(sel[fid])]
    if m.kind == 0:
        return x
    if m.kind == 5:
        return torch.roll(x, shifts=(m.b, m.a), dims=(2, 3))
    if m.kind == 6:
        yy = torch.arange(x.shape[2], device=dev).view(-1, 1)
        xx = torch.arange(x.shape[3], device=dev).view(1, -1)
        if m.a == 0:
            sg = ((yy + xx) & 1) * 2 - 1
        elif m.a == 1:
            sg = (yy & 1) * 2 - 1 + 0 * xx
        elif m.a == 2:
            sg = (xx & 1) * 2 - 1 + 0 * yy
        else:
            sg = (((yy >> 2) + (xx >> 2)) & 1) * 2 - 1
        delta = (sg * m.b).to(x.dtype)[None, None]
    elif m.kind == 3:
        delta = float(m.a)
    elif m.kind == 4:
        delta = torch.tensor([m.a, m.b, m.c], dtype=x.dtype, device=dev).view(1, 3, 1, 1)
    else:
        raise ValueError(m.kind)
    return hard_ste(x + delta)


def render_frame0(i, basis_n):
    car = torch.einsum("k,kchw->chw", coef_rows[i], basis_n)[None] / math.sqrt(R.CARRIER_DIM)
    x = hard_ste(127.5 + R.CARRIER_AMPLITUDE * car)
    x = hard_ste(F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False))
    return selector_one(x, fa + i)


# ---------------- token maps: band logits, straight-through hard map
maps0 = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)[fa:fb].copy()
m0 = torch.from_numpy(maps0).to(dev).long()
edge = torch.zeros_like(m0, dtype=torch.bool)
edge[:, 1:] |= m0[:, 1:] != m0[:, :-1]; edge[:, :-1] |= m0[:, :-1] != m0[:, 1:]
edge[:, :, 1:] |= m0[:, :, 1:] != m0[:, :, :-1]; edge[:, :, :-1] |= m0[:, :, :-1] != m0[:, :, 1:]
band = F.max_pool2d(edge.float()[:, None], 2 * args.band + 1, 1, args.band)[:, 0] > 0
costs = torch.from_numpy(np.load(args.costs).astype(np.float32)).to(dev)
assert costs.shape[0] == n, "cost dump must cover exactly --frames"
logits = torch.zeros((n, 5, R.EVAL_H, R.EVAL_W), device=dev)
logits.scatter_(1, m0[:, None], args.init_logit)
logits.requires_grad_(True)
onehot0 = F.one_hot(m0, 5).permute(0, 3, 1, 2).float()


def soft_map(i):
    qd = logits[i:i + 1].softmax(1)
    hard = F.one_hot(qd.argmax(1), 5).permute(0, 3, 1, 2).float()
    st = hard + qd - qd.detach()
    bm = band[i:i + 1, None]
    return torch.where(bm, st, onehot0[i:i + 1]), torch.where(bm, qd, onehot0[i:i + 1])


def pair_terms(i, basis_n):
    s, qd = soft_map(i)
    f1 = render_frame1(s, i)
    f0 = render_frame0(i, basis_n)
    x = torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2)
    pin, sin_ = net.preprocess_input(x)
    sl = net.segnet(sin_)
    tgt = seg_t[i:i + 1]
    margin = sl.gather(1, tgt[:, None])[:, 0] - sl.scatter(1, tgt[:, None], -1e9).amax(1)
    pose = (net.posenet(pin)["pose"][:, :6] - pose_t[i:i + 1]).pow(2).mean()
    bits = (qd * costs[i:i + 1]).sum()
    return torch.sigmoid(-margin / args.tau).mean(), (margin <= 0).float().mean(), pose, bits


def evaluate():
    """Hard-path score. Bytes are priced on the HARD map -- the thing that actually gets encoded.

    The earlier version (and train_maps135.py's) summed q * costs over the SOFT probabilities. At init each
    band pixel carries ~17% mass on the other four classes, so that "cost" was inflated (908k bits for 20
    frames, more than #135's whole 600-frame stream) and fell as the logits sharpened while the hard map did
    not change at all: 0 pixels changed, seg and pose identical, "score" -0.034. The acceptance gate then
    kept a state with 19x worse pose because the fictional byte saving outweighed it.
    """
    tot = {"seg": 0.0, "pose": 0.0, "bits": 0.0}
    with torch.no_grad():
        bn = R.normalized_basis(basis_raw)
        hard = logits.argmax(1)
        hard = torch.where(band, hard, m0)
        for i in range(n):
            _, sh, po, _ = pair_terms(i, bn)
            tot["seg"] += sh.item(); tot["pose"] += po.item()
            tot["bits"] += costs[i].gather(0, hard[i:i + 1]).sum().item()
    score = W_SEG * tot["seg"] + W_POSE * tot["pose"] + 25.0 / ORIG * tot["bits"] / 8.0
    return tot, score


def code_step(name, p):
    ent = GRID[name]
    return p.detach().abs().mean().clamp_min(1e-8).item() * 1e-3 if ent is None else float(ent[0].mean())


opt_map = torch.optim.Adam([logits], lr=args.lr_map)
opt_ren = (torch.optim.Adam([{"params": [p], "lr": args.lr_ren * code_step(k, p)}
                             for k, p in rend.named_parameters()]) if args.train_ren else None)
opt_coef = torch.optim.Adam(coef_rows, lr=args.lr_coef) if args.train_carrier else None
opt_basis = torch.optim.Adam([basis_raw], lr=args.lr_basis) if args.train_carrier else None


def state():
    return {"logits": logits.detach().clone(),
            "ren": {k: v.detach().clone() for k, v in rend.named_parameters()},
            "basis": basis_raw.detach().clone(), "coef": [c.detach().clone() for c in coef_rows]}


def restore(st):
    with torch.no_grad():
        logits.copy_(st["logits"])
        for k, v in rend.named_parameters():
            v.copy_(st["ren"][k])
        basis_raw.copy_(st["basis"])
        for c, c0 in zip(coef_rows, st["coef"]):
            c.copy_(c0)
    for o in (opt_map, opt_ren, opt_coef, opt_basis):
        if o is not None:
            o.state.clear()


# wiring check: the manual differentiable forward must equal the module's own forward on the shipped weights
with torch.no_grad():
    ref = F.interpolate(rend(m0[0:1], torch.tensor([fa], device=dev)), size=(R.CAMERA_H, R.CAMERA_W),
                        mode="bilinear", align_corners=False).clamp(0, 255).round()
    mine = render_frame1(onehot0[0:1], 0)
    diff = (ref - mine).abs().max().item()
print(f"[{args.tag}] renderer wiring check: max |module - manual| = {diff:.3g}", flush=True)
if diff > 0:
    print(f"[{args.tag}] WIRING MISMATCH -- refusing to train", flush=True)
    sys.exit(1)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
base, base_score = evaluate()
print(f"[{args.tag}] frames {args.frames}  train: maps{' +renderer' if args.train_ren else ''}"
      f"{' +basis +coef' if args.train_carrier else ''}  band px {int(band.sum()):,}", flush=True)
print(f"[{args.tag}] start: seg {base['seg']/n:.6f} pose {base['pose']/n:.4e} bits {base['bits']:,.0f} "
      f"local score {base_score:.6f}", flush=True)
best_score, best_state, best_tot = base_score, state(), base
t0 = time.time()
for ep in range(1, args.epochs + 1):
    if opt_basis is not None:
        opt_basis.zero_grad(set_to_none=True)
    for i in torch.randperm(n).tolist():
        bn = R.normalized_basis(basis_raw)
        ss, _, po, bi = pair_terms(i, bn)
        loss = W_SEG * ss + W_POSE * po + W_RATE * bi
        for o in (opt_map, opt_ren, opt_coef):
            if o is not None:
                o.zero_grad(set_to_none=True)
        loss.backward()                          # basis grad accumulates over the whole epoch
        for o in (opt_map, opt_ren, opt_coef):
            if o is not None:
                o.step()
    if opt_basis is not None:
        opt_basis.step()
    cur, cur_score = evaluate()
    changed = int((logits.argmax(1) != m0).sum())
    if cur_score < best_score:
        best_score, best_state, best_tot = cur_score, state(), cur
        verdict = "BEST"
    else:
        restore(best_state)
        verdict = "reverted"
    print(f"[{args.tag}] ep {ep:3d}: seg {cur['seg']/n:.6f} pose {cur['pose']/n:.4e} bits {cur['bits']:,.0f} "
          f"changed px {changed:,}  local {cur_score:.6f}  best {best_score:.6f} "
          f"(delta {best_score - base_score:+.6f})  {verdict}  {time.time()-t0:.0f}s", flush=True)
    json.dump({"tag": args.tag, "frames": args.frames, "start_score": base_score, "best_score": best_score,
               "delta": best_score - base_score, "best": best_tot, "start": base},
              open(out / f"{args.tag}.json", "w"), indent=1)
    torch.save(best_state, out / f"{args.tag}_best.pt")
print(f"[{args.tag}] FINAL local score {best_score:.6f}  start {base_score:.6f}  delta {best_score-base_score:+.6f}",
      flush=True)
