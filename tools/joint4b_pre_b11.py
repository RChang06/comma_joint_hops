# B, iteration 2: the connected joint trainer on #135 with the four flaws of joint4.py fixed.
#
#   1. the map actually moves      lower logit lead, larger map lr, and a per-frame cap on flips so it can't
#                                  blow up (joint4: 0 pixels changed in 40 epochs -> B was never really joint)
#   2. per-part acceptance         per-frame parts (map, coef) accepted per frame; shared parts (renderer,
#                                  basis) accepted separately -- a bad carrier step can't veto a good map edit
#   3. step-size backoff           each part's lr halves when rejected, grows x1.25 when accepted, instead of
#                                  re-proposing the same rejected move forever
#   4. everything on its grid      coef = shipped + integer int12 steps; basis = 5-bit codes (4-bit for
#                                  patterns 2,5,9) x per-pattern step; renderer = #135's stored 4-bit grid.
#                                  each part is parameterized in its OWN code units, so lrs mean "codes".
#
# score per frame, linearized (same weights as train_maps135.py):
#   100/600 * seg  +  dscore/dpose / 600 * pose  +  25/ORIG/8 * bits   (bits priced on the HARD map)
# monotone: nothing is kept unless the exact hard-path score improves.
import argparse, sys, math, time, json
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
from runtime.frame0_selector import decode_selector
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0:20")
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--costs", default=str(HERE / "pr135_cost/costs_0_20.npy"))
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--init-logit", type=float, default=2.0)
ap.add_argument("--lr-map", type=float, default=0.3)
ap.add_argument("--flip-cap", type=int, default=40, help="max changed band pixels per frame per epoch")
ap.add_argument("--map-mode", default="pixel", choices=["batch", "pixel"],
                help="pixel: each gradient-proposed flip is scored exactly on its own and kept only if it pays")
ap.add_argument("--pix-k", type=int, default=8, help="pixel mode: candidates tried per frame per epoch")
ap.add_argument("--cand", default="grad", choices=["grad", "segerr", "both"],
                help="where pixel candidates come from: gradient margin, or SegNet's wrong pixels "
                     "(propose token := target class there), or both")
ap.add_argument("--lr-ren", type=float, default=0.01, help="renderer codes per Adam step")
ap.add_argument("--lr-coef", type=float, default=0.6, help="int12 codes per Adam step")
ap.add_argument("--lr-basis", type=float, default=0.6, help="basis codes per (once-per-epoch) step")
ap.add_argument("--parts", default="map,ren,coef,basis")
ap.add_argument("--bo-down", type=float, default=0.7)
ap.add_argument("--bo-up", type=float, default=1.3)
ap.add_argument("--bo-floor", type=float, default=0.05)
ap.add_argument("--restart", type=int, default=8, help="epochs stuck at the floor before resetting to 1x")
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--pose0", type=float, default=6.884e-6)
ap.add_argument("--resume", default="", help="checkpoint from a previous run (acc_map, ren, bcode, coff)")
ap.add_argument("--resume-frames", default="", help="a:b frame range the checkpoint's map/coef cover")
ap.add_argument("--dump-map", default="", help="write the full 600-frame accepted map (uint8) here and exit "
                                              "after the start evaluation if --epochs 0")
ap.add_argument("--out", default=str(HERE / "joint4b_run"))
ap.add_argument("--tag", default="B2")
args = ap.parse_args()
PARTS = set(args.parts.split(","))

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
    p.requires_grad_("ren" in PARTS)
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in net.parameters():
    p.requires_grad_(False)
tg = torch.load(args.targets)
seg_t = tg["seg"][fa:fb].to(dev).long()
pose_t = tg["pose"][fa:fb, :6].to(dev).float()


def rste(x):
    return x + (x.round() - x).detach()


def hard_ste(x):
    y = x.clamp(0.0, 255.0)
    return y + (y.round() - y).detach()


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
    return rste((v / s).clamp(-lim, lim)) * s


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
    return hard_ste(F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False))


# ---------------- carrier, parameterized in storage-code units
basis0 = comp["basis_raw"].to(dev).double()
B_STEP, B_LIM = [], []
for k in range(basis0.shape[0]):
    u = torch.unique(basis0[k]); d = torch.diff(u)
    s = d[d > 1e-12].min()
    B_STEP.append(s); B_LIM.append(int((basis0[k] / s).abs().round().max()))
B_STEP = torch.stack(B_STEP).float().view(-1, 1, 1, 1)
B_LIM = torch.tensor(B_LIM, device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
bcode = (comp["basis_raw"].to(dev).float() / B_STEP).clone().requires_grad_("basis" in PARTS)   # 5/4-bit codes


def basis_value():
    return torch.maximum(torch.minimum(rste(bcode), B_LIM), -B_LIM) * B_STEP


cbase = comp["coefficients"][fa:fb].to(dev).float()
c_all = comp["coefficients"].to(dev).double()
C_STEP = []
for k in range(c_all.shape[1]):
    u = torch.unique(c_all[:, k]); d = torch.diff(u)
    C_STEP.append(d[d > 1e-12].min())
C_STEP = torch.stack(C_STEP).float()
coff = [torch.zeros(12, device=dev, requires_grad="coef" in PARTS) for _ in range(n)]     # int12 offsets


def coef_value(i):
    return cbase[i] + rste(coff[i]) * C_STEP


modes, sel = decode_selector(comp["selector"])


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
    car = torch.einsum("k,kchw->chw", coef_value(i), basis_n)[None] / math.sqrt(R.CARRIER_DIM)
    x = hard_ste(127.5 + R.CARRIER_AMPLITUDE * car)
    x = hard_ste(F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False))
    return selector_one(x, fa + i)


# ---------------- token maps
maps0 = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)[fa:fb].copy()
m0 = torch.from_numpy(maps0).to(dev).long()
edge = torch.zeros_like(m0, dtype=torch.bool)
edge[:, 1:] |= m0[:, 1:] != m0[:, :-1]; edge[:, :-1] |= m0[:, :-1] != m0[:, 1:]
edge[:, :, 1:] |= m0[:, :, 1:] != m0[:, :, :-1]; edge[:, :, :-1] |= m0[:, :, :-1] != m0[:, :, 1:]
band = F.max_pool2d(edge.float()[:, None], 2 * args.band + 1, 1, args.band)[:, 0] > 0
if args.costs == "none":
    costs = torch.zeros((n, 5, R.EVAL_H, R.EVAL_W), device=dev, dtype=torch.float16)
else:
    costs = torch.from_numpy(np.load(args.costs).astype(np.float32)).to(dev)
assert costs.shape[0] == n
logits = torch.zeros((n, 5, R.EVAL_H, R.EVAL_W), device=dev)
logits.scatter_(1, m0[:, None], args.init_logit)
logits.requires_grad_("map" in PARTS)
acc_map = m0.clone()                      # the ACCEPTED hard map; logits only propose changes to it


def soft_map(i):
    qd = logits[i:i + 1].softmax(1)
    hard = F.one_hot(qd.argmax(1), 5).permute(0, 3, 1, 2).float()
    st = hard + qd - qd.detach()
    bm = band[i:i + 1, None]
    oh = F.one_hot(acc_map[i:i + 1], 5).permute(0, 3, 1, 2).float()
    return torch.where(bm, st, oh), torch.where(bm, qd, oh)


def pair_loss(i, basis_n):
    s, qd = soft_map(i)
    f1 = render_frame1(s, i)
    f0 = render_frame0(i, basis_n)
    pin, sin_ = net.preprocess_input(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
    sl = net.segnet(sin_)
    tgt = seg_t[i:i + 1]
    margin = sl.gather(1, tgt[:, None])[:, 0] - sl.scatter(1, tgt[:, None], -1e9).amax(1)
    pose = (net.posenet(pin)["pose"][:, :6] - pose_t[i:i + 1]).pow(2).mean()
    return (W_SEG * torch.sigmoid(-margin / args.tau).mean() + W_POSE * pose
            + W_RATE * (qd * costs[i:i + 1]).sum())


@torch.no_grad()
def eval_one(i, map_i, bn):
    """Exact hard-path score of ONE frame for a given hard map row (1,H,W)."""
    oh = F.one_hot(map_i, 5).permute(0, 3, 1, 2).float()
    f1 = render_frame1(oh, i)
    f0 = render_frame0(i, bn)
    pin, sin_ = net.preprocess_input(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
    sh = (net.segnet(sin_).argmax(1) != seg_t[i:i + 1]).float().mean()
    po = (net.posenet(pin)["pose"][:, :6] - pose_t[i:i + 1]).pow(2).mean()
    bi = costs[i].gather(0, map_i).sum()
    return W_SEG * sh + W_POSE * po + W_RATE * bi, torch.stack([sh, po, bi])


@torch.no_grad()
def eval_frames(map_hard):
    """Exact hard-path score per frame for a given hard map (renderer/carrier = current params)."""
    bn = R.normalized_basis(basis_value())
    out = torch.empty(n, device=dev)
    parts = torch.empty(n, 3, device=dev)
    for i in range(n):
        oh = F.one_hot(map_hard[i:i + 1], 5).permute(0, 3, 1, 2).float()
        f1 = render_frame1(oh, i)
        f0 = render_frame0(i, bn)
        pin, sin_ = net.preprocess_input(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
        sh = (net.segnet(sin_).argmax(1) != seg_t[i:i + 1]).float().mean()
        po = (net.posenet(pin)["pose"][:, :6] - pose_t[i:i + 1]).pow(2).mean()
        bi = costs[i].gather(0, map_hard[i:i + 1]).sum()
        parts[i] = torch.stack([sh, po, bi])
        out[i] = W_SEG * sh + W_POSE * po + W_RATE * bi
    return out, parts


def proposed_map():
    """Hard map proposed by the logits, capped to the flip-cap largest-margin changes per frame."""
    with torch.no_grad():
        prop = torch.where(band, logits.argmax(1), acc_map)
        for i in range(n):
            ch = prop[i] != acc_map[i]
            k = int(ch.sum())
            if k > args.flip_cap:
                lg = logits[i]
                margin = lg.gather(0, prop[i:i + 1])[0] - lg.gather(0, acc_map[i:i + 1])[0]
                margin = torch.where(ch, margin, torch.full_like(margin, -1e9))
                keep = torch.zeros_like(ch)
                keep.view(-1)[margin.view(-1).topk(args.flip_cap).indices] = True
                prop[i] = torch.where(keep, prop[i], acc_map[i])
        return prop


# ---------------- optimizers (each part in its own code units) with backoff scales
def code_step(name, p):
    ent = GRID[name]
    return p.detach().abs().mean().clamp_min(1e-8).item() * 1e-3 if ent is None else float(ent[0].mean())


LR = {"map": args.lr_map, "ren": args.lr_ren, "coef": args.lr_coef, "basis": args.lr_basis}
SCALE = {k: 1.0 for k in LR}
opt = {}
if "map" in PARTS:
    opt["map"] = torch.optim.Adam([logits], lr=LR["map"])
if "ren" in PARTS:
    opt["ren"] = torch.optim.Adam([{"params": [p], "lr": LR["ren"] * code_step(k, p), "base": code_step(k, p)}
                                   for k, p in rend.named_parameters()])
if "coef" in PARTS:
    opt["coef"] = torch.optim.Adam(coff, lr=LR["coef"])
if "basis" in PARTS:
    opt["basis"] = torch.optim.Adam([bcode], lr=LR["basis"])


def set_lr(part):
    if part not in opt:
        return
    for g in opt[part].param_groups:
        g["lr"] = LR[part] * SCALE[part] * g.get("base", 1.0)


STUCK = {}


def backoff(part, accepted):
    """iteration 2 halved on every rejection with a 1e-3 floor: everything froze by epoch 15. Now gentler,
    with a restart so a part stuck at the floor gets a fresh full-size step instead of dying."""
    if accepted:
        SCALE[part] = min(SCALE[part] * args.bo_up, 4.0)
        STUCK[part] = 0
    else:
        SCALE[part] = max(SCALE[part] * args.bo_down, args.bo_floor)
        if SCALE[part] <= args.bo_floor:
            STUCK[part] = STUCK.get(part, 0) + 1
            if STUCK[part] >= args.restart:
                SCALE[part] = 1.0
                STUCK[part] = 0
    set_lr(part)


# ---------------- wiring check
with torch.no_grad():
    ref = F.interpolate(rend(m0[0:1], torch.tensor([fa], device=dev)), size=(R.CAMERA_H, R.CAMERA_W),
                        mode="bilinear", align_corners=False).clamp(0, 255).round()
    diff = (ref - render_frame1(F.one_hot(m0[0:1], 5).permute(0, 3, 1, 2).float(), 0)).abs().max().item()
    b_diff = (basis_value() - comp["basis_raw"].to(dev)).abs().max().item()
print(f"[{args.tag}] wiring: renderer |module-manual| {diff:.3g}  basis grid |recon-shipped| {b_diff:.2e}", flush=True)
if diff > 0 or b_diff > 1e-4:
    print(f"[{args.tag}] WIRING MISMATCH -- refusing to train", flush=True)
    sys.exit(1)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
orig_f, orig_parts = eval_frames(acc_map)
if args.resume:
    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    ra, rb = map(int, args.resume_frames.split(":"))
    lo, hi = max(ra, fa), min(rb, fb)
    with torch.no_grad():
        if hi > lo:
            acc_map[lo - fa:hi - fa] = ck["acc_map"][lo - ra:hi - ra].to(dev).long()
            for j in range(lo, hi):
                coff[j - fa].copy_(ck["coff"][j - ra].to(dev))
        for k, v in rend.named_parameters():
            v.copy_(ck["ren"][k].to(dev))
        bcode.copy_(ck["bcode"].to(dev))
        logits.zero_()
        logits.scatter_(1, acc_map[:, None], args.init_logit)
    print(f"[{args.tag}] resumed from {args.resume} (map/coef frames {args.resume_frames})", flush=True)
if args.dump_map:
    full = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy()
    full[fa:fb] = acc_map.to(torch.uint8).cpu().numpy()
    full.tofile(args.dump_map)
    print(f"[{args.tag}] wrote full accepted map to {args.dump_map}", flush=True)
cur_f, cur_parts = eval_frames(acc_map)
print(f"[{args.tag}] vs original #135 on these frames: seg {orig_parts[:,0].mean():.7f} -> {cur_parts[:,0].mean():.7f}"
      f"  pose {orig_parts[:,1].mean():.4e} -> {cur_parts[:,1].mean():.4e}"
      f"  pose term (sqrt) {math.sqrt(10*float(orig_parts[:,1].mean())):.7f} -> {math.sqrt(10*float(cur_parts[:,1].mean())):.7f}",
      flush=True)
best = float(cur_f.sum())
start = best
print(f"[{args.tag}] parts {sorted(PARTS)}  frames {args.frames}  band px {int(band.sum()):,}", flush=True)
print(f"[{args.tag}] start: seg {cur_parts[:,0].mean():.6f} pose {cur_parts[:,1].mean():.4e} "
      f"bits {cur_parts[:,2].sum():,.0f}  local {best:.7f}", flush=True)


def snap():
    return {"ren": {k: v.detach().clone() for k, v in rend.named_parameters()},
            "bcode": bcode.detach().clone(), "coff": [c.detach().clone() for c in coff]}


t0 = time.time()
for ep in range(1, args.epochs + 1):
    S = snap()
    # ---- training epoch: every part proposes
    for o in opt.values():
        o.zero_grad(set_to_none=True)
    for i in torch.randperm(n).tolist():
        bn = R.normalized_basis(basis_value())
        loss = pair_loss(i, bn)
        for k, o in opt.items():
            if k != "basis":
                o.zero_grad(set_to_none=True)
        loss.backward()                              # basis grad accumulates across the epoch
        for k, o in opt.items():
            if k != "basis":
                o.step()
    if "basis" in opt:
        opt["basis"].step()
    T = snap()
    log = []

    # ---- (a) maps, per frame, under the accepted renderer/carrier
    with torch.no_grad():
        for k, v in rend.named_parameters():
            v.copy_(S["ren"][k])
        bcode.copy_(S["bcode"])
        for c, c0 in zip(coff, S["coff"]):
            c.copy_(c0)
    if "map" in opt and args.map_mode == "pixel":
        bn_s = R.normalized_basis(basis_value())
        tried = kept = 0
        with torch.no_grad():
            for i in range(n):
                lg = logits[i]
                prop_i = torch.where(band[i], lg.argmax(0), acc_map[i])
                tried_idx = []
                if args.cand in ("grad", "both"):
                    ch = prop_i != acc_map[i]
                    if bool(ch.any()):
                        margin = lg.gather(0, prop_i[None])[0] - lg.gather(0, acc_map[i:i + 1])[0]
                        margin = torch.where(ch, margin, torch.full_like(margin, -1e9)).view(-1)
                        tried_idx += margin.topk(min(args.pix_k, int(ch.sum()))).indices.tolist()
                if args.cand in ("segerr", "both"):
                    # SegNet's wrong pixels under the accepted state: propose token := target class there
                    oh = F.one_hot(acc_map[i:i + 1], 5).permute(0, 3, 1, 2).float()
                    pin, sin_ = net.preprocess_input(torch.stack([render_frame0(i, bn_s), render_frame1(oh, i)], 1)
                                                     .permute(0, 1, 3, 4, 2))
                    wrong = (net.segnet(sin_).argmax(1)[0] != seg_t[i]) & (acc_map[i] != seg_t[i])
                    widx = torch.nonzero(wrong.view(-1))[:, 0].tolist()
                    for fi in widx:
                        prop_i.view(-1)[fi] = seg_t[i].view(-1)[fi]
                    tried_idx += [fi for fi in widx if fi not in set(tried_idx)]
                if not tried_idx:
                    continue
                for fi in tried_idx:
                    tried += 1
                    trial = acc_map[i:i + 1].clone()
                    trial.view(-1)[fi] = prop_i.view(-1)[fi]
                    v, pp = eval_one(i, trial, bn_s)
                    if float(v) < float(cur_f[i]) - 1e-15:
                        acc_map[i] = trial[0]
                        cur_f[i], cur_parts[i] = v, pp
                        kept += 1
                # re-anchor ONLY the pixels just tried, on whatever they now are in the accepted map; untried
                # pixels keep accumulating so they can become candidates in later epochs
                flat = logits[i].view(5, -1)
                idx_t = torch.tensor(tried_idx, device=dev)
                flat[:, idx_t] = 0.0
                flat[acc_map[i].view(-1)[idx_t], idx_t] = args.init_logit
        if tried:
            backoff("map", kept > 0)
        log.append(f"map px kept {kept}/{tried}")
    elif "map" in opt:
        prop = proposed_map()
        f_new, p_new = eval_frames(prop)
        differs = (prop != acc_map).flatten(1).any(1)
        good = (f_new < cur_f - 1e-12) & differs
        bad = differs & ~good
        changed = int(((prop != acc_map) & good[:, None, None]).sum())
        acc_map = torch.where(good[:, None, None], prop, acc_map)
        cur_f = torch.where(good, f_new, cur_f)
        cur_parts = torch.where(good[:, None], p_new, cur_parts)
        with torch.no_grad():                     # re-anchor logits on the accepted map for rejected frames
            for i in range(n):
                if bool(bad[i]):                  # tried a real change and it scored worse
                    logits[i].zero_()
                    logits[i].scatter_(0, acc_map[i:i + 1], args.init_logit)
        if bool(differs.any()):
            backoff("map", bool(good.any()))
        log.append(f"map {int(good.sum())}/{int(differs.sum())} tried ({changed} px)")

    # ---- (b) coefficients, per frame
    if "coef" in opt:
        with torch.no_grad():
            for c, c1 in zip(coff, T["coff"]):
                c.copy_(c1)
        f_new, p_new = eval_frames(acc_map)
        differs = torch.tensor([bool((S["coff"][i].round() != T["coff"][i].round()).any()) for i in range(n)],
                               device=dev)
        good = (f_new < cur_f - 1e-12) & differs
        with torch.no_grad():
            for i in range(n):
                if bool(differs[i]) and not bool(good[i]):
                    coff[i].copy_(S["coff"][i])
                    opt["coef"].state.pop(coff[i], None)
        cur_f = torch.where(good, f_new, cur_f)
        cur_parts = torch.where(good[:, None], p_new, cur_parts)
        if bool(differs.any()):
            backoff("coef", bool(good.any()))
        log.append(f"coef {int(good.sum())}/{int(differs.sum())} tried")

    # ---- (c) renderer, shared: whole-range accept
    if "ren" in opt:
        with torch.no_grad():
            for k, v in rend.named_parameters():
                v.copy_(T["ren"][k])
        with torch.no_grad():
            same = all(torch.equal(q(k, S["ren"][k]), q(k, T["ren"][k])) for k in S["ren"])
        if same:
            log.append("ren no-op")
        else:
            f_new, p_new = eval_frames(acc_map)
            ok = float(f_new.sum()) < float(cur_f.sum()) - 1e-12
            if ok:
                cur_f, cur_parts = f_new, p_new
            else:
                with torch.no_grad():
                    for k, v in rend.named_parameters():
                        v.copy_(S["ren"][k])
                opt["ren"].state.clear()
            backoff("ren", ok)
            log.append("ren " + ("ACC" if ok else "rej"))

    # ---- (d) basis, shared: whole-range accept
    if "basis" in opt:
        with torch.no_grad():
            bcode.copy_(T["bcode"])
        if torch.equal(S["bcode"].round(), T["bcode"].round()):
            log.append("basis no-op")
        else:
            f_new, p_new = eval_frames(acc_map)
            ok = float(f_new.sum()) < float(cur_f.sum()) - 1e-12
            if ok:
                cur_f, cur_parts = f_new, p_new
            else:
                with torch.no_grad():
                    bcode.copy_(S["bcode"])
                opt["basis"].state.clear()
            backoff("basis", ok)
            log.append("basis " + ("ACC" if ok else "rej"))

    best = float(cur_f.sum())
    print(f"[{args.tag}] ep {ep:3d}: {' | '.join(log)}  -> seg {cur_parts[:,0].mean():.6f} "
          f"pose {cur_parts[:,1].mean():.4e} bits {cur_parts[:,2].sum():,.0f}  local {best:.7f} "
          f"(delta {best-start:+.7f})  scales {', '.join('%s %.3g' % (k, SCALE[k]) for k in opt)}  "
          f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"tag": args.tag, "start": start, "best": best, "delta": best - start, "epoch": ep},
              open(out / f"{args.tag}.json", "w"))
    torch.save({"acc_map": acc_map.to(torch.uint8).cpu(), **snap()}, out / f"{args.tag}_best.pt")
print(f"[{args.tag}] FINAL local {best:.7f}  start {start:.7f}  delta {best-start:+.7f}", flush=True)
