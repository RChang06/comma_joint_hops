# joint training of #135's stored state against the score, with an exact differentiable rate term.
#
# trainable together, one loss (100*seg + sqrt(10*pose) + 25*bytes/ORIG):
#   class maps         5-way logits on boundary-band pixels, straight-through hard map in the forward pass
#   hpac               its float masters (the integer model is ste-rounded, as #130 trained it)
#   boundary table     the 125 corrections added to hpac's logits (#135 fitted it once by counting)
#   carrier strengths  600x12, in int12-code units, ste-rounded
#   gray, amplitude    the carrier's 127.5 / 64 constants live in decoder code, so moving them costs no bytes
# the rate is hpac's own probability of each pixel under the soft map as context: it includes every edit's effect
# on neighbours and on the next frame, which the per-pixel price list missed (4.6x low on the first search).
# safety: per-step flip cap on the map, and an exact gate every --gate steps (hard state, full 600-frame rate,
# seg and pose on all pairs) that rolls back to the last good state when the exact score gets worse.
# gate 0 (--check-rate): with the starting hard map the full-model bytes must equal exact_cost135.py's.
import argparse, sys, math, time, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
import hpac_integer as HI
from runtime import residual_archive as RA
from runtime.frame0_selector import apply_pixel_mode, decode_selector
import modules, frame_utils
from modules import DistortionNet
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "extracted/tokens_pr141.u8"))
ap.add_argument("--coefficients", default="", help="starting carrier strengths .npy (default #135's)")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--check-rate", action="store_true", help="only verify the differentiable rate against exact cost")
ap.add_argument("--grad-check", action="store_true", help="one step; report the gradient reaching every trainable group")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--window", type=int, default=4, help="consecutive pairs per step")
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--flip-cap", type=int, default=8, help="max map flips per frame per step")
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--lr-map", type=float, default=0.05)
ap.add_argument("--lr-hpac", type=float, default=0.02)
ap.add_argument("--lr-table", type=float, default=0.01)
ap.add_argument("--lr-coef", type=float, default=0.5)
ap.add_argument("--lr-const", type=float, default=0.01)
ap.add_argument("--lr-render", type=float, default=0.02, help="renderer, in int4-code units (0 = frozen)")
ap.add_argument("--lr-basis", type=float, default=0.02, help="carrier patterns, in 5-bit-code units (0 = frozen)")
ap.add_argument("--gate", type=int, default=200)
ap.add_argument("--out", default=str(HERE / "joint_run"))
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
ORIG = 37_545_489
N, H, W = R.N, R.EVAL_H, R.EVAL_W
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

# ---- fixed pieces
comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96); rend.load_state_dict(comp["renderer"]); rend = rend.to(dev).eval()
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in list(rend.parameters()) + list(net.parameters()):
    p.requires_grad_(False)
tg = torch.load(args.targets)
seg_t, pose_t = tg["seg"].to(dev).long(), tg["pose"][:, :6].to(dev).float()
basis = R.normalized_basis(comp["basis_raw"].to(dev))
modes, sel = decode_selector(comp["selector"])

# ---- trainable pieces
parts = RA.read_residual_archive(SUB / "archive.zip")
hpac = R.load_hpac(RA.materialize_ihs1(parts.hpac_blob, R), dev).train()
table = torch.tensor(np.asarray(parts.table.values, dtype=np.float32), device=dev).requires_grad_(True)   # (25,5)
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
c0 = torch.from_numpy(np.load(args.coefficients)).to(dev).float() if args.coefficients else c135
codes = (c0 / step).round().requires_grad_(True)
gray = torch.tensor(127.5, device=dev, requires_grad=True)
amp = torch.tensor(float(R.CARRIER_AMPLITUDE), device=dev, requires_grad=True)
from runtime.entropy.renderer_weight_codec import decode_wans1
records = decode_wans1(parts.semantic_blob)
r_master, r_scale = {}, {}
for rec in records:
    name = rec.schema.name
    if rec.codes is not None:                                   # int4 codes in [-7,7] * fp16 per-row scale
        shp = [1] * len(rec.schema.shape)
        shp[-1 if name.endswith("embed.weight") else 0] = rec.schema.scale_count
        r_scale[name] = torch.from_numpy(np.asarray(rec.scales, dtype=np.float32)).reshape(shp).to(dev)
        r_master[name] = torch.from_numpy(rec.codes.astype(np.float32)).to(dev).requires_grad_(args.lr_render > 0)
    else:                                                       # fp16 tensor
        r_master[name] = torch.from_numpy(np.asarray(rec.values, dtype=np.float32)).to(dev).requires_grad_(args.lr_render > 0)
r_codes0 = {k: v.detach().clone() for k, v in r_master.items() if k in r_scale}


def render_params():
    out = {}
    for k, v in r_master.items():
        if k in r_scale:
            out[k] = HI.ste_round(v).clamp(-7, 7) * r_scale[k]
        else:
            out[k] = v + (v.half().float() - v).detach()
    return out


from runtime.carrier_repack import split_frame0_selector_carrier as _split, materialize_cpr1 as _mat
_bs, _bc, _, _ = R.decode_compact_carrier(_mat(_split(parts.carrier_blob)[0], R),
                                          basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W,
                                          frames=N, dimensions=R.CARRIER_DIM)
b_scale = torch.from_numpy(np.asarray(_bs, dtype=np.float32)).to(dev)[:, None, None, None]
b_master = torch.from_numpy(np.asarray(_bc, dtype=np.float32).reshape(R.CARRIER_DIM, 3, R.CARRIER_H, R.CARRIER_W)).to(dev)
b_master.requires_grad_(args.lr_basis > 0)
b_codes0 = b_master.detach().clone()


def current_basis():
    return R.normalized_basis(HI.ste_round(b_master).clamp(-15, 15) * b_scale)


def entropy_bytes(codes):
    _, cnt = torch.unique(codes.round(), return_counts=True)
    pr = cnt.float() / cnt.sum()
    return float(-(cnt.float() * pr.log2()).sum() / 8)


def side_bytes_delta():
    # order-0 entropy change of the renderer int4 codes and the 5-bit pattern codes vs the start (estimate; the
    # final archive gets an exact re-encode with encode_wans1 / the carrier encoder)
    d = 0.0
    for k in r_codes0:
        d += entropy_bytes(HI.ste_round(r_master[k]).clamp(-7, 7).detach()) - entropy_bytes(r_codes0[k])
    d += entropy_bytes(HI.ste_round(b_master).clamp(-15, 15).detach()) - entropy_bytes(b_codes0)
    return d


maps0 = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint8).reshape(N, H, W).copy()).to(dev).long()


def band_of(m):
    e = torch.zeros_like(m, dtype=torch.bool)
    e[:, 1:] |= m[:, 1:] != m[:, :-1]; e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:, :, 1:] |= m[:, :, 1:] != m[:, :, :-1]; e[:, :, :-1] |= m[:, :, :-1] != m[:, :, 1:]
    return F.max_pool2d(e.float()[:, None], 2 * args.band + 1, 1, args.band)[:, 0] > 0


band = band_of(maps0)                                               # (N,H,W) fixed set of trainable pixels
logits = (F.one_hot(maps0, 5).permute(0, 3, 1, 2).float() * 3.0).requires_grad_(True)   # (N,5,H,W)
print(f"band pixels {int(band.sum()):,} ({band.float().mean()*100:.2f}%)", flush=True)


def onehot_st(sl):
    q = logits[sl].softmax(1)
    hard = F.one_hot(q.argmax(1), 5).permute(0, 3, 1, 2).float()
    b = band[sl][:, None]
    oh0 = F.one_hot(maps0[sl], 5).permute(0, 3, 1, 2).float()
    return torch.where(b, hard + q - q.detach(), oh0), torch.where(b, q, oh0)


# ---- hpac on soft one-hot maps (#135 uses norm_mode none, no norm gates; mirrors cpr1/hpac_integer.py)
def hpac_context(idx, prev_oh):
    m = hpac
    batch, _, height, width = prev_oh.shape
    patch_count = (height // m.P) * (width // m.P)
    emb = m.frame_codes()[idx]
    shift = HI.requantize(m.frame_shift(emb), 1, -m.activation_bound, m.activation_bound)
    shift = shift.view(batch, 1, m.ch, 1, 1).expand(batch, patch_count, m.ch, 1, 1).reshape(batch * patch_count, m.ch, 1, 1)
    past = HI.requantize(m.conv_past(prev_oh), 0, -m.activation_bound, m.activation_bound)
    pr, pc = height // m.P, width // m.P
    pooled = HI.ste_round(past.view(batch, m.ch, pr, m.P, pc, m.P).mean(dim=(3, 5)))
    pooled = HI.integer_activation(HI.requantize(m.spm_dw(pooled), 3, -m.activation_bound, m.activation_bound), m.activation)
    pooled = HI.requantize(m.spm_pw(pooled), 4, -m.activation_bound, m.activation_bound)
    spm = pooled.unsqueeze(3).unsqueeze(5).expand(batch, m.ch, pr, m.P, pc, m.P).contiguous().view(batch, m.ch, height, width)
    scale = HI.requantize(m.frame_scale(emb), 4, -8, 8)
    scale = scale.view(batch, 1, m.ch, 1, 1).expand(batch, patch_count, m.ch, 1, 1).reshape(batch * patch_count, m.ch, 1, 1)
    return shift, m._to_patches(past), scale, m._to_patches(spm)


def hpac_logits(cur_oh, ctx):
    m = hpac
    batch, _, height, width = cur_oh.shape
    pr, pc = height // m.P, width // m.P
    coords = m._patch_coord_grid(batch * pr * pc, cur_oh.device)
    hidden = HI.requantize(m.conv_a(torch.cat([m._to_patches(cur_oh), coords], 1)), 1, -m.activation_bound, m.activation_bound)
    shift, past, scale, spm = ctx
    hidden = HI.requantize(hidden * (16 + scale), 4, -m.activation_bound, m.activation_bound)
    past = HI.requantize(past + spm, 0, -m.activation_bound, m.activation_bound)
    hidden = HI.integer_activation(HI.requantize(hidden + shift + past, 0, -m.activation_bound, m.activation_bound), m.activation)
    hidden = HI.integer_activation(HI.requantize(m.conv_b1(hidden), 3, -m.activation_bound, m.activation_bound), m.activation)
    hidden = HI.integer_activation(HI.requantize(m.conv_b2(hidden), 3, -m.activation_bound, m.activation_bound), m.activation)
    lg = HI.requantize(m.head(hidden), 3, -32768, 32767)
    return m._from_patches(lg, batch, pr, pc) / 8.0


def boundary_buckets(prev_hard):
    # torch port of runtime/residual_archive.py::_boundary_buckets (4-neighbour, saturating at 4)
    edge = torch.zeros_like(prev_hard, dtype=torch.bool)
    edge[:, 1:] |= prev_hard[:, 1:] != prev_hard[:, :-1]; edge[:, :-1] |= prev_hard[:, :-1] != prev_hard[:, 1:]
    edge[:, :, 1:] |= prev_hard[:, :, 1:] != prev_hard[:, :, :-1]; edge[:, :, :-1] |= prev_hard[:, :, :-1] != prev_hard[:, :, 1:]
    res = torch.full_like(prev_hard, 4)
    res[edge] = 0
    act = edge.clone()
    for d in range(1, 4):
        g = act.clone()
        g[:, 1:] |= act[:, :-1]; g[:, :-1] |= act[:, 1:]; g[:, :, 1:] |= act[:, :, :-1]; g[:, :, :-1] |= act[:, :, 1:]
        act = g
        res[(res == 4) & act] = d
    return res


def frame_bits(f_idx, cur_oh, cur_q, prev_oh, prev_hard, first):
    # bits of frames f_idx coded with hpac + table: -sum_c q_c log2 p_c (= exact cost when q is hard)
    ctx = hpac_context(f_idx, prev_oh)
    base = hpac_logits(cur_oh, ctx)                                          # (b,5,H,W)
    bucket = torch.where(first[:, None, None], torch.full_like(prev_hard, 4), boundary_buckets(prev_hard))
    feat = bucket * 5 + base.argmax(1)
    corr = base + table.reshape(-1, 5)[feat].permute(0, 3, 1, 2)
    corr = HI.ste_round(corr * R.HPAC_LOGIT_PRECISION) / R.HPAC_LOGIT_PRECISION
    logp = corr.log_softmax(1) / math.log(2.0)
    return -(cur_q * logp).sum(dim=(1, 2, 3))


def slave(ci, sl):
    car = torch.einsum("bk,kchw->bchw", HI.ste_round(ci) * step, current_basis()) / math.sqrt(R.CARRIER_DIM)
    x = (gray + amp * car).clamp(0, 255)
    x = x + (x.round() - x).detach()
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


class SoftRenderer(torch.nn.Module):
    # #135's renderer forward on a soft one-hot map (the embedding lookup becomes a mix of class vectors)
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, oh, idx):
        r = self.r
        v = torch.einsum("bchw,cd->bdhw", oh, r.token_embed.weight)
        v = r.coord_mix(torch.cat([v, r.coordinates(v.shape[0], oh.device, v.dtype)], 1))
        fr = r.frame_embed(idx)
        for blk in r.blocks:
            v = blk(v, fr)
        return torch.sigmoid(r.head(F.gelu(v))) * 255.0


soft_rend = SoftRenderer(rend)


def render(oh, idx):
    params = {"r." + k: v for k, v in render_params().items()}
    x = torch.func.functional_call(soft_rend, params, (oh, idx))
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def apply_selector(f0, idxs):
    # the 14-byte selector is integer pixel ops on a few frames; applied without gradient
    out = f0.clone()
    for j, p in enumerate(idxs.tolist()):
        if True:
            a = f0[j:j + 1].detach().permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            b = torch.from_numpy(apply_pixel_mode(a, modes[sel[p]])).to(dev).permute(0, 3, 1, 2).float()
            out[j:j + 1] = f0[j:j + 1] + (b - f0[j:j + 1]).detach()
    return out


@torch.no_grad()
def exact_state():
    # hard map, current hpac/table/carrier: exact seg, pose, rate over all 600 frames
    hard = torch.where(band, logits.argmax(1), maps0)
    oh = F.one_hot(hard, 5).permute(0, 3, 1, 2).float()
    bits = 0.0
    for s in range(0, N, 8):
        e = min(s + 8, N)
        idx = torch.arange(s, e, device=dev)
        prev = torch.cat([torch.zeros_like(hard[:1]), hard[:-1]])[s:e]
        prev_oh = F.one_hot(prev, 5).permute(0, 3, 1, 2).float()
        bits += frame_bits(idx, oh[s:e], oh[s:e], prev_oh, prev, idx == 0).sum().item()
    seg = pose = 0.0
    for s in range(0, N, 4):
        e = min(s + 4, N)
        idx = torch.arange(s, e, device=dev)
        f1 = render(oh[s:e], idx)
        f0 = apply_selector(slave(codes[s:e], None), idx)
        po, so = net(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
        seg += (so.argmax(1) != seg_t[s:e]).float().mean(dim=(1, 2)).sum().item()
        pose += (po["pose"][:, :6] - pose_t[s:e]).pow(2).mean(1).sum().item()
    seg, pose = seg / N, pose / N
    other = 186_724 - len(parts.token_stream)          # archive bytes outside the token stream (#135)
    total = other + bits / 8 + side_bytes_delta()
    score = 100 * seg + math.sqrt(10 * pose) + 25 * total / ORIG
    return {"seg": seg, "pose": pose, "token_bytes": bits / 8, "archive_bytes": total, "score": score,
            "flips_vs_start": int((hard != maps0).sum())}


if args.check_rate:
    st = exact_state()
    print(json.dumps(st, indent=1))
    print("compare token_bytes with exact_cost135.py --tokens on the same map (114,705.46 for #135's map, "
          "118,852.6 for #141's)")
    sys.exit(0)

hpac_params = [p for p in hpac.parameters() if p.requires_grad]
opt = torch.optim.Adam([{"params": [logits], "lr": args.lr_map},
                        {"params": hpac_params, "lr": args.lr_hpac},
                        {"params": [table], "lr": args.lr_table},
                        {"params": [codes], "lr": args.lr_coef},
                        {"params": [gray, amp], "lr": args.lr_const},
                        {"params": list(r_master.values()), "lr": max(args.lr_render, 1e-12)},
                        {"params": [b_master], "lr": max(args.lr_basis, 1e-12)}])
W_SEG = 100.0 / N
W_RATE = 25.0 / ORIG / 8.0
P0 = None


def snapshot():
    return {"logits": logits.detach().clone(), "hpac": copy.deepcopy(hpac.state_dict()), "table": table.detach().clone(),
            "codes": codes.detach().clone(), "gray": gray.detach().clone(), "amp": amp.detach().clone(),
            "render": {k: v.detach().clone() for k, v in r_master.items()}, "basis": b_master.detach().clone()}


def restore(s):
    with torch.no_grad():
        logits.copy_(s["logits"]); table.copy_(s["table"]); codes.copy_(s["codes"]); gray.copy_(s["gray"]); amp.copy_(s["amp"])
        for k, v in s["render"].items():
            r_master[k].copy_(v)
        b_master.copy_(s["basis"])
    hpac.load_state_dict(s["hpac"])


best = exact_state()
P0 = best["pose"]
good = snapshot()
print(f"gate 0: {json.dumps(best)}", flush=True)
t0 = time.time()
for it in range(1, args.steps + 1):
    f = int(torch.randint(0, N - args.window, (1,)))
    sl = slice(f, f + args.window + 1)                  # +1: the next frame's rate depends on this window's maps
    idx = torch.arange(f, f + args.window + 1, device=dev)
    before = logits[sl].argmax(1).detach().clone()
    oh, q = onehot_st(sl)
    prev_sl = slice(f - 1, f + args.window) if f > 0 else None
    prev_oh = torch.cat([onehot_st(slice(f - 1, f))[0] if f > 0 else torch.zeros_like(oh[:1]), oh[:-1]])
    prev_hard = prev_oh.argmax(1).detach()
    bits = frame_bits(idx, oh, q, prev_oh, prev_hard, idx == 0).sum()
    pw = slice(f, f + args.window)
    f1 = render(oh[:args.window], idx[:args.window])
    f0 = apply_selector(slave(codes[pw], None), idx[:args.window])
    x = torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2)
    pin, sin_ = net.preprocess_input(x)
    sl_ = net.segnet(sin_)
    tgt = seg_t[pw]
    margin = sl_.gather(1, tgt[:, None])[:, 0] - sl_.scatter(1, tgt[:, None], -1e9).amax(1)
    seg_soft = torch.sigmoid(-margin / args.tau).mean(dim=(1, 2)).sum()
    pose_mse = (net.posenet(pin)["pose"][:, :6] - pose_t[pw]).pow(2).mean(1).sum()
    w_pose = 10.0 / (2.0 * math.sqrt(10.0 * max(P0, 1e-9))) / N
    loss = W_SEG * seg_soft + w_pose * pose_mse + W_RATE * bits
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_check:
        groups = {"map logits": [logits], "hpac": hpac_params, "table": [table], "carrier strengths": [codes],
                  "gray/amp": [gray, amp], "renderer": list(r_master.values()), "patterns": [b_master]}
        for gname, ps in groups.items():
            norms = [p.grad.norm().item() if p.grad is not None else None for p in ps]
            nz = sum(1 for n in norms if n not in (None, 0.0))
            print(f"grad {gname:18s}: {nz}/{len(ps)} tensors with nonzero grad, "
                  f"total norm {sum(n for n in norms if n) :.3e}", flush=True)
        print(f"loss terms: seg {W_SEG*seg_soft.item():.3e}  pose {w_pose*pose_mse.item():.3e}  rate {W_RATE*bits.item():.3e}")
        sys.exit(0)
    opt.step()
    with torch.no_grad():
        # flip cap: per frame keep only the flip-cap flips with the largest logit margin, revert the rest
        after = logits[sl].argmax(1)
        changed = (after != before) & band[sl]
        for j in range(changed.shape[0]):
            nflip = int(changed[j].sum())
            if nflip > args.flip_cap:
                lj = logits[sl][j]
                gap = lj.gather(0, after[j][None])[0] - lj.gather(0, before[j][None])[0]
                gap = torch.where(changed[j], gap, torch.full_like(gap, -1e9))
                keep = torch.topk(gap.reshape(-1), args.flip_cap).indices
                revert = changed[j].reshape(-1).clone(); revert[keep] = False
                revert = revert.reshape(H, W)
                lv = logits.data[f + j]
                lv[:, revert] = F.one_hot(before[j][revert], 5).T.float() * 3.0
    if it % 20 == 0:
        print(f"step {it}: loss {loss.item():.6f}  seg_soft {seg_soft.item():.2f}  pose {pose_mse.item()/args.window:.3e}  "
              f"bits {bits.item():,.0f}  gray {gray.item():.2f} amp {amp.item():.2f}  {time.time()-t0:.0f}s", flush=True)
    if it % args.gate == 0 or it == args.steps:
        st = exact_state()
        verdict = "ACCEPT" if st["score"] < best["score"] else "ROLLBACK"
        print(f"gate {it}: {verdict} {json.dumps(st)}  (best {best['score']:.6f})", flush=True)
        if verdict == "ACCEPT":
            best, good = st, snapshot()
            hard = torch.where(band, logits.argmax(1), maps0).to(torch.uint8).cpu().numpy()
            hard.tofile(out_dir / "tokens_joint.u8")
            np.save(out_dir / "coef_joint.npy", (HI.ste_round(codes) * step).detach().cpu().numpy())
            torch.save({"hpac": hpac.state_dict(), "table": table.detach().cpu(), "gray": gray.item(), "amp": amp.item(),
                        "render_codes": {k: v.detach().cpu() for k, v in r_master.items()},
                        "basis_codes": b_master.detach().cpu()},
                       out_dir / "joint_state.pt")
            json.dump(best, open(out_dir / "best.json", "w"), indent=1)
        else:
            restore(good)
print("final best", json.dumps(best))
