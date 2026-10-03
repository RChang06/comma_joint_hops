# joint training of #135's stored state against the score, with an exact differentiable rate term.
#
# trainable together, one loss (100*seg + sqrt(10*pose) + 25*bytes/ORIG):
#   class maps         5-way logits on boundary-band pixels only, straight-through hard map in the forward pass
#   hpac               ste-rounded weights/biases, each row clamped to its starting range so the packed ihs2
#                      row depths cannot grow; frame embedding clamped to its int4 field [-8,7]; exponents frozen
#   boundary table     int6 codes * the stored fp16 scale, ste-rounded (its real storage grid)
#   carrier strengths  600x12 int12 codes, clamped to [-2048, 2047]
#   gray, amplitude    the carrier's 127.5 / 64 constants live in decoder code, so moving them costs no bytes
#   renderer           wans1 grid: int4 codes in [-7,7] * fixed fp16 row scales, fp16 tensors rounded to fp16
#   carrier patterns   5-bit codes in [-16,15] * fixed per-pattern scale
# per-frame parameters (map logits, carrier strengths, hpac and renderer frame embeddings) use a sliced adam:
# only rows of frames in the current window move, each with its own moments and step count, so momentum can't
# drift frames outside the window. rollback restores every optimizer state together with the weights.
# the rate is hpac's probability of each pixel with the map as context (forward on the hard map = exact cost,
# gradients through the soft one), covering each edit's effect on neighbours and on the next frame.
# safety: per-frame flip cap per step, and an exact gate every --gate steps (hard state, full 600-frame rate,
# seg and pose on all pairs) that rolls back to the last good state when the exact score gets worse.
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
from runtime.entropy.renderer_weight_codec import decode_wans1
from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
import modules, frame_utils
from modules import DistortionNet
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--coefficients", default="", help="starting carrier strengths .npy (default #135's)")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--check-rate", action="store_true", help="only print the exact state (gate 0)")
ap.add_argument("--grad-check", action="store_true", help="one step; report the gradient reaching every group")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--window", type=int, default=4, help="consecutive pairs whose distortion is trained per step")
ap.add_argument("--band", type=int, default=2)
ap.add_argument("--flip-cap", type=int, default=8, help="max map flips per frame per step")
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--lr-map", type=float, default=0.05)
ap.add_argument("--lr-hpac", type=float, default=0.02, help="int-code units")
ap.add_argument("--lr-table", type=float, default=0.05, help="int6-code units")
ap.add_argument("--lr-coef", type=float, default=0.5, help="int12-code units")
ap.add_argument("--lr-const", type=float, default=0.01)
ap.add_argument("--lr-render", type=float, default=0.02, help="int4-code units (fp16 tensors: value units)")
ap.add_argument("--lr-basis", type=float, default=0.02, help="5-bit-code units")
ap.add_argument("--gate", type=int, default=200)
ap.add_argument("--err-focus", type=int, default=0,
                help="if >0, only map pixels within this many px of a CURRENT segnet error may move (gradient masked)")
ap.add_argument("--w-pose", type=float, default=1.0, help="multiplier on the pose term in the loss and the gate")
ap.add_argument("--w-rate", type=float, default=1.0, help="multiplier on the rate term in the loss and the gate")
ap.add_argument("--frame-accept", action="store_true",
                help="at each gate keep map changes only in frames whose exact seg wrong-count did not get worse "
                     "(maps-only runs); other frames revert to the last good map")
ap.add_argument("--map-mode", default="st", choices=["st", "soft"],
                help="st: straight-through hard map; soft: renderer and hpac see the probability mix softmax(z/T), "
                     "T annealed from --map-t0 to --map-t1 (the gate always scores the hard argmax map)")
ap.add_argument("--map-t0", type=float, default=1.0)
ap.add_argument("--map-t1", type=float, default=0.1)
ap.add_argument("--hpac-batch", type=int, default=4,
                help="random frames per step for the hpac/table rate gradient on the CURRENT maps (#130 batches random "
                     "frames; consecutive dashcam frames are near-duplicates and drag a shared model off course)")
ap.add_argument("--shared-every", type=int, default=75,
                help="accumulate gradients of the SHARED params (hpac, table, renderer, patterns, gray/amp) over this "
                     "many windows (75 x window 8 = one sweep of all 600 frames) before one adam step; a shared model "
                     "stepped on 8 frames at a time fits those frames and gets worse on the rest")
ap.add_argument("--lr-render-fp16", type=float, default=1e-3,
                help="fp16 renderer tensors move this fraction of their own max |value| per step (maxnorm)")
ap.add_argument("--carrier-gn", type=int, default=8, help="gauss-newton iterations on the carrier strengths at each gate")
ap.add_argument("--carrier-polish", type=int, default=1, help="exact +-1/+-2 integer polish passes after the gn")
ap.add_argument("--opt", default="maxnorm", choices=["maxnorm", "adam"],
                help="maxnorm: each step the entry with the largest |grad| moves lr, the rest proportionally "
                     "(adam moves every entry ~lr regardless of gradient, which mass-flips maps and codes)")
ap.add_argument("--out", default=str(HERE / "joint_run"))
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
ORIG = 37_545_489
N, H, W = R.N, R.EVAL_H, R.EVAL_W
HW = H * W
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ fixed pieces
comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96); rend.load_state_dict(comp["renderer"]); rend = rend.to(dev).eval()
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in list(rend.parameters()) + list(net.parameters()):
    p.requires_grad_(False)
tg = torch.load(args.targets)
seg_t, pose_t = tg["seg"].to(dev).long(), tg["pose"][:, :6].to(dev).float()
modes, sel = decode_selector(comp["selector"])
parts = RA.read_residual_archive(SUB / "archive.zip")
OTHER_BYTES = 186_724 - len(parts.token_stream)       # archive bytes outside the token stream (#135)

# ------------------------------------------------------------------ trainable pieces
hpac = R.load_hpac(RA.materialize_ihs1(parts.hpac_blob, R), dev).train()
hpac_shared, hpac_bound = [], {}
for name, prm in hpac.named_parameters():
    if name.endswith("exponent"):
        prm.requires_grad_(False)
        continue
    prm.requires_grad_(True)
    b = prm.detach().abs().reshape(prm.shape[0], -1).amax(1).clamp_min(1.0)
    if name == "frame_embed.weight":
        b = torch.full_like(b, 7.0)                    # stored as int4 in [-8,7]; keep it symmetric inside
    hpac_bound[name] = b
    if name != "frame_embed.weight":
        hpac_shared.append(prm)
hpac_frame = dict(hpac.named_parameters())["frame_embed.weight"]          # (600, 8): per-frame

t_scale = float(parts.table.scale)
t_master = torch.tensor(np.asarray(parts.table.codes, dtype=np.float32).reshape(25, 5), device=dev).requires_grad_(True)


def table_values():
    return HI.ste_round(t_master).clamp(-32, 31) * t_scale


_canon = materialize_cpr1(split_frame0_selector_carrier(parts.carrier_blob)[0], R)
_bs, _bc, _cs, _ = R.decode_compact_carrier(_canon, basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W,
                                            frames=N, dimensions=R.CARRIER_DIM)
step = torch.from_numpy(np.asarray(_cs, dtype=np.float32)).to(dev)
c0 = torch.from_numpy(np.load(args.coefficients)).to(dev).float() if args.coefficients else comp["coefficients"].to(dev).float()
codes = (c0 / step).round().clamp(-2048, 2047).requires_grad_(True)       # (600, 12): per-frame
codes0 = codes.detach().clone()
b_scale = torch.from_numpy(np.asarray(_bs, dtype=np.float32)).to(dev)[:, None, None, None]
b_master = torch.from_numpy(np.asarray(_bc, dtype=np.float32).reshape(R.CARRIER_DIM, 3, R.CARRIER_H, R.CARRIER_W)).to(dev)
b_master.requires_grad_(args.lr_basis > 0)
b_codes0 = b_master.detach().clone()
gray = torch.tensor(127.5, device=dev, requires_grad=True)
amp = torch.tensor(float(R.CARRIER_AMPLITUDE), device=dev, requires_grad=True)


def current_basis():
    return R.normalized_basis(HI.ste_round(b_master).clamp(-16, 15) * b_scale)


r_master, r_scale = {}, {}
for rec in decode_wans1(parts.semantic_blob):
    name = rec.schema.name
    if rec.codes is not None:
        shp = [1] * len(rec.schema.shape)
        shp[-1 if name.endswith("embed.weight") else 0] = rec.schema.scale_count
        r_scale[name] = torch.from_numpy(np.asarray(rec.scales, dtype=np.float32)).reshape(shp).to(dev)
        r_master[name] = torch.from_numpy(rec.codes.astype(np.float32)).to(dev)
    else:
        r_master[name] = torch.from_numpy(np.asarray(rec.values, dtype=np.float32)).to(dev)
    r_master[name].requires_grad_(args.lr_render > 0)
r_codes0 = {k: v.detach().clone() for k, v in r_master.items() if k in r_scale}
r_frame = r_master["frame_embed.weight"]                                  # (600, 8): per-frame
r_shared = [v for k, v in r_master.items() if k != "frame_embed.weight"]


def render_params():
    out = {}
    for k, v in r_master.items():
        out[k] = HI.ste_round(v).clamp(-7, 7) * r_scale[k] if k in r_scale else v + (v.half().float() - v).detach()
    return out


maps0 = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint8).reshape(N, H, W).copy()).to(dev).long()


def band_of(m):
    e = torch.zeros_like(m, dtype=torch.bool)
    e[:, 1:] |= m[:, 1:] != m[:, :-1]; e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:, :, 1:] |= m[:, :, 1:] != m[:, :, :-1]; e[:, :, :-1] |= m[:, :, :-1] != m[:, :, 1:]
    return F.max_pool2d(e.float()[:, None], 2 * args.band + 1, 1, args.band)[:, 0] > 0


band_idx = torch.nonzero(band_of(maps0).reshape(-1)).squeeze(1)
row_off = torch.searchsorted(band_idx, torch.arange(N + 1, device=dev) * HW)
z = (F.one_hot(maps0.reshape(-1)[band_idx], 5).float() * 3.0).requires_grad_(True)
print(f"band pixels {band_idx.numel():,} ({band_idx.numel() / (N * HW) * 100:.2f}%)", flush=True)


def hard_maps():
    m = maps0.clone().reshape(-1)
    m[band_idx] = z.argmax(1)
    return m.reshape(N, H, W)


MAP_T = [args.map_t0]


def onehot_frames(f0, f1, grad=True):
    oh = F.one_hot(maps0[f0:f1], 5).reshape(-1, 5).float()
    a, b = int(row_off[f0]), int(row_off[f1])
    loc = band_idx[a:b] - f0 * HW
    q = (z[a:b] / (MAP_T[0] if args.map_mode == "soft" else 1.0)).softmax(1)
    hard = F.one_hot(q.argmax(1), 5).float()
    if not grad:
        val = hard.detach()
    elif args.map_mode == "soft":
        val = q
    else:
        val = hard + q - q.detach()
    oh = oh.index_put((loc,), val)
    return oh.reshape(f1 - f0, H, W, 5).permute(0, 3, 1, 2)


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


def frame_bits(idx, cur_oh, prev_oh, round_logits=True):
    prev_hard = prev_oh.argmax(1).detach()
    first = idx == 0
    ctx = hpac_context(idx, prev_oh)
    base = hpac_logits(cur_oh, ctx)
    bucket = torch.where(first[:, None, None], torch.full_like(prev_hard, 4), boundary_buckets(prev_hard))
    feat = bucket * 5 + base.argmax(1)
    corr = base + table_values().reshape(-1, 5)[feat].permute(0, 3, 1, 2)
    if round_logits:
        corr = HI.ste_round(corr * R.HPAC_LOGIT_PRECISION) / R.HPAC_LOGIT_PRECISION
    return -(cur_oh * corr.log_softmax(1)).sum(dim=(1, 2, 3)) / math.log(2.0)


def zero_frame_oh(b=1):
    return F.one_hot(torch.zeros((b, H, W), dtype=torch.long, device=dev), 5).permute(0, 3, 1, 2).float()


class SoftRenderer(torch.nn.Module):
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
    x = torch.func.functional_call(soft_rend, {"r." + k: v for k, v in render_params().items()}, (oh, idx))
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def slave(ci):
    car = torch.einsum("bk,kchw->bchw", HI.ste_round(ci).clamp(-2048, 2047) * step, current_basis()) / math.sqrt(R.CARRIER_DIM)
    x = (gray + amp * car).clamp(0, 255)
    x = x + (x.round() - x).detach()
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def apply_selector(f0, idxs):
    out = f0.clone()
    for j, p in enumerate(idxs.tolist()):
        a = f0[j:j + 1].detach().permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        b = torch.from_numpy(apply_pixel_mode(a, modes[sel[p]])).to(dev).permute(0, 3, 1, 2).float()
        out[j:j + 1] = f0[j:j + 1] + (b - f0[j:j + 1]).detach()
    return out


def entropy_bytes(c):
    _, cnt = torch.unique(c.round(), return_counts=True)
    pr = cnt.float() / cnt.sum()
    return float(-(cnt.float() * pr.log2()).sum() / 8)


def side_bytes_delta():
    # order-0 entropy changes vs the start: renderer int4 codes, pattern codes, carrier-strength first differences.
    # hpac rows are clamped to their starting range, so its packed size does not grow.
    d = 0.0
    with torch.no_grad():
        for k in r_codes0:
            d += entropy_bytes(HI.ste_round(r_master[k]).clamp(-7, 7)) - entropy_bytes(r_codes0[k])
        d += entropy_bytes(HI.ste_round(b_master).clamp(-16, 15)) - entropy_bytes(b_codes0)
        cq = HI.ste_round(codes).clamp(-2048, 2047)
        d += entropy_bytes(torch.diff(cq, dim=0)) - entropy_bytes(torch.diff(codes0, dim=0))
    return d


RATE_ONLY = None
_CACHED_DIST = {}
FRAME_SEG = torch.zeros(N, device=dev)


@torch.no_grad()
def exact_state():
    global RATE_ONLY
    if RATE_ONLY is None:
        RATE_ONLY = (args.lr_map == 0 and args.lr_coef == 0 and args.lr_render == 0 and args.lr_basis == 0
                     and args.lr_const == 0 and args.carrier_gn == 0 and args.carrier_polish == 0)
    hard = hard_maps()
    oh = F.one_hot(hard, 5).permute(0, 3, 1, 2).float()
    bits = 0.0
    for s in range(0, N, 8):
        e = min(s + 8, N)
        idx = torch.arange(s, e, device=dev)
        prev_oh = torch.cat([zero_frame_oh(), oh[:-1]])[s:e]
        bits += frame_bits(idx, oh[s:e], prev_oh).sum().item()
    if RATE_ONLY and _CACHED_DIST:
        seg, pose = _CACHED_DIST["seg"], _CACHED_DIST["pose"]
        total = OTHER_BYTES + bits / 8 + side_bytes_delta()
        score = 100 * seg + math.sqrt(10 * pose) + 25 * total / ORIG
        return {"seg": seg, "pose": pose, "token_bytes": bits / 8, "archive_bytes_est": total, "score": score,
                "flips_vs_start": int((hard != maps0).sum()), "gray": gray.item(), "amp": amp.item()}
    seg = pose = 0.0
    for s in range(0, N, 4):
        e = min(s + 4, N)
        idx = torch.arange(s, e, device=dev)
        f1 = render(oh[s:e], idx)
        f0 = apply_selector(slave(codes[s:e]), idx)
        po, so = net(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
        fw = (so.argmax(1) != seg_t[s:e]).float().mean(dim=(1, 2))
        FRAME_SEG[s:e] = fw
        seg += fw.sum().item()
        pose += (po["pose"][:, :6] - pose_t[s:e]).pow(2).mean(1).sum().item()
    seg, pose = seg / N, pose / N
    _CACHED_DIST.update(seg=seg, pose=pose)
    total = OTHER_BYTES + bits / 8 + side_bytes_delta()
    score = 100 * seg + args.w_pose * math.sqrt(10 * pose) + args.w_rate * 25 * total / ORIG
    return {"seg": seg, "pose": pose, "token_bytes": bits / 8, "archive_bytes_est": total, "score": score,
            "flips_vs_start": int((hard != maps0).sum()), "gray": gray.item(), "amp": amp.item()}


def carrier_block(iters, polish):
    # the carrier strengths' update rule inside the cycle: damped gauss-newton (6x12 jacobian per pair, exact int12
    # acceptance) on the CURRENT maps/renderer/patterns/gray/amp, then an exact +-1/+-2 integer polish
    if iters <= 0 and polish <= 0:
        return
    hard = hard_maps()
    masters = torch.empty((N, 3, R.CAMERA_H, R.CAMERA_W), dtype=torch.uint8, device=dev)
    with torch.no_grad():
        for s0 in range(0, N, 8):
            e0 = min(s0 + 8, N)
            masters[s0:e0] = render(F.one_hot(hard[s0:e0], 5).permute(0, 3, 1, 2).float(),
                                    torch.arange(s0, e0, device=dev)).to(torch.uint8)
    bas = current_basis().detach()
    g0, a0 = float(gray), float(amp)

    def outs(cq, s0, e0, hard_round):
        car = torch.einsum("bk,kchw->bchw", cq * step, bas) / math.sqrt(R.CARRIER_DIM)
        x = (g0 + a0 * car).clamp(0, 255)
        x = x.round() if hard_round else x
        x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False).clamp(0, 255)
        x = x.round() if hard_round else x
        if hard_round:
            x = apply_selector(x, torch.arange(s0, e0, device=dev))
        return net.posenet(net.posenet.preprocess_input(torch.stack([x, masters[s0:e0].float()], 1)))["pose"][:, :6]

    def exact_mse(cq):
        o = torch.empty(N, device=dev)
        with torch.no_grad():
            for s0 in range(0, N, 24):
                e0 = min(s0 + 24, N)
                q = cq[s0:e0].round().clamp(-2048, 2047)
                o[s0:e0] = (outs(q, s0, e0, True) - pose_t[s0:e0]).pow(2).mean(1)
        return o

    cq = codes.detach().round().clamp(-2048, 2047).clone()
    cur = exact_mse(cq)
    lam = torch.full((N,), 1e-3, device=dev)
    for _ in range(iters):
        delta = torch.zeros_like(cq)
        for s0 in range(0, N, 24):
            e0 = min(s0 + 24, N)
            c = cq[s0:e0].clone().requires_grad_(True)
            o = outs(c, s0, e0, False)
            r = (o - pose_t[s0:e0]).detach()
            J = torch.zeros(e0 - s0, 6, R.CARRIER_DIM, device=dev)
            for k in range(6):
                J[:, k], = torch.autograd.grad(o[:, k].sum(), c, retain_graph=k < 5)
            JtJ = J.transpose(1, 2) @ J
            A = JtJ + lam[s0:e0, None, None] * torch.diag_embed(JtJ.diagonal(dim1=1, dim2=2).clamp_min(1e-12))
            delta[s0:e0] = torch.linalg.solve(A, -(J.transpose(1, 2) @ r[:, :, None])[:, :, 0])
        trial = exact_mse(cq + delta)
        better = trial < cur
        cq = torch.where(better[:, None], (cq + delta).round().clamp(-2048, 2047), cq)
        cur = torch.where(better, trial, cur)
        lam = torch.where(better, (lam / 3).clamp_min(1e-6), (lam * 4).clamp_max(1e6))
    for _ in range(polish):
        for k in range(R.CARRIER_DIM):
            for m in (-2, -1, 1, 2):
                t = cq.clone(); t[:, k] = (t[:, k] + m).clamp(-2048, 2047)
                tr = exact_mse(t)
                better = tr < cur
                cq = torch.where(better[:, None], t, cq); cur = torch.where(better, tr, cur)
    with torch.no_grad():
        codes.copy_(cq)


if args.check_rate:
    print(json.dumps(exact_state(), indent=1))
    sys.exit(0)


class SlicedAdam:
    """adam over a per-frame parameter: only rows of frames in the current window are updated."""

    def __init__(self, param, lr, rows_of, b1=0.9, b2=0.999, eps=1e-8, mode=None):
        self.p, self.lr, self.rows_of, self.b1, self.b2, self.eps = param, lr, rows_of, b1, b2, eps
        self.mode = mode or args.opt
        self.m = torch.zeros_like(param)
        self.v = torch.zeros_like(param)
        self.t = torch.zeros(N, device=param.device)

    @torch.no_grad()
    def step(self, frames):
        if self.p.grad is None:
            return
        if self.mode == "maxnorm":
            for f in frames:
                r = self.rows_of(f)
                g = self.p.grad[r]
                self.p[r] -= self.lr * g / (g.abs().max() + 1e-30)
            return
        for f in frames:
            r = self.rows_of(f)
            self.t[f] += 1
            g = self.p.grad[r]
            self.m[r] = self.b1 * self.m[r] + (1 - self.b1) * g
            self.v[r] = self.b2 * self.v[r] + (1 - self.b2) * g * g
            mh = self.m[r] / (1 - self.b1 ** self.t[f])
            vh = self.v[r] / (1 - self.b2 ** self.t[f])
            self.p[r] -= self.lr * mh / (vh.sqrt() + self.eps)

    def state(self):
        return (self.m.clone(), self.v.clone(), self.t.clone())

    def load(self, s):
        self.m.copy_(s[0]); self.v.copy_(s[1]); self.t.copy_(s[2])


frame_rows = lambda f: slice(f, f + 1)
map_opt = SlicedAdam(z, args.lr_map, lambda f: slice(int(row_off[f]), int(row_off[f + 1])))
coef_opt = SlicedAdam(codes, args.lr_coef, frame_rows)
hframe_opt = SlicedAdam(hpac_frame, args.lr_hpac, frame_rows, mode="adam")
rframe_opt = SlicedAdam(r_frame, max(args.lr_render, 1e-12), frame_rows, mode="adam")
_hw = {n[:-len("exponent")] + "weight" for n, _ in hpac.named_parameters() if n.endswith("exponent")
       and not n.startswith(("frame_scale", "spm_"))}
_hn = dict(hpac.named_parameters())
shared_opt = torch.optim.Adam([{"params": [_hn[n] for n in _hn if id(_hn[n]) in {id(q) for q in hpac_shared} and n not in _hw], "lr": args.lr_hpac},
                               {"params": [_hn[n] for n in _hw], "lr": args.lr_hpac * 8},
                               {"params": [t_master], "lr": args.lr_table},
                               {"params": [gray, amp], "lr": args.lr_const},
                               {"params": [v for k, v in r_master.items() if k in r_scale and k != "frame_embed.weight"],
                                "lr": max(args.lr_render, 1e-12)},
                               {"params": [v for k, v in r_master.items() if k not in r_scale],
                                "lr": max(args.lr_render_fp16 * args.lr_render / max(args.lr_render, 1e-12), 1e-12) * 0.1},
                               {"params": [b_master], "lr": max(args.lr_basis, 1e-12)}])
sliced = [map_opt, coef_opt, hframe_opt, rframe_opt]
all_params = [z, codes, hpac_frame, r_frame, t_master, gray, amp, b_master] + hpac_shared + r_shared


@torch.no_grad()
def project():
    for name, prm in hpac.named_parameters():
        if name in hpac_bound:
            bnd = hpac_bound[name].reshape([-1] + [1] * (prm.dim() - 1))
            prm.copy_(torch.maximum(torch.minimum(prm, bnd), -bnd))
    codes.clamp_(-2048, 2047)
    t_master.clamp_(-32, 31)
    b_master.clamp_(-16, 15)
    for k in r_scale:
        r_master[k].clamp_(-7, 7)


def snapshot():
    return {"params": [p.detach().clone() for p in all_params], "shared": copy.deepcopy(shared_opt.state_dict()),
            "sliced": [o.state() for o in sliced]}


def restore(snap):
    with torch.no_grad():
        for p, v in zip(all_params, snap["params"]):
            p.copy_(v)
    shared_opt.load_state_dict(snap["shared"])
    for o, s in zip(sliced, snap["sliced"]):
        o.load(s)


def save_state(st):
    hard_maps().to(torch.uint8).cpu().numpy().tofile(out_dir / "tokens_joint.u8")
    np.save(out_dir / "coef_codes_joint.npy", HI.ste_round(codes).clamp(-2048, 2047).detach().cpu().numpy().astype(np.int32))
    np.save(out_dir / "coef_joint.npy", (HI.ste_round(codes).clamp(-2048, 2047) * step).detach().cpu().numpy())
    torch.save({"hpac": hpac.state_dict(), "table_codes": HI.ste_round(t_master).clamp(-32, 31).detach().cpu(),
                "table_scale": t_scale, "gray": gray.item(), "amp": amp.item(),
                "render_masters": {k: v.detach().cpu() for k, v in r_master.items()},
                "basis_codes": HI.ste_round(b_master).clamp(-16, 15).detach().cpu()}, out_dir / "joint_state.pt")
    json.dump(st, open(out_dir / "best.json", "w"), indent=1)


W_SEG = 100.0 / N
W_RATE = 25.0 / ORIG / 8.0
if args.steps == 0:
    # evaluation mode: carrier update for the given maps, then the exact state; also dump the starting wrong-pixel masks
    carrier_block(args.carrier_gn, args.carrier_polish)
    print("eval:", json.dumps(exact_state()), flush=True)
    sys.exit(0)
best = exact_state()
GOOD_FRAME_SEG = FRAME_SEG.clone()
P0 = best["pose"]
good = snapshot()
print(f"gate 0: {json.dumps(best)}", flush=True)
t0 = time.time()
base_lrs = [g["lr"] for g in shared_opt.param_groups]
base_sliced = [o.lr for o in sliced]
for it in range(1, args.steps + 1):
    decay = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (it - 1) / max(args.steps - 1, 1)))
    for g, b0 in zip(shared_opt.param_groups, base_lrs):
        g["lr"] = b0 * decay
    for o, b0 in zip(sliced, base_sliced):
        o.lr = b0 * decay
    MAP_T[0] = args.map_t0 * (args.map_t1 / args.map_t0) ** ((it - 1) / max(args.steps - 1, 1))
    f = int(torch.randint(0, N - args.window + 1, (1,)))
    e = f + args.window
    win = list(range(f, e))
    idx = torch.arange(f, e, device=dev)
    before = z[int(row_off[f]):int(row_off[e])].argmax(1).detach().clone()
    oh = onehot_frames(f, e, grad=True)
    prev_oh = torch.cat([onehot_frames(f - 1, f, grad=False) if f > 0 else zero_frame_oh(), oh[:-1]])
    bits = frame_bits(idx, oh, prev_oh).sum()
    if e < N:
        bits = bits + frame_bits(torch.tensor([e], device=dev), onehot_frames(e, e + 1, grad=False), oh[-1:]).sum()
    f1 = render(oh, idx)
    f0 = apply_selector(slave(codes[f:e]), idx)
    pin, sin_ = net.preprocess_input(torch.stack([f0, f1], 1).permute(0, 1, 3, 4, 2))
    sl_ = net.segnet(sin_)
    tgt = seg_t[f:e]
    margin = sl_.gather(1, tgt[:, None])[:, 0] - sl_.scatter(1, tgt[:, None], -1e9).amax(1)
    seg_soft = torch.sigmoid(-margin / args.tau).mean(dim=(1, 2)).sum()
    pose_mse = (net.posenet(pin)["pose"][:, :6] - pose_t[f:e]).pow(2).mean(1).sum()
    w_pose = 10.0 / (2.0 * math.sqrt(10.0 * max(P0, 1e-9))) / N
    loss = W_SEG * seg_soft + args.w_pose * w_pose * pose_mse + args.w_rate * W_RATE * bits
    for p in (z, codes, hpac_frame, r_frame):
        p.grad = None
    loss.backward()
    if args.err_focus > 0 and z.grad is not None:
        with torch.no_grad():
            wrong = (margin <= 0).float()[:, None]                                    # (window,1,H,W)
            near = F.max_pool2d(wrong, 2 * args.err_focus + 1, 1, args.err_focus)[:, 0] > 0
            for j, fr in enumerate(win):
                a0, b0 = int(row_off[fr]), int(row_off[fr + 1])
                loc = band_idx[a0:b0] - fr * HW
                z.grad[a0:b0] *= near[j].reshape(-1)[loc][:, None].float()
    if args.lr_hpac > 0 or args.lr_table > 0:
        # the hpac/table gradient from random frames of the current hard maps (mean nats per pixel, like #130);
        # this replaces the window's contribution to the shared hpac params
        with torch.no_grad():
            hm = hard_maps()
        for prm in hpac_shared + [t_master]:
            prm.grad = None
        hpac_frame.grad = None
        ridx = torch.randperm(N, device=dev)[:args.hpac_batch]
        cur = F.one_hot(hm[ridx], 5).permute(0, 3, 1, 2).float()
        prv = torch.where((ridx == 0)[:, None, None], torch.zeros_like(hm[ridx]), hm[(ridx - 1).clamp_min(0)])
        prv = F.one_hot(prv, 5).permute(0, 3, 1, 2).float()
        hb = frame_bits(ridx, cur, prv, round_logits=False).sum() * math.log(2.0) / (args.hpac_batch * HW)
        hb.backward()
        rand_frames = ridx.tolist()
    else:
        rand_frames = []
    if args.grad_check:
        groups = {"map logits": [z], "hpac shared": hpac_shared, "hpac frame": [hpac_frame], "table": [t_master],
                  "carrier strengths": [codes], "gray/amp": [gray, amp], "renderer shared": r_shared,
                  "renderer frame": [r_frame], "patterns": [b_master]}
        for gname, ps in groups.items():
            norms = [p.grad.norm().item() if p.grad is not None else 0.0 for p in ps]
            print(f"grad {gname:18s}: {sum(1 for n in norms if n > 0)}/{len(ps)} nonzero, norm {sum(norms):.3e}", flush=True)
        print(f"loss terms: seg {W_SEG*seg_soft.item():.3e}  pose {w_pose*pose_mse.item():.3e}  rate {W_RATE*bits.item():.3e}")
        sys.exit(0)
    if it % args.shared_every == 0:
        with torch.no_grad():
            for grp in shared_opt.param_groups:
                for prm in grp["params"]:
                    if prm.grad is not None:
                        prm.grad /= args.shared_every
        torch.nn.utils.clip_grad_norm_(hpac_shared, 10.0)
        shared_opt.step()
        shared_opt.zero_grad(set_to_none=True)
    for o in sliced:
        o.step(rand_frames if o is hframe_opt else win)
    project()
    with torch.no_grad():
        for fr in win:
            a, b = int(row_off[fr]), int(row_off[fr + 1])
            ba = before[a - int(row_off[f]):b - int(row_off[f])]
            zz = z[a:b]
            af = zz.argmax(1)
            ch = af != ba
            if int(ch.sum()) > args.flip_cap:
                gap = zz.gather(1, af[:, None])[:, 0] - zz.gather(1, ba[:, None])[:, 0]
                gap = torch.where(ch, gap, torch.full_like(gap, -1e9))
                keep = torch.topk(gap, args.flip_cap).indices
                rev = ch.clone(); rev[keep] = False
                zz[rev] = F.one_hot(ba[rev], 5).float() * 3.0
                map_opt.m[a:b][rev] = 0; map_opt.v[a:b][rev] = 0
    if it % 20 == 0:
        print(f"step {it}: loss {loss.item():.6f}  seg_soft {seg_soft.item():.2f}  pose {pose_mse.item()/args.window:.3e}  "
              f"bits {bits.item():,.0f}  gray {gray.item():.2f} amp {amp.item():.2f}  {time.time()-t0:.0f}s", flush=True)
    if it % args.gate == 0 or it == args.steps:
        carrier_block(args.carrier_gn, args.carrier_polish)
        st = exact_state()
        if args.frame_accept:
            worse = (FRAME_SEG > GOOD_FRAME_SEG + 1e-12).nonzero().squeeze(1).tolist()
            with torch.no_grad():
                for fr in worse:
                    a0, b0 = int(row_off[fr]), int(row_off[fr + 1])
                    z[a0:b0] = good["params"][0][a0:b0]
                    map_opt.m[a0:b0] = 0; map_opt.v[a0:b0] = 0
            kept = int(((FRAME_SEG < GOOD_FRAME_SEG - 1e-12)).sum())
            st = exact_state()
            print(f"  frame-accept: reverted {len(worse)} worse frames, {kept} frames improved", flush=True)
        # a tie means the float masters moved without crossing a rounding step: keep them (rolling back here erased
        # all sub-step progress whenever gates were frequent)
        verdict = "ACCEPT" if st["score"] < best["score"] else ("TIE" if st["score"] <= best["score"] + 1e-12 else "ROLLBACK")
        print(f"gate {it}: {verdict} {json.dumps(st)}  (best {best['score']:.6f})", flush=True)
        if verdict == "ACCEPT":
            best, good = st, snapshot()
            GOOD_FRAME_SEG = FRAME_SEG.clone()
            P0 = best["pose"]
            save_state(best)
        elif verdict == "ROLLBACK":
            restore(good)
print("final best", json.dumps(best))
