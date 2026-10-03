# step 3 feasibility test: can a zero-byte basis generated in code drive posenet as well as
# pr141's stored 12x3x24x32 learned basis?
#
# frame 1 of every pair (what segnet grades) is taken unchanged from pr141's decoded output, so
# seg cannot move. only frame 0 (the pose carrier) is rebuilt and its 600x12 coefficients are
# refit against the dali posenet targets.
#
# runs:
#   A  pr141's decoded frame 0 as-is (includes the frame-0 selector)       -> should be 6.37e-6
#   B  frame 0 re-rendered from pr141's basis + coefficients, no selector   -> selector's share
#   C  control: refit pr141's own basis from its own coefficients           -> can the fit hold?
#   D  dct basis, coefficients initialised by projection, then refit       -> the actual question
import sys, math, time, argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/root/work/challenge")
ap.add_argument("--targets", default="/root/work/gt_dali_t2000.pt")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--chunk", type=int, default=40)
ap.add_argument("--atoms", type=int, default=12)
ap.add_argument("--runs", default="ABCD")
ap.add_argument("--basis", default="random", choices=["dct", "random"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ROOT = Path(args.root)
SUB = ROOT / "submissions/semantic_blocks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
from modules import DistortionNet
import modules, frame_utils
# the official rgb_to_yuv6 is wrapped in no_grad; same math, gradients allowed
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__
from runtime.residual_archive import read_residual_archive
from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
import inflate as R

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda", 0)
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in net.parameters():
    p.requires_grad_(False)
gt_pose = torch.load(args.targets)["pose"][:, :6].to(dev).clone()

# pr141's carrier, decoded exactly the way f26_inflate does it
parts = read_residual_archive(SUB / "archive.zip")
carrier, _ = split_frame0_selector_carrier(parts.carrier_blob)
canonical = materialize_cpr1(carrier, R)
marker = bytes(40_252)
import struct
_, basis_raw, coef0 = R.unpack_semantic_pose(struct.pack("<II", len(marker), len(canonical)) + marker + canonical)
basis_raw, coef0 = basis_raw.float().to(dev).clone(), coef0.float().to(dev).clone()
print("pr141 basis", tuple(basis_raw.shape), "coefficients", tuple(coef0.shape))

raw = np.memmap(SUB / "inflated/0.raw", dtype=np.uint8, mode="r").reshape(R.N, 2, R.CAMERA_H, R.CAMERA_W, 3)


def ste_round(x):
    return x + (x.round() - x).detach()


def render_frame0(coef, basis_n, hard):
    # same op order as cpr1/inflate.py render_video, with straight-through rounding when training
    rnd = torch.round if hard else ste_round
    carrier = torch.einsum("bk,kchw->bchw", coef, basis_n) / math.sqrt(basis_n.shape[0])
    x = rnd((127.5 + R.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0))
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bicubic", align_corners=False)
    return rnd(x.clamp(0.0, 255.0))


def pose_err(frame0_bchw, sl):
    f1 = torch.from_numpy(np.ascontiguousarray(raw[sl, 1])).to(dev).permute(0, 3, 1, 2).float()
    x = torch.stack([frame0_bchw, f1], dim=1)
    out = net.posenet(net.posenet.preprocess_input(x))["pose"][:, :6]
    return (out - gt_pose[sl]).pow(2).mean(dim=1)


def evaluate(coef, basis_n):
    total = 0.0
    with torch.no_grad():
        for s in range(0, R.N, args.chunk):
            sl = slice(s, min(s + args.chunk, R.N))
            total += pose_err(render_frame0(coef[sl], basis_n, True), sl).sum().item()
    return total / R.N


@torch.enable_grad()
def fit(coef_init, basis_n, label):
    basis_n = basis_n.clone()
    coef = coef_init.clone()
    unit = coef.std(dim=0).clamp_min(1e-6)
    started = time.time()
    for s in range(0, R.N, args.chunk):
        sl = slice(s, min(s + args.chunk, R.N))
        c = (coef[sl] / unit).clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        best, best_c = float("inf"), c.detach().clone()
        for _ in range(args.steps):
            loss = pose_err(render_frame0(c * unit, basis_n, False), sl).sum()
            if loss.item() < best:
                best, best_c = loss.item(), (c.detach() * unit).clone()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        coef[sl] = best_c
    print(f"  {label}: fit {time.time() - started:.0f}s")
    return coef


def quantize12(coef):
    # int12 codes with a per-dimension scale, as the cpr1 carrier stores them
    scale = coef.abs().amax(dim=0).clamp_min(1e-8) / 2047
    return (coef / scale).round().clamp(-2048, 2047) * scale


def term(p):
    return math.sqrt(10 * p)


def dct_basis(k):
    # lowest spatial frequencies (zigzag, dc excluded) on a gray channel pattern, plus the two
    # chroma-contrast dc patterns, all at the carrier's native 24x32 grid
    h, w = R.CARRIER_H, R.CARRIER_W
    ys, xs = torch.arange(h).float(), torch.arange(w).float()
    freqs = sorted(((u, v) for u in range(8) for v in range(8) if (u, v) != (0, 0)), key=lambda t: (t[0] + t[1], t[0]))
    atoms = []
    for chroma in (torch.tensor([1.0, -1.0, 0.0]), torch.tensor([0.5, 0.5, -1.0])):
        atoms.append(chroma[:, None, None].expand(3, h, w).clone())
    for u, v in freqs:
        if len(atoms) >= k:
            break
        pat = torch.cos(math.pi * (ys[:, None] + 0.5) * u / h) * torch.cos(math.pi * (xs[None, :] + 0.5) * v / w)
        atoms.append(pat[None].expand(3, h, w).clone())
    return torch.stack(atoms[:k]).to(dev)


base = None
if "A" in args.runs:
    tot = 0.0
    with torch.no_grad():
        for s in range(0, R.N, args.chunk):
            sl = slice(s, min(s + args.chunk, R.N))
            f0 = torch.from_numpy(np.ascontiguousarray(raw[sl, 0])).to(dev).permute(0, 3, 1, 2).float()
            tot += pose_err(f0, sl).sum().item()
    base = tot / R.N
    print(f"A pr141 decoded frame0: pose {base:.4e} term {term(base):.5f}")

basis141 = R.normalized_basis(basis_raw)
if "B" in args.runs:
    p = evaluate(coef0, basis141)
    print(f"B pr141 basis re-rendered, no selector: pose {p:.4e} term {term(p):.5f}")

if "C" in args.runs:
    c = fit(coef0, basis141, "C")
    p, pq = evaluate(c, basis141), evaluate(quantize12(c), basis141)
    print(f"C control refit of pr141 basis: pose {p:.4e} (int12 {pq:.4e}) term {term(pq):.5f}")

if "D" in args.runs:
    if args.basis == "dct":
        bd = R.normalized_basis(dct_basis(args.atoms))
    else:
        # zero-byte basis: gaussian noise at the carrier grid, reproducible from one seed
        g = torch.Generator().manual_seed(args.seed)
        bd = R.normalized_basis(torch.randn(args.atoms, 3, R.CARRIER_H, R.CARRIER_W, generator=g).to(dev))
    # projection init: least squares of pr141's carrier images onto the dct atoms
    with torch.no_grad():
        target = torch.einsum("bk,kchw->bchw", coef0, basis141) / math.sqrt(12)
        A = bd.reshape(bd.shape[0], -1).T / math.sqrt(bd.shape[0])
        init = torch.linalg.lstsq(A, target.reshape(R.N, -1).T).solution.T
        captured = 1 - (A @ init.T - target.reshape(R.N, -1).T).pow(2).sum() / target.pow(2).sum()
    print(f"D {args.basis} atoms {args.atoms}: pr141 carrier energy captured by projection {captured.item():.3f}")
    p0 = evaluate(init, bd)
    print(f"  projection only: pose {p0:.4e}")
    c = fit(init, bd, "D")
    p, pq = evaluate(c, bd), evaluate(quantize12(c), bd)
    print(f"D {args.basis} refit: pose {p:.4e} (int12 {pq:.4e}) term {term(pq):.5f}")
    if base is not None:
        saved = 12_277 * 25 / 37_545_489
        delta = term(pq) - term(base) - saved
        print(f"  net score change vs pr141 (basis bytes only, coefficient bytes assumed equal): {delta:+.5f}")
