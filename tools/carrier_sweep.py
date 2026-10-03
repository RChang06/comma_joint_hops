# quick room check on pr141's pose carrier: round the basis or the coefficients coarser,
# refit the coefficients against the dali posenet targets, and see how far pose moves.
# prints each config as soon as it finishes; stops early when everything is clearly too costly.
import sys, math
sys.argv = [sys.argv[0], "--runs", ""] + sys.argv[1:]
exec(open("/root/work/fit_carrier.py").read().split("base = None")[0])

BASE_POSE = 6.3695e-06
STEPS = 100
basis141 = R.normalized_basis(basis_raw)


def requant_basis(bits):
    # per-atom symmetric rounding of the stored basis to `bits` bits
    levels = 2 ** (bits - 1) - 1
    scale = basis_raw.abs().amax(dim=(1, 2, 3), keepdim=True) / levels
    return (basis_raw / scale).round().clamp(-levels - 1, levels) * scale


def q_coef(c, bits, scale):
    levels = 2 ** (bits - 1) - 1
    q = (c / scale).clamp(-levels - 1, levels)
    return (q + (q.round() - q).detach()) * scale


@torch.enable_grad()
def refit(basis_n, coef_bits=None, steps=STEPS):
    basis_n = basis_n.clone()
    coef = coef0.clone()
    unit = coef.std(dim=0).clamp_min(1e-6)
    scale = None
    if coef_bits:
        scale = coef.abs().amax(dim=0) / (2 ** (coef_bits - 1) - 1)
    for s in range(0, R.N, args.chunk):
        sl = slice(s, min(s + args.chunk, R.N))
        c = (coef[sl] / unit).clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=0.02)
        best, best_c = float("inf"), None
        for _ in range(steps):
            cc = c * unit
            if coef_bits:
                cc = q_coef(cc, coef_bits, scale)
            loss = pose_err(render_frame0(cc, basis_n, False), sl).sum()
            if loss.item() < best:
                best, best_c = loss.item(), cc.detach().clone()
            opt.zero_grad()
            loss.backward()
            opt.step()
        coef[sl] = best_c
    return coef


def entropy_bytes(codes):
    # order-0 entropy of previous-pair deltas per dimension, a rough stand-in for the rice coder
    total = 0.0
    d = torch.diff(codes, dim=0, prepend=torch.zeros_like(codes[:1]))
    for k in range(codes.shape[1]):
        _, counts = torch.unique(d[:, k], return_counts=True)
        p = counts.float() / counts.sum()
        total += -(p * p.log2()).sum().item() * codes.shape[0]
    return total / 8


results = []
def report(label, pose, bytes_saved):
    t = math.sqrt(10 * pose) - math.sqrt(10 * BASE_POSE)
    r = -bytes_saved * 25 / 37_545_489
    ratio = pose / BASE_POSE
    results.append(ratio)
    print(f"{label:34s} pose {pose:.4e} ({ratio:5.2f}x)  pose term {t:+.5f}  bytes saved ~{bytes_saved:6.0f}  "
          f"rate {r:+.5f}  NET {t + r:+.5f}", flush=True)


print("baseline pose", BASE_POSE)
# basis at fewer bits: the stored basis is 98,213 bits (3.55 bits/value); estimate new size as
# bits * values * (3.55/5), i.e. keep the same entropy-coding gain the huffman stage gets over raw 5-bit
for bits in (3, 2):
    b = R.normalized_basis(requant_basis(bits))
    c = refit(b)
    new_bytes = 12_277 * bits / 5
    report(f"basis {bits}-bit, coef refit", evaluate(c, b), 12_277 - new_bytes)

# coefficients at fewer bits, rounding inside the fit; size from delta entropy vs the stored 9,829 bytes
for bits in (8, 6):
    c = refit(basis141, coef_bits=bits)
    scale = coef0.abs().amax(dim=0) / (2 ** (bits - 1) - 1)
    codes = (c / scale).round()
    report(f"coef {bits}-bit, refit", evaluate(c, basis141), 9_829 - entropy_bytes(codes))

if all(r > 3 for r in results):
    print("FLAG: every config pushes pose past 3x baseline - carrier looks tight, stop here")
