# gradient-ranked, exactly-accepted token edits on #135's class maps (pre-distortion search).
#
# research (work/research/token_history.md): soft-token training collapses; what works is single-token moves at a
# wrong cell or one of its 8 neighbours, accepted only when the REAL render's wrong-cell count falls. #140's author
# halved seg debt this way. our addition is the exact #135 code length per class (price list from
# exact_cost135.py --dump-costs), so every accept is priced in real bits under the coder we ship.
#
# per pair, per round:
#   render -> segnet margins vs dali targets -> wrong cells W
#   candidates: (pixel in W dilated by 1, class != current), ranked by first-order seg gain (gradient wrt the
#               one-hot map) minus rate cost (price list), best K non-adjacent kept
#   exact test: render each candidate map in a batch, count wrong cells; accept the best one whose seg credit
#               exceeds its rate cost; repeat until no candidate helps or the round budget runs out
# pose is NOT priced here (a stale carrier makes it explode); the carrier is re-solved after the whole pass.
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
modules.rgb_to_yuv6 = frame_utils.rgb_to_yuv6.__wrapped__

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="0:20")
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--costs", required=True, help="costs_a_b.npy for the same frame range")
ap.add_argument("--targets", default=str(HERE / "targets/gt_dali_t2000.pt"))
ap.add_argument("--rounds", type=int, default=40, help="max accepted edits per pair")
ap.add_argument("--k", type=int, default=8, help="candidates exactly tested per round")
ap.add_argument("--out", default=str(HERE / "search_run"))
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
ORIG = 37_545_489
CELL = 100.0 / 600 / (R.EVAL_H * R.EVAL_W)      # score per wrong cell in one pair
BIT = 25.0 / ORIG / 8.0                          # score per bit
a, b = map(int, args.frames.split(":"))

comp = torch.load(HERE / "extracted/pr135_components.pt", weights_only=False)
rend = R.SemanticTokenRenderer(96); rend.load_state_dict(comp["renderer"]); rend = rend.to(dev).eval()
net = DistortionNet().eval().to(dev)
net.load_state_dicts(ROOT / "models/posenet.safetensors", ROOT / "models/segnet.safetensors", dev)
for p in list(rend.parameters()) + list(net.parameters()):
    p.requires_grad_(False)
seg_t = torch.load(args.targets)["seg"][a:b].to(dev).long()
costs = torch.from_numpy(np.load(args.costs).astype(np.float32)).to(dev)       # (n,5,H,W) bits
full = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy()


def render(onehot, pair):
    # onehot (B,5,H,W) -> camera-res frame 1 exactly as inflate (bilinear up, clamp, round)
    v = torch.einsum("bchw,cd->bdhw", onehot, rend.token_embed.weight)
    v = rend.coord_mix(torch.cat([v, rend.coordinates(v.shape[0], dev, v.dtype)], 1))
    fr = rend.frame_embed(torch.full((v.shape[0],), pair, device=dev))
    for blk in rend.blocks:
        v = blk(v, fr)
    x = torch.sigmoid(rend.head(F.gelu(v))) * 255.0
    x = F.interpolate(x, size=(R.CAMERA_H, R.CAMERA_W), mode="bilinear", align_corners=False).clamp(0, 255)
    return x + (x.round() - x).detach()


def seg_logits(frame1):
    # segnet reads frame[-1] only; its preprocess is a bilinear resize to 384x512
    return net.segnet(net.segnet.preprocess_input(frame1[:, None]))


log = []
t0 = time.time()
for i in range(b - a):
    pair = a + i
    cur = torch.from_numpy(full[pair]).to(dev).long()
    tgt = seg_t[i]
    accepted, start_wrong, bits_spent = 0, None, 0.0
    for r in range(args.rounds):
        oh = F.one_hot(cur, 5).permute(2, 0, 1)[None].float().requires_grad_(True)
        lg = seg_logits(render(oh, pair))
        true = lg.gather(1, tgt[None, None])[:, 0]
        other = lg.scatter(1, tgt[None, None], -1e9).amax(1)
        margin = true - other
        wrong = margin[0] <= 0
        nwrong = int(wrong.sum())
        if start_wrong is None:
            start_wrong = nwrong
        if nwrong == 0:
            break
        torch.sigmoid(-margin / 0.05).sum().backward()
        g = oh.grad[0]                                                       # (5,H,W) d(soft wrong)/d(onehot)
        near = F.max_pool2d(wrong.float()[None, None], 3, 1, 1)[0, 0] > 0
        gain = g - g.gather(0, cur[None])                                     # predicted change in soft wrong cells
        dbits = costs[i] - costs[i].gather(0, cur[None])                      # exact own-pixel rate change
        score = gain * CELL + dbits * BIT                                     # predicted score change (negative = good)
        score[:, ~near] = float("inf")
        score.scatter_(0, cur[None], float("inf"))
        flat = score.reshape(-1)
        top = torch.topk(-flat, k=min(args.k * 8, int(torch.isfinite(flat).sum())))
        cands, used = [], torch.zeros_like(near)
        for v, idx in zip(top.values.tolist(), top.indices.tolist()):
            if -v >= 0:
                break
            c, pix = divmod(idx, R.EVAL_H * R.EVAL_W)
            y, x = divmod(pix, R.EVAL_W)
            if used[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                continue                                                      # keep candidates non-adjacent
            used[y, x] = True
            cands.append((c, y, x))
            if len(cands) == args.k:
                break
        if not cands:
            break
        with torch.no_grad():
            batch = cur[None].repeat(len(cands), 1, 1)
            for j, (c, y, x) in enumerate(cands):
                batch[j, y, x] = c
            lgb = seg_logits(render(F.one_hot(batch, 5).permute(0, 3, 1, 2).float(), pair))
            nw = (lgb.argmax(1) != tgt[None]).sum(dim=(1, 2))
        best, best_val = None, 0.0
        for j, (c, y, x) in enumerate(cands):
            d_bits = float(costs[i, c, y, x] - costs[i, cur[y, x], y, x])
            val = (int(nw[j]) - nwrong) * CELL + d_bits * BIT             # exact seg change + priced rate change
            if val < best_val:
                best, best_val = (c, y, x, d_bits), val
        if best is None:
            break
        c, y, x, d_bits = best
        cur[y, x] = c
        accepted += 1
        bits_spent += d_bits
    full[pair] = cur.to(torch.uint8).cpu().numpy()
    log.append({"pair": pair, "start_wrong": start_wrong, "end_wrong": nwrong, "accepted": accepted,
                "bits": bits_spent})
    print(f"pair {pair}: wrong {start_wrong} -> {nwrong}  edits {accepted}  priced bits {bits_spent:+.1f}  "
          f"{time.time()-t0:.0f}s", flush=True)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
full.tofile(out / "tokens_searched.u8")
json.dump(log, open(out / "search_log.json", "w"), indent=1)
sw = sum(l["start_wrong"] for l in log); ew = sum(l["end_wrong"] for l in log)
print(f"total wrong cells {sw} -> {ew}  ({(ew-sw)*CELL:+.6f} score)  edits {sum(l['accepted'] for l in log)}  "
      f"priced bits {sum(l['bits'] for l in log):+.0f} ({sum(l['bits'] for l in log)*BIT:+.6f} score)")
