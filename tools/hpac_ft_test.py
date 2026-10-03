# a/b diagnostic for the joint trainer's hpac block: fine-tune #135's integer hpac on #135's maps with #130's OWN
# training loop (IntegerHPAC.forward, cross-entropy on hard tokens with the previous frame as context, adamw with
# separate lr groups, grad clip 10), and report the exact hpac-only bits after every epoch.
# if this lowers the bits steadily, joint_train135.py's hpac path has a bug; if it also blows up, fine-tuning a
# converged integer hpac needs much gentler steps.
import argparse, sys, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from runtime import residual_archive as RA

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "pr135_cost/tokens_pr135.u8"))
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-3, help="#130 used 2e-2 from scratch")
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
parts = RA.read_residual_archive(SUB / "archive.zip")
model = R.load_hpac(RA.materialize_ihs1(parts.hpac_blob, R), dev)
tokens = torch.from_numpy(np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W).copy()).to(dev).long()
previous = torch.cat([torch.zeros_like(tokens[:1]), tokens[:-1]])
named = dict(model.named_parameters())
expo = [n for n in named if n.endswith("exponent")]
weights = [n[:-len("exponent")] + "weight" for n in expo if not n.startswith(("frame_scale", "spm_"))]
for n in expo:
    named[n].requires_grad_(False)
groups = [{"params": [named[n] for n in named if n not in expo and n not in weights], "lr": args.lr},
          {"params": [named[n] for n in weights], "lr": args.lr * 8}]
opt = torch.optim.AdamW(groups, weight_decay=0.0)


@torch.no_grad()
def bits():
    model.eval()
    tot = 0.0
    for s in range(0, R.N, 8):
        e = min(s + 8, R.N)
        idx = torch.arange(s, e, device=dev)
        lg = model(tokens[s:e], idx, previous[s:e])
        q = torch.round(lg * R.HPAC_LOGIT_PRECISION) / R.HPAC_LOGIT_PRECISION     # the coder's 1/8 lattice
        tot += float(F.cross_entropy(q, tokens[s:e], reduction="sum")) / math.log(2)
    return tot / 8


print(f"epoch 0: hpac-only bytes {bits():,.1f}  (exact_cost135 raw-hpac reference: 114,851.8)", flush=True)
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(R.N, device=dev)
    t0 = time.time()
    for s in range(0, R.N, args.batch):
        idx = perm[s:s + args.batch]
        lg = model(tokens[idx], idx, previous[idx])
        loss = F.cross_entropy(lg, tokens[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    print(f"epoch {ep}: hpac-only bytes {bits():,.1f}  {time.time()-t0:.0f}s", flush=True)
