# exact code length of a class map under PR #135's token coder.
#
# #135 codes each pixel with probabilities = softmax(quantised(HPAC logits + boundary table row)) and RC64.
# nothing adapts while decoding, so the real cost of ANY map is sum(-log2 p[true class]) under that model,
# up to RC64's tiny frequency rounding. this script:
#   decode mode (default): decodes #135's real token stream and sums the cost of every decoded pixel, so the
#                          total can be checked against the stream's actual size. saves the decoded maps.
#   --tokens FILE:         teacher-forces an arbitrary uint8 [600,384,512] map instead of decoding, and reports
#                          its exact cost. this is the ground truth any training-time rate estimate must match.
# both modes also report the raw-HPAC cost (no boundary table), the cheapest differentiable estimate.
import argparse, sys, time, os, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from runtime import residual_archive as RA
from runtime.hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default="", help="uint8 map to cost (default: decode #135's own stream)")
ap.add_argument("--frames", type=int, default=600)
ap.add_argument("--out", default=str(HERE / "pr135_cost"))
ap.add_argument("--dump-costs", default="", help="a:b frame range; saves -log2 p for all 5 classes per pixel "
                "(float16 [n,5,384,512]) given the forced map as context, i.e. the per-pixel price list for training")
args = ap.parse_args()

dev = torch.device("cuda")
configure_cuda_reproducibility()
parts = RA.read_residual_archive(SUB / "archive.zip")
model = R.load_hpac(RA.materialize_ihs1(parts.hpac_blob, R), dev)
masks = R.group_masks(dev)
sparse = RA._sparse_class(SUB / "cpr1")(model, R.EVAL_H, R.EVAL_W)
plans = [(torch.from_numpy(np.flatnonzero(m.cpu().numpy().reshape(-1))).to(dev),
          np.flatnonzero(m.cpu().numpy().reshape(-1))) for m in masks]

forced = None
if args.tokens:
    forced = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)
    decoder = None
else:
    decoder = RA.NativeDecoder(Path(os.environ["CPR1_RC64_LIBRARY"]), parts.token_stream)

LN2 = np.log(2.0)
bits_exact = np.zeros(args.frames)      # with the boundary table: what #135 actually pays
bits_raw = np.zeros(args.frames)        # hpac alone
tokens = np.empty((args.frames, R.EVAL_H, R.EVAL_W), dtype=np.uint8)
dump = None
if args.dump_costs:
    da, db = map(int, args.dump_costs.split(":"))
    dump = np.zeros((db - da, 5, R.EVAL_H * R.EVAL_W), dtype=np.float16)
t0 = time.time()
with torch.inference_mode():
    optimize_sparse_evaluator(sparse)
    previous = torch.zeros((1, R.EVAL_H, R.EVAL_W), dtype=torch.long, device=dev)
    for f in range(args.frames):
        idx = torch.tensor([f], dtype=torch.long, device=dev)
        current = torch.zeros_like(previous)
        context = model.prepare_frame_context(idx, previous)
        if f:
            boundary = RA._boundary_buckets(previous[0].to("cpu", torch.uint8).numpy()).reshape(-1)
        else:
            boundary = np.full(R.EVAL_H * R.EVAL_W, 4, dtype=np.uint8)
        for g, (dpos, fpos) in enumerate(plans):
            base = sparse.selected_logits(current, context, g).cpu().numpy()
            predicted = base.argmax(axis=1).astype(np.int64)
            corrected = base + parts.table.values[boundary[fpos].astype(np.int64) * RA.NUM_CLASSES + predicted]
            prob = RA._probability_table(corrected, R.HPAC_LOGIT_PRECISION)
            if decoder is not None:
                sym = decoder.decode(prob).astype(np.int64)
            else:
                sym = forced[f].reshape(-1)[fpos].astype(np.int64)
            if dump is not None and da <= f < db:
                dump[f - da][:, fpos] = (-np.log(np.maximum(prob, 1e-12)) / LN2).T.astype(np.float16)
            rows = np.arange(sym.size)
            bits_exact[f] -= np.log(np.maximum(prob[rows, sym], 1e-12)).sum() / LN2
            raw = RA._probability_table(base, R.HPAC_LOGIT_PRECISION)
            bits_raw[f] -= np.log(np.maximum(raw[rows, sym], 1e-12)).sum() / LN2
            current.reshape(-1)[dpos] = torch.from_numpy(sym).to(dev)
        tokens[f] = current[0].to("cpu", torch.uint8).numpy()
        previous = current
        if (f + 1) % 50 == 0:
            print(f"frame {f+1}/{args.frames}  exact {bits_exact[:f+1].sum()/8:,.0f} B  raw-hpac {bits_raw[:f+1].sum()/8:,.0f} B  {time.time()-t0:.0f}s", flush=True)

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
res = {"frames": args.frames, "exact_bytes": bits_exact.sum() / 8, "raw_hpac_bytes": bits_raw.sum() / 8,
       "stream_bytes": len(parts.token_stream), "mode": "forced" if forced is not None else "decode"}
if decoder is not None:
    res["decoder_bit_position_bytes"] = decoder.bit_position / 8
    tokens.tofile(out / "tokens_pr135.u8")
np.save(out / ("bits_exact_forced.npy" if forced is not None else "bits_exact.npy"), bits_exact)
np.save(out / ("bits_raw_forced.npy" if forced is not None else "bits_raw.npy"), bits_raw)
if dump is not None:
    np.save(out / f"costs_{da}_{db}.npy", dump.reshape(db - da, 5, R.EVAL_H, R.EVAL_W))
print(json.dumps(res, indent=1))
