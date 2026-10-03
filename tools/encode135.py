# encode a class map into #135's exact token-stream format and splice it into #135's archive.
#
# the token stream is the tail of the zip member p (no length field), so a new map ships as:
#   new_p = old_p[:len(old_p) - len(old_stream)] + new_stream, re-zipped with #135's exact zip settings.
# the probability path mirrors runtime/residual_archive.py::decode_production_tokens exactly; the rc64 encoder
# is codexblack's (ExperimentBook src/cpr1_sub4/entropy), byte-compatible with #135's decoder.
# gate: encoding #135's own decoded map must reproduce its stream byte-for-byte.
import argparse, sys, io, zipfile, hashlib, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
EB = HERE / "research/repos/CommaVideoCompressionChallenge_ExperimentBook"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from runtime import residual_archive as RA
from runtime.hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
sys.path.insert(0, str(EB / "src"))
from cpr1_sub4.entropy.rc64 import NativeEncoder, compile_backend

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True, help="uint8 [600,384,512] map to encode")
ap.add_argument("--out", required=True, help="output archive.zip")
ap.add_argument("--lib", default="/tmp/rc64_full.so")
args = ap.parse_args()

if not Path(args.lib).exists():
    compile_backend(Path(args.lib))
dev = torch.device("cuda")
configure_cuda_reproducibility()
src_zip = SUB / "archive.zip"
parts = RA.read_residual_archive(src_zip)
model = R.load_hpac(RA.materialize_ihs1(parts.hpac_blob, R), dev)
masks = R.group_masks(dev)
sparse = RA._sparse_class(SUB / "cpr1")(model, R.EVAL_H, R.EVAL_W)
plans = [(torch.from_numpy(np.flatnonzero(m.cpu().numpy().reshape(-1))).to(dev),
          np.flatnonzero(m.cpu().numpy().reshape(-1))) for m in masks]
target = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)

enc = NativeEncoder(Path(args.lib))
t0 = time.time()
with torch.inference_mode():
    optimize_sparse_evaluator(sparse)
    previous = torch.zeros((1, R.EVAL_H, R.EVAL_W), dtype=torch.long, device=dev)
    for f in range(R.N):
        idx = torch.tensor([f], dtype=torch.long, device=dev)
        current = torch.zeros_like(previous)
        context = model.prepare_frame_context(idx, previous)
        if f:
            boundary = RA._boundary_buckets(previous[0].to("cpu", torch.uint8).numpy()).reshape(-1)
        else:
            boundary = np.full(R.EVAL_H * R.EVAL_W, 4, dtype=np.uint8)
        flat = target[f].reshape(-1)
        for g, (dpos, fpos) in enumerate(plans):
            base = sparse.selected_logits(current, context, g).cpu().numpy()
            predicted = base.argmax(axis=1).astype(np.int64)
            corrected = base + parts.table.values[boundary[fpos].astype(np.int64) * RA.NUM_CLASSES + predicted]
            prob = RA._probability_table(corrected, R.HPAC_LOGIT_PRECISION)
            sym = flat[fpos].astype(np.int64)
            enc.encode(sym, prob)
            current.reshape(-1)[dpos] = torch.from_numpy(sym).to(dev)
        previous = current
        if (f + 1) % 100 == 0:
            print(f"encoded {f+1}/600  {time.time()-t0:.0f}s", flush=True)
stream = enc.finish()

with zipfile.ZipFile(src_zip) as z:
    old_p = z.read("p")
assert old_p.endswith(parts.token_stream), "token stream is not the tail of p"
new_p = old_p[: len(old_p) - len(parts.token_stream)] + stream
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as z:
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    z.writestr(info, new_p)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_bytes(buf.getvalue())
same = stream == parts.token_stream
print(f"stream {len(stream):,} B (was {len(parts.token_stream):,})  identical={same}  "
      f"archive {len(buf.getvalue()):,} B  sha {hashlib.sha256(buf.getvalue()).hexdigest()[:16]}  -> {args.out}")
