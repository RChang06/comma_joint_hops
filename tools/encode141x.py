# encode a class map into #141's exact token-stream format and splice it into #141's archive.
# encode141x: copy of encode141.py with --hpac-ihs1 / --table-body / --table-state to code with a different
# hpac (IHS1 blob) and/or rcf1 table; the corrector path is unchanged. with overrides only the stream is valid
# (the archive splice keeps #141's hpac/table), so use --stream-out and build the archive separately.
#
# the probability path is #141's own runtime/residual_archive.py::decode_production_tokens, driven with known
# symbols: same hpac/sparse evaluator, same rcf1 boundary table, same _probability_table, and the same adaptive
# corrector object (native c when built, else the python FreeCorrector) fed group_state -> coding_row -> observe.
# the coding rows go to codexblack's rc64 encoder (byte-compatible: #141's rc64_backend.c is #135's, crlf aside).
#
# the token stream is the tail of zip member p (no length field, blk2 tail = 96 B rcf1 body + stream), so:
#   new_p = old_p[:len(old_p) - len(old_stream)] + new_stream, re-zipped with compress.py's zip settings.
# gate: encoding #141's own decoded map must reproduce its stream (and archive) byte for byte.
# --lockstep also runs #141's rc64 decoder on the stored stream with the same rows and reports the first
# position where the decoded symbol differs from the target (only meaningful for #141's own map).
import argparse, sys, os, io, zipfile, hashlib, time, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "submissions/semantic_blocks"
EB = HERE / "research/repos/CommaVideoCompressionChallenge_ExperimentBook/src"
BUILD = HERE / "build141"
SRC_SHA = "0e2d95c29e87dca3f9ed14bbc842e57011090ce1e0ac0703c0bb8df3fc70c7e1"

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "extracted/tokens_pr141.u8"), help="uint8 [600,384,512] map")
ap.add_argument("--out", default=None, help="output archive.zip (full runs only)")
ap.add_argument("--frames", type=int, default=600, help="encode only the first n frames (gate subset)")
ap.add_argument("--lockstep", action="store_true", help="decode the stored stream alongside and compare")
ap.add_argument("--python-corrector", action="store_true", help="force the python FreeCorrector")
ap.add_argument("--stream-out", default=None, help="also write the raw token stream here")
ap.add_argument("--hpac-ihs1", default=None, help="IHS1 blob to use instead of #141's hpac")
ap.add_argument("--table-body", default=None, help="96 B rcf1 body (no magic) instead of #141's table")
ap.add_argument("--table-state", default=None, help="joint_state.pt with table_codes/table_scale")
args = ap.parse_args()


def build_libs():
    # same flags as inflate.sh; -ffp-contract=off is required for the corrector
    BUILD.mkdir(exist_ok=True)
    jobs = [
        (BUILD / "rc64_backend.so", ["gcc", "-O3", "-std=c11", "-shared", "-fPIC",
                                     str(SUB / "runtime/entropy/rc64_backend.c")]),
        (BUILD / "f26_corrector_native.so", ["gcc", "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off",
                                             "-fno-fast-math", str(SUB / "runtime/f26_corrector_native.c"), "-lm"]),
    ]
    for out, cmd in jobs:
        if not out.exists():
            subprocess.run(cmd + ["-o", str(out)], check=True)
    enc = BUILD / "rc64_encoder.so"
    if not enc.exists():
        compile_backend(enc)
    return BUILD / "rc64_backend.so", BUILD / "f26_corrector_native.so", enc


sys.path.insert(0, str(SUB))
sys.path.insert(0, str(EB))
from cpr1_sub4.entropy.rc64 import NativeEncoder, compile_backend
dec_lib, corr_lib, enc_lib = build_libs()
if not args.python_corrector:
    os.environ["F26_CORRECTOR_NATIVE_LIBRARY"] = str(corr_lib)
os.environ["CPR1_RC64_LIBRARY"] = str(dec_lib)

import torch
from runtime import residual_archive as RA
from runtime.f26_inflate import _load_renderer
from runtime.ihs2 import materialize_ihs1
from runtime.hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
from runtime.entropy.rc64 import NativeDecoder

src_zip = SUB / "archive.zip"
src_bytes = src_zip.read_bytes()
assert hashlib.sha256(src_bytes).hexdigest() == SRC_SHA, "unexpected #141 archive"
parts = RA.read_residual_archive(src_zip)
hpac_blob, table = parts.hpac_blob, parts.table
if args.hpac_ihs1:
    hpac_blob = Path(args.hpac_ihs1).read_bytes()
    assert hpac_blob.startswith(b"IHS1")
if args.table_state:
    assert not args.table_body
    st = torch.load(args.table_state, map_location="cpu", weights_only=False)
    codes = np.asarray(torch.as_tensor(st["table_codes"]).double().numpy()).reshape(-1)
    assert np.array_equal(codes, np.round(codes)) and codes.min() >= -32 and codes.max() <= 31
    sc = np.asarray([float(st["table_scale"])], dtype="<f2")
    assert float(sc[0]) == float(st["table_scale"])
    acc = 0
    for i, v in enumerate(codes.astype(np.int64)):
        acc |= (int(v) & 0x3F) << (6 * i)
    body = sc.tobytes() + acc.to_bytes(94, "little")
    table = RA._decode_fixed_table(RA.FIXED_MAGIC + body)
    assert np.array_equal(table.codes.reshape(-1), codes)
    if args.stream_out:
        Path(args.stream_out + ".table").write_bytes(body)
elif args.table_body:
    table = RA._decode_fixed_table(RA.FIXED_MAGIC + Path(args.table_body).read_bytes())
override = hpac_blob != parts.hpac_blob or not np.array_equal(table.values, parts.table.values)
assert not (override and args.out), "--out splices #141's hpac/table; use --stream-out with overrides"
renderer_dir = SUB / "cpr1"
R = _load_renderer(renderer_dir)
dev = torch.device("cuda")
configure_cuda_reproducibility()

# identical setup to decode_production_tokens
model = R.load_hpac(materialize_ihs1(hpac_blob, R), dev)
masks = R.group_masks(dev)
sparse = RA._sparse_class(renderer_dir)(model, R.EVAL_H, R.EVAL_W)
corrector = RA._rr8_select_corrector(R.EVAL_H * R.EVAL_W)
plans = []
for m in masks:
    fpos = np.flatnonzero(m.detach().cpu().numpy().reshape(-1))
    plans.append((torch.from_numpy(fpos).to(dev), fpos))

nf = args.frames
target = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)
assert target.max() < RA.NUM_CLASSES
enc = NativeEncoder(enc_lib)
dec = NativeDecoder(dec_lib, parts.token_stream) if args.lockstep else None
mismatch = None
cdf_digest = hashlib.sha256()
print(f"corrector {type(corrector).__name__}  frames {nf}  lockstep {bool(dec)}  tokens {args.tokens}", flush=True)
print(f"hpac {args.hpac_ihs1 or '#141'} ({len(hpac_blob)} B)  table {args.table_state or args.table_body or '#141'} "
      f"scale {table.scale!r}  override {override}", flush=True)

t0 = time.time()
with torch.inference_mode():
    optimize_sparse_evaluator(sparse)
    previous = torch.zeros((1, R.EVAL_H, R.EVAL_W), dtype=torch.long, device=dev)
    for f in range(nf):
        index = torch.tensor([f], dtype=torch.long, device=dev)
        current = torch.zeros_like(previous)
        context = model.prepare_frame_context(index, previous)
        if f:
            boundary = RA._boundary_buckets(previous[0].to(device="cpu", dtype=torch.uint8).numpy()).reshape(-1)
        else:
            boundary = np.full(R.EVAL_H * R.EVAL_W, 4, dtype=np.uint8)
        corrector.begin_frame(boundary)
        flat = target[f].reshape(-1)
        for g, (dpos, fpos) in enumerate(plans):
            base = sparse.selected_logits(current, context, g).cpu().numpy()
            predicted = base.argmax(axis=1).astype(np.int64)
            feature = boundary[fpos].astype(np.int64) * RA.NUM_CLASSES + predicted
            corrected = base + table.values[feature]
            probability = RA._probability_table(corrected, R.HPAC_LOGIT_PRECISION)
            cdf_digest.update(np.ascontiguousarray(probability, dtype="<f4").tobytes())
            state = corrector.group_state(probability, predicted, fpos)
            row = corrector.coding_row(state)
            sym = flat[fpos].astype(np.int64)
            enc.encode(sym, row)
            if dec is not None and mismatch is None:
                got = dec.decode(row).astype(np.int64)
                bad = np.flatnonzero(got != sym)
                if bad.size:
                    i = int(bad[0]); p = int(fpos[i])
                    mismatch = (f, g, i, p // R.EVAL_W, p % R.EVAL_W, int(got[i]), int(sym[i]))
                    print(f"LOCKSTEP MISMATCH frame {f} group {g} idx {i} (y {p // R.EVAL_W}, x {p % R.EVAL_W}) "
                          f"decoded {got[i]} target {sym[i]}", flush=True)
            corrector.observe(state, sym)
            current.reshape(-1)[dpos] = torch.from_numpy(sym).to(dev)
        tokens_f = current[0].to(device="cpu", dtype=torch.uint8).numpy().reshape(-1)
        assert np.array_equal(tokens_f, flat)
        corrector.end_frame(tokens_f)
        previous = current
        if (f + 1) % 10 == 0 or f + 1 == nf:
            el = time.time() - t0
            print(f"encoded {f+1}/{nf}  {el:.0f}s  ({el/(f+1):.2f}s/frame)", flush=True)
stream = enc.finish()
elapsed = time.time() - t0
old = parts.token_stream

common = 0
lim = min(len(stream), len(old))
diff = np.flatnonzero(np.frombuffer(stream[:lim], np.uint8) != np.frombuffer(old[:lim], np.uint8))
common = int(diff[0]) if diff.size else lim
print(f"encode time {elapsed:.0f}s  stream {len(stream):,} B (stored {len(old):,} B)  "
      f"common prefix {common:,} B  lockstep {'ok' if mismatch is None else mismatch}", flush=True)
if dec is not None:
    print(f"decoder bit position {dec.bit_position:,} after {nf} frames", flush=True)
if nf == R.N:
    print(f"cdf input sha {cdf_digest.hexdigest()} (local decode receipt: 370a5e2a85ccbb1e...)")
if args.stream_out:
    Path(args.stream_out).write_bytes(stream)

if nf == R.N:
    identical = stream == old
    print(f"GATE stream identical: {identical}" + ("" if identical else f"  first differing byte {common}"))
    with zipfile.ZipFile(src_zip) as z:
        old_p = z.read("p")
    assert old_p.endswith(old), "token stream is not the tail of p"
    new_p = old_p[: len(old_p) - len(old)] + stream
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, new_p)
    out_bytes = buf.getvalue()
    sha = hashlib.sha256(out_bytes).hexdigest()
    print(f"archive {len(out_bytes):,} B sha {sha}  identical to #141 archive: {out_bytes == src_bytes}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(out_bytes)
        print("wrote", args.out)
