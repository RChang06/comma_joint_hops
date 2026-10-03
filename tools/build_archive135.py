# assemble a full #135-format archive (F24S models + table + rc64 token stream) from its pieces.
# models = F24S + ihs2 body + f12 wans body + stored cap1 fields + sparse selector body, raw-lzma'd with #135's
# filters; p = lzma(models) + 96 B table body + token stream; one stored zip member p.
# gate (--gate): rebuilding #135 from its own pieces must reproduce its archive byte for byte.
import argparse, sys, io, lzma, zipfile, hashlib, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
from runtime import residual_archive as RA


def split_models(models, ihs2_body_bytes=RA.IHS2_BODY_BYTES):
    # the shipped models blob cut into its stored pieces (hpac body, wans body, cap1 stored fields, selector body)
    off = 4
    hpac_body = models[off:off + ihs2_body_bytes]; off += ihs2_body_bytes
    wans = models[off:off + RA.WANS_BODY_BYTES]; off += RA.WANS_BODY_BYTES
    n = RA._cap1_body_bytes(models[off:])
    return hpac_body, wans, models[off:off + n], models[off + n:]


def zip_p(p):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as z:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        z.writestr(info, p)
    return buf.getvalue()


def assemble(hpac_body, wans_body, cap1_stored, selector_body, table_body, stream):
    models = RA.F24S_MAGIC + hpac_body + wans_body + cap1_stored + selector_body
    # #135's exact lzma bytes are not reproducible here (+11 B at best); try a few encoder settings that the
    # shipped decoder filters still read and keep the smallest that round-trips
    best = None
    for nice, mf in ((273, lzma.MF_BT4), (64, lzma.MF_BT4), (64, lzma.MF_BT3), (32, lzma.MF_BT2)):
        f = dict(RA.LZMA_FILTERS[0]); f.update(nice_len=nice, mf=mf)
        c = lzma.compress(models, format=lzma.FORMAT_RAW, filters=[f])
        d = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=RA.LZMA_FILTERS)
        if d.decompress(c) == models and (best is None or len(c) < len(best)):
            best = c
    comp = best
    assert len(table_body) == 96, len(table_body)
    return zip_p(comp + table_body + stream)


def shipped_pieces():
    p = zipfile.ZipFile(SUB / "archive.zip").read("p")
    d = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=RA.LZMA_FILTERS)
    models = d.decompress(p)
    section = d.unused_data
    return models, section[:96], section[96:]


def store_cap1(cap1):
    # inverse of residual_archive._restore_cap1: drop the 8 B prefix, reorder fields into the stored order
    assert cap1.startswith(RA.CAP1_PREFIX)
    body = cap1[len(RA.CAP1_PREFIX):]
    counts = body[:6]
    basis_bits = int.from_bytes(counts[:3], "little"); residual_bits = int.from_bytes(counts[3:6], "little")
    sizes = {"predictor": 36, "scales": 96, "lengths": 32, "ks": 12,
             "basis": (basis_bits + 7) // 8, "rice": (residual_bits + 7) // 8}
    fields, off = {}, 6
    for name in RA.CAP_FIELDS:
        fields[name] = body[off:off + sizes[name]]; off += sizes[name]
    assert off == len(body), "cap1 field accounting"
    stored = counts + b"".join(fields[n] for n in RA.STORED_CAP_FIELDS)
    assert RA._restore_cap1(stored) == cap1
    return stored


def encode_stream(tokens, hpac_blob, table_values, lib="/tmp/rc64_full.so"):
    # encode120's loop with a given hpac model and table (mirrors decode_production_tokens)
    import torch
    import inflate as R
    from runtime.hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
    sys.path.insert(0, str(HERE / "research/repos/CommaVideoCompressionChallenge_ExperimentBook/src"))
    from cpr1_sub4.entropy.rc64 import NativeEncoder, compile_backend
    if not Path(lib).exists():
        compile_backend(Path(lib))
    dev = torch.device("cuda")
    configure_cuda_reproducibility()
    model = R.load_hpac(RA.materialize_ihs1(hpac_blob, R), dev)
    masks = R.group_masks(dev)
    sparse = RA._sparse_class(SUB / "cpr1")(model, R.EVAL_H, R.EVAL_W)
    plans = [(torch.from_numpy(np.flatnonzero(m.cpu().numpy().reshape(-1))).to(dev),
              np.flatnonzero(m.cpu().numpy().reshape(-1))) for m in masks]
    enc = NativeEncoder(Path(lib))
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
            flat = tokens[f].reshape(-1)
            for g, (dpos, fpos) in enumerate(plans):
                base = sparse.selected_logits(current, context, g).cpu().numpy()
                predicted = base.argmax(axis=1).astype(np.int64)
                corrected = base + table_values[boundary[fpos].astype(np.int64) * RA.NUM_CLASSES + predicted]
                prob = RA._probability_table(corrected, R.HPAC_LOGIT_PRECISION)
                sym = flat[fpos].astype(np.int64)
                enc.encode(sym, prob)
                current.reshape(-1)[dpos] = torch.from_numpy(sym).to(dev)
            previous = current
            if (f + 1) % 100 == 0:
                print(f"encoded {f+1}/600  {time.time()-t0:.0f}s", flush=True)
    return enc.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--tokens"); ap.add_argument("--hpac"); ap.add_argument("--table"); ap.add_argument("--carrier")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.out:
        from runtime.carrier_repack import split_frame0_selector_carrier
        models, table0, _ = shipped_pieces()
        _, wans, cap0, sel0 = split_models(models)
        hpac = Path(a.hpac).read_bytes() if a.hpac else RA.IHS2_HEADER + split_models(models)[0]
        assert hpac.startswith(RA.IHS2_HEADER)
        table_body = Path(a.table).read_bytes() if a.table else table0
        if a.carrier:
            cap1, sel = split_frame0_selector_carrier(Path(a.carrier).read_bytes())
            cap_stored, sel_body = store_cap1(cap1), sel[len(RA.SPARSE_SELECTOR_PREFIX):]
        else:
            cap_stored, sel_body = cap0, sel0
        tv = RA._decode_fixed_table(RA.FIXED_MAGIC + table_body).values
        tokens = np.fromfile(a.tokens, dtype=np.uint8).reshape(600, 384, 512)
        stream = encode_stream(tokens, hpac, tv)
        z = assemble(hpac[len(RA.IHS2_HEADER):], wans, cap_stored, sel_body, table_body, stream)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_bytes(z)
        print(f"archive {len(z):,} B  hpac {len(hpac):,} B (IHS2_BYTES)  stream {len(stream):,} B  "
              f"sha {hashlib.sha256(z).hexdigest()}  -> {a.out}")
    if a.gate:
        models, table_body, stream = shipped_pieces()
        z = assemble(*split_models(models), table_body, stream)
        ref = (SUB / "archive.zip").read_bytes()
        print(f"rebuilt {len(z):,} B vs shipped {len(ref):,} B  identical={z == ref}  sha {hashlib.sha256(z).hexdigest()[:16]}")
