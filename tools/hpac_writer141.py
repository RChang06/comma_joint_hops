# hpac state_dict -> #141's stored hpac section: brotli(IHS1), as RX1 holds it (header field hpac_bytes).
# IHS1 = b"IHS1" + 517 depth nibbles + signed rows at those depths (lsb-first) + raw tail in named_parameters
# order (bias <i2, the rest int8), the exact inverse of cpr1/integer_model_io._deserialize_self_compressed.
# gate: #141's own hpac re-encodes to its IHS1 bytes, and its section re-compresses byte-identically.
# run: --state <joint_state.pt> writes the IHS1 blob and the brotli section for our hpac.
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import brotli

HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from hpac_integer import IntegerConv2d, IntegerLinear
from runtime import residual_archive as RA
from runtime import ihs2 as I2

COMPRESSED = (IntegerConv2d, IntegerLinear)
MAGIC = b"IHS1"
# found by search in gate(): reproduces #141's stored section byte for byte
BROTLI = dict(quality=10, lgwin=24, mode=0)


def hpac_shell():
    # same construction as cpr1/inflate.load_hpac
    return R.IntegerHPAC(num_pairs=R.N, num_classes=R.NUM_CLASSES, patch=R.HPAC_PATCH, delta=R.HPAC_DELTA,
                         channels=R.HPAC_CHANNELS, frame_dim=R.HPAC_FILM_DIM, norm_mode="none", activation="relu",
                         use_frame_scale=True, weight_bound=127, activation_bound=127, use_weight_scales=True,
                         weight_exponent_min=-6, use_spm=True, use_norm_gates=False).eval()


def _int_exact(t, name):
    a = t.detach().cpu().double().numpy()
    if not np.array_equal(a, np.round(a)):
        raise ValueError(f"{name}: non-integer values (round first)")
    return a.astype(np.int64)


def _rows(model):
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, COMPRESSED):
            continue
        w = _int_exact(m.weight, name + ".weight")
        if np.abs(w).max() > m.weight_bound:
            raise ValueError(f"{name}.weight: |w| > {m.weight_bound}")
        if isinstance(m, IntegerConv2d):
            mask = m.mask.to(torch.bool).expand_as(m.weight).cpu().numpy()
            if np.any(w[~mask]):
                raise ValueError(f"{name}.weight: non-zero entries under the conv mask")
            rows.extend(w[i][mask[i]] for i in range(w.shape[0]))
        else:
            rows.extend(w[i].reshape(-1) for i in range(w.shape[0]))
    return tuple(r.astype(np.int16) for r in rows)


def encode_ihs1(state_dict, depths=None):
    """IHS1 bytes. state_dict values must be integers. depths: 517 row depths (default: tight)."""
    model = hpac_shell()
    model.load_state_dict({k: v.detach().cpu().float() for k, v in state_dict.items()})
    layout = I2.layout_from_model(model)
    rows = _rows(model)
    tight = np.asarray([I2.minimum_signed_depth(r) for r in rows], dtype=np.uint8)
    d = tight if depths is None else np.asarray(depths, dtype=np.uint8)
    if np.any(d > 15) or np.any(tight > d):
        raise ValueError("bad row depths")
    out = bytearray(MAGIC)
    out += I2._pack_nibbles(d)
    out += I2._write_signed_rows(rows, d, layout)
    mods = dict(model.named_modules())
    for name, p in model.named_parameters():
        mname, field = name.rsplit(".", 1)
        if field == "weight" and isinstance(mods[mname], COMPRESSED):
            continue
        v = _int_exact(p, name).reshape(-1)
        if field == "bias":
            if v.min(initial=0) < -32768 or v.max(initial=0) > 32767:
                raise ValueError(f"{name} outside int16")
            out += v.astype("<i2").tobytes()
        else:
            if v.min(initial=0) < -128 or v.max(initial=0) > 127:
                raise ValueError(f"{name} outside int8")
            out += v.astype(np.int8).tobytes()
    return bytes(out)


def ihs1_depths(blob):
    return I2._unpack_nibbles(blob[4:4 + 259], 517)


def decode_ihs1(blob):
    return R.load_hpac(blob, torch.device("cpu"))


def section(ihs1):
    return brotli.compress(ihs1, **BROTLI)


def best_section(ihs1):
    # the decoder only needs a valid brotli stream: take the smallest over q10/11, lgwin 16..24, modes
    cands = [brotli.compress(ihs1, quality=q, lgwin=w, mode=m) for q in (10, 11) for w in range(16, 25)
             for m in range(3)]
    return min(cands, key=len)


def round_state(hpac_sd):
    # trainer forward = ste_round of weights/biases/frame codes; exponents are integer and frozen
    out = {}
    for k, v in hpac_sd.items():
        v = v.detach().cpu().float()
        if k.endswith(".exponent"):
            if not torch.equal(v, torch.round(v)):
                raise ValueError(f"{k}: non-integer exponent")
            out[k] = v
        else:
            out[k] = torch.round(v)
    return out


def state_equal(a, b):
    bad = [k for k in a if not torch.equal(a[k].cpu().float(), b[k].cpu().float())]
    return sorted(a) == sorted(b) and not bad, bad


def gate(search=False):
    parts = RA.read_residual_archive(SUB / "archive.zip")
    blob = parts.hpac_blob
    assert blob.startswith(MAGIC)
    ref = decode_ihs1(blob).state_dict()
    d = ihs1_depths(blob)
    mine = encode_ihs1(ref, depths=d)
    tight = encode_ihs1(ref)
    print(f"#141 IHS1 {len(blob)} B; re-encode with its depths byte-identical {mine == blob}; "
          f"tight depths len {len(tight)} (depths == tight: {tight == blob})")
    # stored section: rx1 header then hpac stream
    import zipfile
    from runtime.block_container import decode_model_blocks
    models, _ = decode_model_blocks(zipfile.ZipFile(SUB / "archive.zip").read("p"))
    f = RA.RX1_MODEL_HEADER.unpack_from(models)
    stored = models[RA.RX1_MODEL_HEADER.size:RA.RX1_MODEL_HEADER.size + f[5]]
    assert brotli.decompress(stored) == blob
    sec = section(blob)
    print(f"#141 hpac section {len(stored)} B; brotli{BROTLI} -> {len(sec)} B byte-identical {sec == stored}")
    if search and sec != stored:
        for q in range(12):
            for w in range(10, 25):
                for m in range(3):
                    if brotli.compress(blob, quality=q, lgwin=w, mode=m) == stored:
                        print("match", q, w, m)
    assert mine == blob
    return parts, ref, blob, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=None, help="joint_state.pt with key hpac")
    ap.add_argument("--key", default="hpac")
    ap.add_argument("--out-ihs1", default=None)
    ap.add_argument("--out-section", default=None)
    ap.add_argument("--search", action="store_true", help="search brotli params for the gate")
    args = ap.parse_args()
    parts, ref, blob, stored = gate(args.search)
    if args.state is None:
        return
    st = torch.load(args.state, map_location="cpu", weights_only=False)
    sd = round_state(st[args.key])
    ihs1 = encode_ihs1(sd)
    ok, bad = state_equal(decode_ihs1(ihs1).state_dict(), sd)
    assert ok, bad
    n = sum(int((sd[k] != ref[k].float()).sum()) for k in sd)
    ex = [k for k in sd if k.endswith(".exponent") and not torch.equal(sd[k], ref[k].float())]
    sec = best_section(ihs1)
    assert brotli.decompress(sec) == ihs1
    print(f"{args.state}: {n} values differ from #141's hpac ({len(ex)} exponent tensors differ); "
          f"IHS1 {len(ihs1)} B (#141 {len(blob)}), section {len(sec)} B (#141 {len(stored)}), "
          f"delta {len(sec) - len(stored):+d} B; decode -> load == rounded: {ok}")
    if args.out_ihs1:
        Path(args.out_ihs1).write_bytes(ihs1)
    if args.out_section:
        Path(args.out_section).write_bytes(sec)


if __name__ == "__main__":
    main()
