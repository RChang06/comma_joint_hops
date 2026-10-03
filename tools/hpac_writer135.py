# hpac state_dict -> #135's IHS2 v3 blob (header b"IHS2\x03\x31": frame E0L0 int4, tight rows, packed biases,
# raw int8 exponents), the exact inverse of runtime/ihs2_gate_a.decode_v3. also the 96 B boundary table body
# (fp16 scale + 125 int6 codes, magic RCF1 elided), the inverse of runtime/residual_archive._decode_fixed_table.
# gate a: #135's own hpac and table must re-encode to its stored bytes.
# run: build the cycle 2 blobs from a joint_state.pt, asserting every storage constraint (no silent clamps).
import argparse, sys, zipfile
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from hpac_integer import IntegerConv2d, IntegerLinear
from runtime import residual_archive as RA
from runtime import ihs2 as I2
from runtime import ihs2_gate_a as G

HEADER = RA.IHS2_HEADER
COMPRESSED = (IntegerConv2d, IntegerLinear)


def hpac_shell():
    # same construction as cpr1/inflate.load_hpac, values left at init
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
    # weight rows in loader order (model.modules(), masked entries per output channel)
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


def encode_hpac(state_dict, original_depths=None):
    """IHS2 v3 bytes as parts.hpac_blob holds them. state_dict values must already be integers.
    original_depths: the 517 'original' nibble depths (decode only needs tight <= original); default = tight."""
    model = hpac_shell()
    model.load_state_dict({k: v.detach().cpu().float() for k, v in state_dict.items()})
    layout = I2.layout_from_model(model)
    rows = _rows(model)
    tight = np.asarray([I2.minimum_signed_depth(r) for r in rows], dtype=np.uint8)
    if np.any(tight > 15):
        raise ValueError("row depth > 15")
    orig = tight if original_depths is None else np.asarray(original_depths, dtype=np.uint8)
    if np.any(tight > orig):
        raise ValueError("tight depth exceeds original depth")
    params = dict(model.named_parameters())
    biases, other, exps, frame = [], [], [], None
    for f in layout.tail_fields:
        v = _int_exact(params[f.name], f.name).reshape(-1)
        if f.name == "frame_embed.weight":
            if v.min() < -8 or v.max() > 7:
                raise ValueError(f"frame_embed outside int4 [-8,7]: [{v.min()}, {v.max()}]")
            frame = v
        elif f.is_exponent:
            if v.min() < -6 or v.max() > 0:
                raise ValueError(f"{f.name} outside [-6,0]")
            exps.append(v.astype(np.int8))
        elif f.is_bias:
            if v.max(initial=0) > 16383 or v.min(initial=0) < -16384:
                raise ValueError(f"{f.name} does not fit 15 signed bits")
            biases.append(v.astype("<i2").tobytes())
        else:
            if v.min(initial=0) < -128 or v.max(initial=0) > 127:
                raise ValueError(f"{f.name} outside int8")
            other.append(v.astype(np.int8).tobytes())
    out = bytearray(HEADER)
    out += I2._pack_nibbles(orig)
    out += I2._pack_nibbles(tight)
    out += I2._write_signed_rows(rows, tight, layout)
    out += I2._pack_nibbles((frame & 0xF).astype(np.uint8))   # e0l0: twos complement nibbles, frame-major
    widths, packed = G._pack_biases(tuple(biases), layout)
    out += widths + packed
    out += b"".join(other)
    out += np.concatenate(exps).tobytes()
    return bytes(out)


def decode_hpac(blob):
    # the shipped path: materialize_ihs1 -> decode_v3 -> load_hpac (_deserialize_self_compressed)
    return R.load_hpac(RA.materialize_ihs1(blob, R), torch.device("cpu"))


def encode_table(codes, scale):
    """96 B body: <f2 scale + 125 signed 6-bit codes, lsb-first (inverse of _decode_fixed_table minus RCF1)."""
    c = np.asarray(codes).reshape(-1)
    if c.size != RA.FIXED_STATES * RA.NUM_CLASSES:
        raise ValueError("table needs 25x5 codes")
    if not np.array_equal(c, np.round(c)):
        raise ValueError("table codes are not integers")
    c = c.astype(np.int64)
    if c.min() < -32 or c.max() > 31:
        raise ValueError(f"table codes outside int6: [{c.min()}, {c.max()}]")
    s = np.asarray([scale], dtype="<f2")
    if float(s[0]) != float(scale):
        raise ValueError(f"table scale {scale!r} is not fp16-exact")
    acc = 0
    for i, v in enumerate(c):
        acc |= (int(v) & 0x3F) << (6 * i)
    body = s.tobytes() + acc.to_bytes((c.size * 6 + 7) // 8, "little")
    t = RA._decode_fixed_table(RA.FIXED_MAGIC + body)
    assert np.array_equal(t.codes.reshape(-1), c) and t.scale == float(s[0])
    return body


def state_equal(a, b):
    bad = [k for k in a if not torch.equal(a[k].cpu().float(), b[k].cpu().float())]
    return (sorted(a) == sorted(b)) and not bad, bad


def round_trained(hpac_sd, ref_sd):
    # the trainer's forward: codes() = ste_round(clamp(w,+-127)) * mask, ste_round(clamp(b,+-32768)),
    # frame_codes() = ste_round(clamp(e,+-127)), exponents frozen. check the clamps were no-ops, then round.
    out, report = {}, []
    shell = hpac_shell()
    mods = dict(shell.named_modules())
    for k, v in hpac_sd.items():
        v = v.detach().cpu().float()
        r = torch.round(v)
        mname, field = k.rsplit(".", 1)
        m = mods[mname]
        if field == "exponent":
            if not torch.equal(v, ref_sd[k].cpu().float()):
                raise ValueError(f"{k}: exponent moved (should be frozen)")
        elif k == "frame_embed.weight":
            if r.min() < -8 or r.max() > 7:
                raise ValueError(f"{k}: rounded range [{r.min():.0f}, {r.max():.0f}] outside int4 [-8,7]")
            report.append(f"{k}: rounded range [{r.min():.0f}, {r.max():.0f}]")
        elif field == "weight" and isinstance(m, COMPRESSED):
            if r.abs().max() > 127:
                raise ValueError(f"{k}: |round(w)| max {r.abs().max():.0f} > 127")
            if isinstance(m, IntegerConv2d):
                mask = m.mask.to(torch.bool).expand_as(r)
                if torch.any(r[~mask] != 0):
                    raise ValueError(f"{k}: non-zero rounded weights under the mask")
        elif field == "bias":
            if r.min() < -16384 or r.max() > 16383:
                raise ValueError(f"{k}: rounded bias [{r.min():.0f}, {r.max():.0f}] exceeds 15 signed bits")
        else:
            raise ValueError(f"{k}: unexpected parameter")
        out[k] = r if field != "exponent" else v
    return out, report


def gate_a():
    parts = RA.read_residual_archive(SUB / "archive.zip")
    blob = parts.hpac_blob
    ref = decode_hpac(blob).state_dict()
    orig = I2._unpack_nibbles(blob[6:6 + 259], 517)
    tight = I2._unpack_nibbles(blob[6 + 259:6 + 518], 517)
    print(f"#135 hpac blob {len(blob)} B, original depths == tight: {np.array_equal(orig, tight)} "
          f"({int((orig != tight).sum())} rows differ)")
    mine_o = encode_hpac(ref, original_depths=orig)
    print(f"gate a hpac (#135 original depths): byte-identical {mine_o == blob}, len {len(mine_o)}")
    mine_t = encode_hpac(ref)
    ok, bad = state_equal(decode_hpac(mine_t).state_dict(), ref)
    print(f"gate a hpac (original = tight): len {len(mine_t)}, differs from #135 only in the first depth array "
          f"{mine_t[:6] + mine_t[6 + 259:] == blob[:6] + blob[6 + 259:]}, decode -> load identical {ok} {bad}")
    tb = encode_table(parts.table.codes, parts.table.scale)
    # residual_payload carries the elided RCF1 magic; in p the 96 B body sits right after the lzma section
    p = zipfile.ZipFile(SUB / "archive.zip").read("p")
    n = len(parts.compressed_models)
    tb_ok = tb == parts.residual_payload[4:] == p[n:n + 96]
    print(f"gate a table: byte-identical {tb_ok} ({len(tb)} B), "
          f"codes range [{parts.table.codes.min()}, {parts.table.codes.max()}], scale {parts.table.scale!r}")
    assert mine_o == blob and ok and tb_ok
    return parts, ref, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(HERE / "cycle1/c1_s3/joint_state.pt"))
    ap.add_argument("--out-hpac", default=str(HERE / "cycle2/hpac_c2.bin"))
    ap.add_argument("--out-table", default=str(HERE / "cycle2/table_c2.bin"))
    args = ap.parse_args()
    parts, ref, orig = gate_a()

    st = torch.load(args.state, map_location="cpu", weights_only=False)
    rounded, report = round_trained(st["hpac"], ref)
    for line in report:
        print(line)
    blob = encode_hpac(rounded)
    ok, bad = state_equal(decode_hpac(blob).state_dict(), rounded)
    assert ok, bad
    assert blob.startswith(HEADER)
    changed = [k for k in rounded if not torch.equal(rounded[k], ref[k].float())]
    print(f"trained hpac: {len(changed)}/{len(rounded)} tensors differ from #135; decode -> load == rounded: {ok}")
    print(f"new hpac blob {len(blob)} B (IHS2_BYTES = {len(blob)}, IHS2_BODY_BYTES = {len(blob) - len(HEADER)}); "
          f"#135 was {len(parts.hpac_blob)}")

    tc, ts = st["table_codes"], float(st["table_scale"])
    if ts != float(parts.table.scale):
        raise ValueError(f"trained table scale {ts!r} != #135's {parts.table.scale!r}")
    tc = tc.detach().cpu().double().numpy() if torch.is_tensor(tc) else np.asarray(tc, dtype=np.float64)
    if not np.array_equal(tc, np.round(tc)):
        raise ValueError("trained table codes are not integers")
    table = encode_table(tc, ts)
    print(f"table: codes range [{tc.min():.0f}, {tc.max():.0f}], {int((tc.reshape(25, 5) != parts.table.codes).sum())}"
          f"/125 codes changed, body {len(table)} B: {table.hex()}")
    Path(args.out_hpac).write_bytes(blob)
    Path(args.out_table).write_bytes(table)
    print(f"wrote {args.out_hpac} and {args.out_table}")


if __name__ == "__main__":
    main()
