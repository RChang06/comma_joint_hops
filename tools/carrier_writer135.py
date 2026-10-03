# carrier int codes -> #135's carrier blob (F0C1 + u16 len + CAP1 + F0E1 selector), the inverse of
# runtime/carrier_repack.split_frame0_selector_carrier + materialize_cpr1 + cpr1/carrier_codec.decode_compact_carrier.
# scales (12 f4 basis + 12 f4 coefficient) and the selector bytes are kept from #135; huffman lengths, rice ks and the
# ar(1)+bias predictor are refit from the codes (the cap1 decoder re-derives the canonical cpr1 and checks it).
# gate: re-encoding #135's own decoded codes must reproduce parts.carrier_blob byte for byte.
import struct
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
EB = HERE / "research/repos/CommaVideoCompressionChallenge_ExperimentBook/src"
FS = HERE / "research/repos/comma-ai-semantic-pose-hpac-cpr1/code"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1")); sys.path.insert(0, str(EB))
from runtime import residual_archive as RA
from runtime.carrier_repack import (_predict_residuals, _rice_encode, _zigzag, materialize_cpr1,
                                    pack_frame0_selector_carrier, split_frame0_selector_carrier)
from runtime.entropy.coefficient_ar1_codec import decode_cap1
from carrier_codec import decode_compact_carrier
from cpr1_sub4.entropy.coefficient_predictor import ar1_bias_residuals, fit_ar1_bias, pack_ar1_bias_metadata

FRAMES, DIMS, BASIS_SHAPE = 600, 12, (12, 3, 24, 32)
CAP1_PREFIX = b"CAP1\x01\x00\x00\x00"
SELECTOR_PREFIX = b"F0E1\x01"

# fs huffman builder (#135's encoder); loaded by path so it does not shadow the decoder's carrier_codec
import importlib.util
_spec = importlib.util.spec_from_file_location("fs_carrier_codec", FS / "carrier_codec.py")
FSC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(FSC)


def _reference():
    parts = RA.read_residual_archive(SUB / "archive.zip")
    cap1, selector = split_frame0_selector_carrier(parts.carrier_blob)
    scales = cap1[14 + 3 * DIMS: 14 + 3 * DIMS + 8 * DIMS]
    return parts, cap1, selector, scales


def _u24(v):
    assert 0 < v < 1 << 24
    return v.to_bytes(3, "little")


def _check_codes(coef_codes, basis_codes):
    k = np.asarray(coef_codes)
    b = np.asarray(basis_codes)
    assert k.shape == (FRAMES, DIMS) and b.size == int(np.prod(BASIS_SHAPE)), (k.shape, b.shape)
    assert np.all(k == np.round(k)) and np.all(b == np.round(b)), "codes must be integer valued"
    k, b = k.astype(np.int64), b.astype(np.int64).reshape(-1)
    assert k.min() >= -2048 and k.max() <= 2047, ("coef out of int12", k.min(), k.max())
    assert b.min() >= -15 and b.max() <= 15, ("basis out of [-15,15]", b.min(), b.max())
    return k.astype(np.int32), b


def encode_cap1_fields(coef_codes, basis_codes, scales):
    # returns the cap1 fields: (basis_bits, residual_bits, dict predictor/scales/lengths/ks/basis/rice)
    k, b = _check_codes(coef_codes, basis_codes)
    assert len(scales) == 8 * DIMS
    lengths, basis_payload, basis_bits = FSC._encode_huffman(FSC._zigzag_signed(b, 5))
    model = fit_ar1_bias(k)
    ks, rice, residual_bits = _rice_encode(_zigzag(ar1_bias_residuals(k, model)), 1)
    fields = {
        "predictor": pack_ar1_bias_metadata(model),
        "scales": bytes(scales),
        "lengths": np.asarray(lengths, dtype=np.uint8).tobytes(),
        "ks": ks.reshape(-1).tobytes(),
        "basis": basis_payload,
        "rice": rice,
    }
    return basis_bits, residual_bits, fields


def encode_cap1(coef_codes, basis_codes, scales):
    basis_bits, residual_bits, f = encode_cap1_fields(coef_codes, basis_codes, scales)
    return CAP1_PREFIX + _u24(basis_bits) + _u24(residual_bits) + b"".join(f[n] for n in RA.CAP_FIELDS)


def f24s_carrier_part(carrier_blob):
    # the tail of the f24s models blob: cap1[8:14] counts, fields in STORED_CAP_FIELDS order, then selector[5:]
    cap1, selector = split_frame0_selector_carrier(carrier_blob)
    assert cap1[:8] == CAP1_PREFIX and selector[:5] == SELECTOR_PREFIX
    basis_bits, residual_bits = int.from_bytes(cap1[8:11], "little"), int.from_bytes(cap1[11:14], "little")
    sizes = {"predictor": 3 * DIMS, "scales": 8 * DIMS, "lengths": 32, "ks": DIMS,
             "basis": (basis_bits + 7) // 8, "rice": (residual_bits + 7) // 8}
    f, off = {}, 14
    for n in RA.CAP_FIELDS:
        f[n] = cap1[off: off + sizes[n]]
        off += sizes[n]
    assert off == len(cap1)
    out = cap1[8:14] + b"".join(f[n] for n in RA.STORED_CAP_FIELDS) + selector[5:]
    # the shipped parser must give back the same cap1 and selector
    stored = out[: RA._cap1_body_bytes(out)]
    assert RA._restore_cap1(stored) == cap1 and SELECTOR_PREFIX + out[len(stored):] == selector
    return out


def encode_carrier(coef_codes, basis_codes, scales=None, selector=None):
    # coef_codes [600,12] int12 coefficient codes, basis_codes [12,3,24,32] in [-15,15] -> parts.carrier_blob layout
    if scales is None or selector is None:
        _, _, ref_selector, ref_scales = _reference()
        scales = ref_scales if scales is None else scales
        selector = ref_selector if selector is None else selector
    cap1 = encode_cap1(coef_codes, basis_codes, scales)
    decode_cap1(cap1, frames=FRAMES, dimensions=DIMS)      # canonical cpr1 self check
    return pack_frame0_selector_carrier(cap1, selector)


def decode_carrier(blob):
    # shipped decoder path -> (coef int12 codes [600,12], basis codes [12,3,24,32], basis scales, coef scales, selector)
    cap1, selector = split_frame0_selector_carrier(blob)
    cpr1 = materialize_cpr1(cap1, type("R", (), {"N": FRAMES, "CARRIER_DIM": DIMS}))
    bs, bc, cs, enc = decode_compact_carrier(cpr1, basis_count=int(np.prod(BASIS_SHAPE)), frames=FRAMES, dimensions=DIMS)
    enc = enc.astype(np.int64)
    k = np.cumsum((enc >> 1) ^ -(enc & 1), axis=0) & 0xFFF       # same as cpr1/inflate.py
    k = np.where(k >= 0x800, k - 0x1000, k).astype(np.int32)
    return k, bc.reshape(BASIS_SHAPE).astype(np.int32), bs, cs, selector


if __name__ == "__main__":
    import torch
    import inflate as R
    parts, cap1_ref, sel_ref, scales_ref = _reference()
    ref = parts.carrier_blob

    # gate: #135's own codes, decoded exactly like joint_train135.py
    canon = materialize_cpr1(split_frame0_selector_carrier(ref)[0], R)
    bs, bc, cs, enc = R.decode_compact_carrier(canon, basis_count=R.CARRIER_DIM * 3 * R.CARRIER_H * R.CARRIER_W,
                                               frames=R.N, dimensions=R.CARRIER_DIM)
    k0, b0, bs0, cs0, _ = decode_carrier(ref)
    assert np.array_equal(b0.reshape(-1), bc) and np.array_equal(bs0, bs) and np.array_equal(cs0, cs)
    # cross check the int12 codes against the canonical cpr1 predictor (residual = delta)
    assert np.array_equal(_zigzag(_predict_residuals(k0, np.zeros(DIMS, np.uint8), 0)), enc)
    out = encode_carrier(k0, b0)
    print(f"gate: ref {len(ref)} B, ours {len(out)} B, identical={out == ref}")
    if out != ref:
        c_ours, _ = split_frame0_selector_carrier(out)
        diff = [i for i in range(min(len(c_ours), len(cap1_ref))) if c_ours[i] != cap1_ref[i]]
        print("  first cap1 diffs at", diff[:10])
        k1, b1, bs1, cs1, s1 = decode_carrier(out)
        assert np.array_equal(k1, k0) and np.array_equal(b1, b0) and np.array_equal(bs1, bs0)
        assert np.array_equal(cs1, cs0) and s1 == sel_ref
        print("  decode-identical fallback gate ok")
    part = f24s_carrier_part(ref)
    import lzma
    raw = lzma.decompress(parts.compressed_models, format=lzma.FORMAT_RAW, filters=RA.LZMA_FILTERS)
    print(f"f24s tail: {len(part)} B, matches models tail={raw.endswith(part)}")

    # trained state
    kc = np.load(HERE / "cycle2/c2_s4/coef_codes_joint.npy")
    st = torch.load(HERE / "cycle2/c2_s4/joint_state.pt", map_location="cpu", weights_only=False)
    bt = st["basis_codes"].detach().cpu().double().numpy()
    print(f"trained: coef {kc.dtype} {kc.shape} [{kc.min()},{kc.max()}], basis {tuple(bt.shape)} [{bt.min()},{bt.max()}]")
    assert np.all(bt == np.round(bt))
    blob = encode_carrier(kc, bt.astype(np.int64))
    k2, b2, bs2, cs2, s2 = decode_carrier(blob)
    assert np.array_equal(k2, kc.astype(np.int32)) and np.array_equal(b2, bt.reshape(BASIS_SHAPE).astype(np.int32))
    assert np.array_equal(bs2, bs0) and np.array_equal(cs2, cs0) and s2 == sel_ref
    # the f24s parser path too
    p2 = f24s_carrier_part(blob)
    print(f"trained carrier: {len(blob)} B vs #135 {len(ref)} B ({len(blob) - len(ref):+d}), f24s tail {len(p2)} B")
    (HERE / "cycle2/carrier_c2.bin").write_bytes(blob)
    print("saved cycle2/carrier_c2.bin")
