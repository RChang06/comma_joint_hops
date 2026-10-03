# renderer records <-> #141 tagged renderer blobs (SM3R mode 5/6, SD1M), plus LAY1 grouping via #141's
# compress.segment_semantic. exact inverse of cpr1/ddm_mp2_semantic_receiver.py.
# gate (python work/sm3r_writer.py): #141's own sm3r blob parses to codes/scales and re-encodes byte for byte,
# also when the codes are re-derived from the decoded float weights; its lay1 section is reproduced too.
import sys, struct
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submissions/semantic_blocks"
for p in (str(SUB), str(SUB / "cpr1")):
    if p not in sys.path:
        sys.path.insert(0, p)
import ddm_mp2_semantic_receiver as S
import inflate as R

TEMPLATE = R.SemanticTokenRenderer(96).state_dict()
NAMES = S._quantized_names(TEMPLATE)


def scale_axis(name, ndim):
    return ndim - 1 if name.endswith("embed.weight") else 0


def pack_signed_bits(codes, bits):
    c = np.asarray(codes).reshape(-1).astype(np.int64)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    assert c.size == 0 or (c.min() >= lo and c.max() <= hi), (bits, c.min(), c.max())
    u = c & ((1 << bits) - 1)
    bitstream = ((u[:, None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitstream, bitorder="little").tobytes()


def encode(entries, fmt="SM3R6", keep_percent=None):
    # entries: name -> dict. fp16 tensors: {"fp16": raw bytes}. quantized: {"codes": int array (full shape,
    # or (kept, cols) for pruned), "scales": raw fp16 bytes (kept rows only for pruned), "depth": bits,
    # "rows": bool mask over rows (pruned tensors only)}
    if fmt == "SD1M":
        out = bytearray(b"SD1M" + bytes((1, len(NAMES))))
        pruned = False
    else:
        mode = {"SM3R5": 5, "SM3R6": 6}[fmt]
        assert 1 <= keep_percent < 100
        out = bytearray(b"SM3R" + bytes((1, mode, keep_percent, 0)))
        out += struct.pack("<H", S._selection_mask(NAMES, S.ROW_PRUNE_NAMES))
        pruned = True
    depths = [entries[n]["depth"] for n in NAMES]
    if fmt in ("SD1M", "SM3R6"):
        d = depths + [0] * (len(depths) & 1)
        out += bytes(d[i] | (d[i + 1] << 4) for i in range(0, len(d), 2))
    else:
        assert all(x == 4 for x in depths)
    for name, value in TEMPLATE.items():
        e = entries[name]
        if value.ndim < 2:
            assert len(e["fp16"]) == value.numel() * 2, name
            out += e["fp16"]
            continue
        codes = np.asarray(e["codes"])
        if pruned and name in S.ROW_PRUNE_NAMES:
            rows = value.shape[0]
            mask = np.asarray(e["rows"], dtype=bool)
            assert mask.size == rows and int(mask.sum()) == max(1, round(rows * keep_percent / 100.0)), name
            out += np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
            assert codes.shape[0] == mask.sum() and len(e["scales"]) == codes.shape[0] * 2, name
        else:
            assert codes.size == value.numel() and len(e["scales"]) == S._scale_count(name, value) * 2, name
        out += e["scales"]
        out += pack_signed_bits(codes, e["depth"])
    blob = bytes(out)
    check(blob, entries)
    return blob


def parse(blob):
    # structural parse of a tagged blob back to entries (inverse of encode)
    mv = memoryview(blob)
    if blob.startswith(b"SD1M"):
        fmt, kp, pruned = "SD1M", None, False
        depths, rem = S._decode_depth_nibbles(mv[6:], len(NAMES), "SD1M")
    else:
        assert blob.startswith(b"SM3R")
        fmt, kp, pruned = ("SM3R5" if blob[5] == 5 else "SM3R6"), blob[6], True
        if blob[5] == 6:
            depths, rem = S._decode_depth_nibbles(mv[10:], len(NAMES), "SM3R")
        else:
            depths, rem = [4] * len(NAMES), mv[10:]
    alloc = dict(zip(NAMES, depths))
    off = len(blob) - len(rem)
    entries = {}
    for name, value in TEMPLATE.items():
        if value.ndim < 2:
            n = value.numel() * 2
            entries[name] = {"fp16": bytes(blob[off:off + n])}; off += n
            continue
        e = {"depth": alloc[name]}
        count, sc = value.numel(), S._scale_count(name, value)
        if pruned and name in S.ROW_PRUNE_NAMES:
            rows = value.shape[0]; mb = (rows + 7) // 8
            mask = np.unpackbits(np.frombuffer(blob[off:off + mb], np.uint8), bitorder="little")[:rows].astype(bool)
            off += mb
            e["rows"] = mask
            count, sc = int(mask.sum()) * (count // rows), int(mask.sum())
        e["scales"] = bytes(blob[off:off + sc * 2]); off += sc * 2
        codes, r2 = S._unpack_signed_bits(mv[off:], count, e["depth"])
        off = len(blob) - len(r2)
        shape = (sc, count // sc) if "rows" in e else tuple(value.shape)
        e["codes"] = codes.numpy().astype(np.int64).reshape(shape)
        entries[name] = e
    assert off == len(blob)
    return entries, fmt, kp


def entries_to_state(entries):
    # dense float32 weights exactly as the receiver builds them
    st = {}
    for name, value in TEMPLATE.items():
        e = entries[name]
        if value.ndim < 2:
            st[name] = torch.from_numpy(np.frombuffer(e["fp16"], "<f2").copy().reshape(value.shape)).float()
            continue
        scales = torch.from_numpy(np.frombuffer(e["scales"], "<f2").copy()).float()
        codes = torch.from_numpy(np.asarray(e["codes"]).astype(np.int8))
        if "rows" in e:
            compact = codes.float() * scales.reshape(-1, 1)
            dense = torch.zeros((value.shape[0], value.numel() // value.shape[0]))
            dense[torch.from_numpy(e["rows"])] = compact
            st[name] = dense.reshape(value.shape)
        else:
            shp = [1] * value.ndim; shp[scale_axis(name, value.ndim)] = -1
            st[name] = codes.reshape(value.shape).float() * scales.reshape(shp)
    return st


def check(blob, entries):
    got = S.unpack_variant_semantic_or_none(blob, TEMPLATE)
    want = entries_to_state(entries)
    for k in TEMPLATE:
        assert got[k].dtype == want[k].dtype and torch.equal(got[k].view(torch.int32), want[k].view(torch.int32)), k


def entries_from_records(records, depth=None):
    # wans1 TensorStorage records (exact codes / raw scales / raw fp16) -> unpruned entries
    depth = depth or {}
    entries = {}
    for r in records:
        if r.codes is None:
            entries[r.schema.name] = {"fp16": r.raw_fp16}
        else:
            entries[r.schema.name] = {"codes": np.asarray(r.codes).astype(np.int64).reshape(r.schema.shape),
                                      "scales": r.raw_scales, "depth": depth.get(r.schema.name, 4)}
    return entries


def prune_entries(entries, keep_percent, drop):
    # drop: name -> list of row indices to zero (must leave exactly the keep count)
    out = {k: dict(v) for k, v in entries.items()}
    for name in S.ROW_PRUNE_NAMES:
        rows = TEMPLATE[name].shape[0]
        mask = np.ones(rows, dtype=bool); mask[list(drop[name])] = False
        assert int(mask.sum()) == max(1, round(rows * keep_percent / 100.0)), name
        e = out[name]
        sc = np.frombuffer(e["scales"], "<f2")
        e["codes"] = np.asarray(e["codes"]).reshape(rows, -1)[mask]
        e["scales"] = sc[mask].astype("<f2").tobytes()
        e["rows"] = mask
    return out


def lay1(blob):
    import compress as C
    packed, _, sizes = C.segment_semantic(blob)
    return packed


def states_equal(a, b):
    diff = {}
    for k in TEMPLATE:
        x, y = a[k].float(), b[k].float()
        bits = ~torch.eq(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32))
        if bits.any():
            d = (x - y).abs()
            diff[k] = (int(bits.sum()), float(d.max()))
    return diff


def entries_from_state(st, like):
    # re-derive codes from decoded float weights, using `like` for scales/depths/row masks
    out = {}
    for name, value in TEMPLATE.items():
        e = dict(like[name])
        if value.ndim >= 2:
            sc = torch.from_numpy(np.frombuffer(e["scales"], "<f2").copy()).float()
            w = st[name].float()
            if "rows" in e:
                w = w.reshape(value.shape[0], -1)[torch.from_numpy(e["rows"])]
                c = torch.round(w / sc.reshape(-1, 1))
            else:
                shp = [1] * value.ndim; shp[scale_axis(name, value.ndim)] = -1
                c = torch.round(w / sc.reshape(shp))
            e["codes"] = c.numpy().astype(np.int64)
        else:
            e["fp16"] = st[name].numpy().astype("<f2").tobytes()
        out[name] = e
    return out


if __name__ == "__main__":
    from runtime import residual_archive as RA
    from runtime.block_container import decode_model_blocks
    import zipfile
    parts = RA.read_residual_archive(SUB / "archive.zip")
    blob = parts.semantic_blob
    entries, fmt, kp = parse(blob)
    re1 = encode(entries, fmt, kp)
    print(f"gate 1 (structural re-encode): {fmt} keep {kp}% {len(re1)} B identical={re1 == blob}")
    st = S.unpack_variant_semantic_or_none(blob, TEMPLATE)
    re2 = encode(entries_from_state(st, entries), fmt, kp)
    print(f"gate 2 (codes re-derived from decoded floats): identical={re2 == blob}")
    p = zipfile.ZipFile(SUB / "archive.zip").read("p")
    models, _ = decode_model_blocks(p)
    f = RA.RX1_MODEL_HEADER.unpack_from(models)
    o = RA.RX1_MODEL_HEADER.size + f[-3]
    sec = models[o:o + f[-2]]
    l1 = lay1(re1)
    print(f"gate 3 (lay1 grouping): {len(l1)} B identical={l1 == sec}")
    assert re1 == blob and re2 == blob and l1 == sec
    print("PASS")
