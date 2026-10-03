# assemble a #141-container archive (BLK2 + RX1, #141's decoder) from:
#   - #141's hpac section (brotli ihs1, verbatim) and 96 B rcf1 table body (verbatim)
#   - a renderer F12 WANS body (36,040 B) from a #135-format archive, wrapped in a one-segment LAY1
#   - a #135-layout carrier blob (F0C1 + u16 + CAP1 + F0E1 selector), re-stored as packed CAP1 metadata with
#     #141's dx2 (cabac coefficients) and rr5 (adaptive arith basis) recoders, reserved bits 0x18 as in #141
#   - a token stream coded with #141's coder (work/encode141.py)
# gates: (a) #141's own carrier re-stores to its shipped carrier section byte for byte; (b) the new archive
# parses with #141's read_residual_archive and every piece round-trips (renderer records equal the #135-format
# archive's, carrier blob equals the input, hpac/table/stream equal what went in).
import argparse, sys, io, re, lzma, json, struct, zipfile, hashlib
from pathlib import Path
import numpy as np
import brotli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUB = ROOT / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB))
from runtime import residual_archive as RA
from runtime.block_container import decode_model_blocks, transpose
from runtime.segment_layout import pack_segments
from runtime.carrier_repack import split_frame0_selector_carrier
from runtime import rr5_arith_basis as RR5
from runtime import dx2_cabac_coefficients as DX2
from runtime.entropy.renderer_weight_codec import decode_wans1

ap = argparse.ArgumentParser()
ap.add_argument("--src135", default=str(HERE / "final_c2/sub_c2"), help="#135-format submission dir (renderer)")
ap.add_argument("--carrier", default=str(HERE / "cycle2/carrier_c2.bin"))
ap.add_argument("--stream", default=str(HERE / "final_c2/stream141_c2map.bin"))
ap.add_argument("--out", default=str(HERE / "final_c2/archive141_c2.zip"))
ap.add_argument("--no-stream", action="store_true", help="pieces + gates only, use #141's stream as a stand-in")
args = ap.parse_args()

HDR = RA.RX1_MODEL_HEADER


def zip_p(p):
    # compress.py's zip settings
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, p)
    return buf.getvalue()


def pack_cap1_metadata(stored):
    # inverse of residual_archive._restore_packed_cap1_metadata: 80 B of predictor/lengths/ks -> 40 B
    head, meta, rest = stored[:102], stored[102:182], stored[182:]
    factors = np.frombuffer(meta[0:24], dtype="<i2").astype(np.int64)
    biases = np.frombuffer(meta[24:36], dtype=np.int8).astype(np.int64)
    lengths = np.frombuffer(meta[36:68], dtype=np.uint8).astype(np.int64)
    ks = np.frombuffer(meta[68:80], dtype=np.uint8).astype(np.int64)
    fbase = int(factors.min())
    kbase = int(ks.min())
    assert 0 <= fbase <= 255 and 0 <= kbase <= 255, (fbase, kbase)
    packed = (head + bytes((fbase,)) + RR5.pack_unsigned(factors - fbase, 7)
              + RR5.pack_unsigned(biases & 63, 6) + RR5.pack_unsigned(lengths, 4)
              + bytes((kbase,)) + RR5.pack_unsigned(ks - kbase, 1) + rest)
    assert len(packed) == len(stored) - 40
    assert RA._restore_packed_cap1_metadata(packed[:142]) == stored[:182], "packed cap1 metadata inverse"
    return packed


def store_cap1(cap1):
    # inverse of residual_archive._restore_cap1 (same as build_archive135.store_cap1)
    assert cap1.startswith(RA.CAP1_PREFIX)
    body = cap1[len(RA.CAP1_PREFIX):]
    counts = body[:6]
    bb = int.from_bytes(counts[:3], "little"); rb = int.from_bytes(counts[3:6], "little")
    sizes = {"predictor": 36, "scales": 96, "lengths": 32, "ks": 12, "basis": (bb + 7) // 8, "rice": (rb + 7) // 8}
    fields, off = {}, 6
    for name in RA.CAP_FIELDS:
        fields[name] = body[off:off + sizes[name]]; off += sizes[name]
    assert off == len(body)
    stored = counts + b"".join(fields[n] for n in RA.STORED_CAP_FIELDS)
    assert RA._restore_cap1(stored) == cap1
    return stored


def carrier_section(f0c1):
    # F0C1 blob -> #141 carrier section (packed cap1, dx2 then rr5, + selector body without its F0E1 prefix)
    cap1, selector = split_frame0_selector_carrier(f0c1)
    assert selector is not None and selector.startswith(RA.SPARSE_SELECTOR_PREFIX)
    packed = pack_cap1_metadata(store_cap1(cap1))
    d = DX2.apply_cabac_to_carrier_body(packed)["body"]
    r = RR5.apply_rider_to_carrier_body(d)
    body = r["body"]
    # decoder order: rr5 restore, then dx2 restore
    assert DX2.restore_carrier_body(RR5.restore_carrier_body(body)) == packed, "rider inverse"
    return body + selector[len(RA.SPARSE_SELECTOR_PREFIX):], r["table_dropped"]


# --- #141's pieces ------------------------------------------------------------------------------------------
p141 = zipfile.ZipFile(SUB / "archive.zip").read("p")
models141, tail141 = decode_model_blocks(p141)
f = HDR.unpack_from(models141)
hpac_len, sem_len, car_len = f[-3:]
o = HDR.size
hpac_stream = models141[o:o + hpac_len]; o += hpac_len
o += sem_len
car141 = models141[o:o + car_len]
parts141 = RA.read_residual_archive(SUB / "archive.zip")
table_body = tail141[:96]
assert RA.FIXED_MAGIC + table_body == parts141.residual_payload

# gate a: #141's own carrier re-stores to its shipped section
mine, dropped = carrier_section(parts141.carrier_blob)
print(f"gate a (#141 carrier re-store): {'PASS' if mine == car141 else 'FAIL'}  ({len(mine)} vs {len(car141)} B, "
      f"huffman table dropped {dropped})")
assert mine == car141

# --- new pieces ---------------------------------------------------------------------------------------------
src = Path(args.src135)
ra_src = (src / "runtime/residual_archive.py").read_text()
ihs2_bytes = int(re.search(r"^IHS2_BYTES = ([\d_]+)", ra_src, re.M).group(1).replace("_", ""))
p135 = zipfile.ZipFile(src / "archive.zip").read("p")
dz = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=RA.LZMA_FILTERS)
models135 = dz.decompress(p135)
assert models135.startswith(RA.F24S_MAGIC)
wans_off = 4 + ihs2_bytes - len(RA.IHS2_HEADER)
f12 = models135[wans_off:wans_off + RA.WANS_BODY_BYTES]
assert len(f12) == RA.WANS_BODY_BYTES
sem_section = pack_segments([f12])

carrier_in = Path(args.carrier).read_bytes()
car_section, dropped = carrier_section(carrier_in)
print(f"carrier: input {len(carrier_in)} B -> section {len(car_section)} B (huffman table dropped {dropped})")

stream = parts141.token_stream if args.no_stream else Path(args.stream).read_bytes()
reserved = f[4]
assert reserved == 0x18
models = HDR.pack(RA.RX1_MAGIC, 1, RA.RX1_CODEC_BROTLI, 0, reserved, len(hpac_stream), len(sem_section),
                  len(car_section)) + hpac_stream + sem_section + car_section
tail = table_body + stream


# --- blk2 block layout: small search over section-aligned splits -------------------------------------------
def enc_block(data, codec, stride=0, params=None):
    d = transpose(data, stride) if stride else data
    if codec == 1:
        d = lzma.compress(d, format=lzma.FORMAT_RAW, filters=[params])
    elif codec == 2:
        d = brotli.compress(d, **params)
    tr = bytes((1, stride)) if stride else b""
    return bytes((codec | (128 if stride else 0),)) + len(d).to_bytes(3, "little") + tr + d


def best_block(data):
    cands = [enc_block(data, 0)]
    for stride in (0, 2, 4):
        for lg in (16, 18, 20, 22, 24):
            cands.append(enc_block(data, 2, stride, dict(quality=11, lgwin=lg)))
        for lc, lp, pb in ((0, 0, 0), (3, 0, 2), (0, 1, 0), (1, 1, 1)):
            cands.append(enc_block(data, 1, stride, dict(id=lzma.FILTER_LZMA2, dict_size=1 << 20, lc=lc, lp=lp,
                                                         pb=pb, preset=9 | lzma.PRESET_EXTREME)))
    return min(cands, key=len)


# candidate cut points: section boundaries plus a few refinements inside the carrier/renderer
s1 = HDR.size + len(hpac_stream)
s2 = s1 + len(sem_section)
s3 = len(models)
cuts = sorted({s1, s2, s2 + 142, s2 + 142 + 2000, s1 + 6 + 2 * 1 + 4000, s1 + 8300})
cuts = [c for c in cuts if 0 < c < s3]
# dynamic program over cut subsets (cuts are few), each block coded with its best codec
pts = [0] + cuts + [s3]
cache = {}


def cost(i, j):
    if (i, j) not in cache:
        cache[(i, j)] = best_block(models[pts[i]:pts[j]])
    return cache[(i, j)]


best = {0: (0, [])}
for j in range(1, len(pts)):
    opts = []
    for i in range(j):
        if i in best:
            b = cost(i, j)
            opts.append((best[i][0] + len(b), best[i][1] + [b]))
    best[j] = min(opts, key=lambda t: t[0])
blocks = best[len(pts) - 1][1]
p = b"BLK2" + struct.pack("<IB", len(models), len(blocks)) + b"".join(blocks) + tail
assert decode_model_blocks(p) == (models, tail)
archive = zip_p(p)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_bytes(archive)

# --- gate b: parse back with #141's reader ------------------------------------------------------------------
parts = RA.read_residual_archive(Path(args.out))
ok = {
    "stream": parts.token_stream == stream,
    "table": parts.residual_payload == parts141.residual_payload,
    "hpac": parts.hpac_blob == parts141.hpac_blob,
    "carrier": parts.carrier_blob == carrier_in,
    "compensation_none": parts.compensation_blob is None,
    "semantic_is_wans1": parts.semantic_blob.startswith(b"WANS"),
}
ref = RA.decode_f12_wans_body(f12, RA.WANS_STREAM_ORDER)
ok["semantic_equals_src135_wans1"] = parts.semantic_blob == ref
recs_a = {r.schema.name: np.asarray(r.values) for r in decode_wans1(parts.semantic_blob)}
recs_b = {r.schema.name: np.asarray(r.values) for r in decode_wans1(ref)}
ok["renderer_records_equal"] = recs_a.keys() == recs_b.keys() and all(np.array_equal(recs_a[k], recs_b[k]) for k in recs_a)
print("gate b:", json.dumps(ok))
assert all(ok.values())

zip_overhead = len(archive) - len(p)
blk_bytes = len(p) - len(tail)
print(f"models {len(models)} B raw -> blk2 {blk_bytes} B in {len(blocks)} blocks "
      f"(hpac {len(hpac_stream)}, renderer lay1 {len(sem_section)}, carrier {len(car_section)} raw)")
for b in blocks:
    print(f"  block codec {b[0] & 127} transform {bool(b[0] & 128)} stored {int.from_bytes(b[1:4], 'little')}")
print(f"table 96 B, stream {len(stream)} B, zip overhead {zip_overhead} B")
print(f"archive {len(archive):,} B sha256 {hashlib.sha256(archive).hexdigest()} -> {args.out}")
