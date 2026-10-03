# split a joint_hops archive into its stored sections plus the exact blk2 block plan, so compress.py can rebuild it
import sys, json, struct, zipfile, hashlib, lzma, itertools
from pathlib import Path
import brotli
SUBD = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SUBD))
from runtime import residual_archive as RA
from runtime.block_container import decode_model_blocks, transpose

az = (SUBD / "archive.zip").read_bytes()
p = zipfile.ZipFile(SUBD / "archive.zip").read("p")
models, tail = decode_model_blocks(p)
HDR = RA.RX1_MODEL_HEADER
f = HDR.unpack_from(models, 0)
o = HDR.size
hl, sl, cl = f[-3:]
pieces = {"hpac.sec": models[o:o + hl], "renderer.sec": models[o + hl:o + hl + sl],
          "carrier.sec": models[o + hl + sl:o + hl + sl + cl], "table.bin": tail[:96], "stream.bin": tail[96:]}
assert o + hl + sl + cl == len(models)
for k, v in pieces.items():
    (OUT / k).write_bytes(v)

assert p[:4] == b"BLK2"
mlen, nb = struct.unpack_from("<IB", p, 4)
pos, raw_off, plan = 9, 0, []
for _ in range(nb):
    flag = p[pos]; n = int.from_bytes(p[pos + 1:pos + 4], "little"); pos += 4
    codec, has_tr = flag & 127, bool(flag & 128)
    stride = 0
    if has_tr:
        assert p[pos] == 1; stride = p[pos + 1]; pos += 2
    data = p[pos:pos + n]; pos += n
    if codec == 0:
        raw, params = data, None
    elif codec == 2:
        raw = brotli.decompress(data)
    else:
        raw = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 20}])
    if stride:
        raw = transpose(raw, stride, inverse=True)
    seg = models[raw_off:raw_off + len(raw)]
    assert raw == seg
    src = transpose(seg, stride) if stride else seg
    if codec == 2:
        params = next(dict(quality=q, lgwin=lg, mode=md) for q, lg, md in itertools.product((10, 11), (16, 17, 18, 19, 20, 22, 24), (0, 1, 2))
                      if brotli.compress(src, quality=q, lgwin=lg, mode=md) == data)
    elif codec == 1:
        params = next(dict(lc=lc, lp=lp, pb=pb) for lc, lp, pb in ((0, 0, 0), (3, 0, 2), (0, 1, 0), (1, 1, 1))
                      if lzma.compress(src, format=lzma.FORMAT_RAW, filters=[dict(id=lzma.FILTER_LZMA2, dict_size=1 << 20, lc=lc, lp=lp, pb=pb, preset=9 | lzma.PRESET_EXTREME)]) == data)
    plan.append({"start": raw_off, "end": raw_off + len(raw), "codec": codec, "stride": stride, "params": params})
    raw_off += len(raw)
assert raw_off == mlen and pos == len(p) - len(tail)
plan_doc = {"rx1_header": list(f[1:5]), "blocks": plan, "archive_bytes": len(az), "archive_sha256": hashlib.sha256(az).hexdigest(),
            "pieces_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in pieces.items()}}
(OUT / "plan.json").write_text(json.dumps(plan_doc, indent=1))
print(json.dumps(plan_doc["blocks"]), plan_doc["archive_sha256"])
