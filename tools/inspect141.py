# dump #141's archive layout: blk2 blocks, rx1 header, section sizes, token stream offset
import sys, struct, zipfile, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB))
from runtime import residual_archive as RA
from runtime.block_container import decode_model_blocks
p = zipfile.ZipFile(SUB / "archive.zip").read("p")
print("p bytes", len(p), "magic", p[:4])
expected, count = struct.unpack_from("<IB", p, 4)
print("blk2 models bytes", expected, "blocks", count)
off = 9
for i in range(count):
    codec = p[off]; size = int.from_bytes(p[off+1:off+4], "little"); off += 4
    tr = None
    if codec & 128:
        codec &= 127; tr = tuple(p[off:off+2]); off += 2
    print(f" block {i}: codec {codec} transform {tr} stored {size} at {off}")
    off += size
print("blocks end at", off, "tail bytes", len(p) - off)
models, tail = decode_model_blocks(p)
f = RA.RX1_MODEL_HEADER.unpack_from(models)
print("rx1 header", f, "header size", RA.RX1_MODEL_HEADER.size)
parts = RA.read_residual_archive(SUB / "archive.zip")
print("hpac", len(parts.hpac_blob), parts.hpac_blob[:6])
print("semantic", len(parts.semantic_blob), parts.semantic_blob[:4])
print("carrier", len(parts.carrier_blob), parts.carrier_blob[:4])
print("compensation", parts.compensation_blob)
print("residual table payload", len(parts.residual_payload), "scale", parts.table.scale)
print("table codes\n", parts.table.codes)
print("token stream", len(parts.token_stream), "is tail of p", p.endswith(parts.token_stream), "starts at", len(p) - len(parts.token_stream))
print("tail = rcf1 compact(", len(parts.residual_payload) - 4, ") + tokens:", len(tail) == len(parts.residual_payload) - 4 + len(parts.token_stream))
print("token sha", hashlib.sha256(parts.token_stream).hexdigest())
