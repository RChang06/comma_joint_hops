# renderer int4 codes -> #135's F12 WANS body (inverse of runtime/entropy/renderer_weight_codec.decode_f12_wans_body).
# the decoder freezes the mode mask (0xFDB7: rans vs raw per stream) and all priors = 0, so we encode with exactly
# those, not with the experiment book's per-tensor mode search.
# gate: re-encoding #135's own decoded records must reproduce its body byte for byte.
import sys, lzma, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
EB = HERE / "research/repos/CommaVideoCompressionChallenge_ExperimentBook/src"
for p in (str(SUB), str(SUB / "cpr1"), str(EB)):
    if p not in sys.path:
        sys.path.insert(0, p)
from runtime.entropy import renderer_weight_codec as RW
from runtime.baseline import SEMANTIC_SCHEMA
from runtime import residual_archive as RA
from cpr1_sub4.entropy.adaptive_ans import encode_adaptive
from cpr1_sub4.bits import pack_signed

MASK = bytes((0xB7, 0xFD))


def encode_body(codes_by_name, fp16_raw_by_name, raw_scales_by_name, order=RA.WANS_STREAM_ORDER, check=True):
    # codes_by_name: int arrays in [-7,7] for every w4 tensor; fp16_raw_by_name / raw_scales_by_name: exact bytes
    metadata, streams, w4 = [], [], 0
    for schema in SEMANTIC_SCHEMA:
        if schema.is_fp16:
            raw = fp16_raw_by_name[schema.name]
            assert len(raw) == schema.count * 2, schema.name
            metadata.append(raw)
            continue
        assert len(raw_scales_by_name[schema.name]) == schema.scale_count * 2, schema.name
        metadata.append(raw_scales_by_name[schema.name])
        c = np.asarray(codes_by_name[schema.name]).reshape(-1).astype(np.int64)
        assert c.size == schema.count and c.min() >= -7 and c.max() <= 7, schema.name
        rans = bool(MASK[w4 // 8] & (1 << (w4 % 8)))
        streams.append(encode_adaptive(c + 8, prior_index=0) if rans else pack_signed(c.tolist(), 4))
        w4 += 1
    ordered = [streams[i] for i in order]
    body = RW._offsets_for_streams(ordered) + b"".join(metadata) + b"".join(ordered)
    if check:
        recs = RW.decode_wans1(RW.decode_f12_wans_body(body, order))
        for rec in recs:
            if rec.codes is not None:
                assert np.array_equal(rec.codes.reshape(-1), np.asarray(codes_by_name[rec.schema.name]).reshape(-1)), rec.schema.name
                assert rec.raw_scales == raw_scales_by_name[rec.schema.name], rec.schema.name
            else:
                assert rec.raw_fp16 == fp16_raw_by_name[rec.schema.name], rec.schema.name
    return body


def records_to_inputs(records):
    codes, fp16, scales = {}, {}, {}
    for rec in records:
        if rec.codes is None:
            fp16[rec.schema.name] = rec.raw_fp16
        else:
            codes[rec.schema.name] = rec.codes
            scales[rec.schema.name] = rec.raw_scales
    return codes, fp16, scales


def shipped_body():
    p = zipfile.ZipFile(SUB / "archive.zip").read("p")
    models = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=RA.LZMA_FILTERS).decompress(p)
    off = 4 + RA.IHS2_BODY_BYTES
    return models[off:off + RA.WANS_BODY_BYTES]


if __name__ == "__main__":
    parts = RA.read_residual_archive(SUB / "archive.zip")
    body = encode_body(*records_to_inputs(RW.decode_wans1(parts.semantic_blob)))
    ref = shipped_body()
    print(f"body {len(body)} B, shipped {len(ref)} B, identical={body == ref}")
