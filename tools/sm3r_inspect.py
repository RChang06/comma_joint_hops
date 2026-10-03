# dump #141's sm3r renderer header, depths, pruned rows
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB)); sys.path.insert(0, str(SUB / "cpr1"))
from runtime import residual_archive as RA
import inflate as R
import ddm_mp2_semantic_receiver as S
parts = RA.read_residual_archive(SUB / "archive.zip")
b = parts.semantic_blob
print(len(b), b[:10], list(b[4:8]))
tpl = R.SemanticTokenRenderer(96).state_dict()
names = S._quantized_names(tpl)
d, rem = S._decode_depth_nibbles(memoryview(b)[10:], len(names), "x")
print(dict(zip(names, d)))
st = S.unpack_variant_semantic_or_none(b, tpl)
for k, v in st.items():
    if v.ndim >= 2:
        rows = v.reshape(v.shape[0], -1)
        z = int((rows.abs().sum(1) == 0).sum())
        print(k, tuple(v.shape), "zero rows", z)
