# our renderer (wans body) code ranges and zero rows
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB))
from runtime import residual_archive as RA
from runtime.entropy.renderer_weight_codec import decode_wans1
body = (HERE / "final_B/wans_body.bin").read_bytes()
recs = decode_wans1(RA.decode_f12_wans_body(body, RA.WANS_STREAM_ORDER))
for r in recs:
    if r.codes is None: continue
    c = np.asarray(r.codes).reshape(r.schema.shape[0], -1)
    sc = np.frombuffer(r.raw_scales, "<f2")
    print(r.schema.name, c.min(), c.max(), "zero rows", int((np.abs(c).sum(1) == 0).sum()), "scale min", sc.min(), "neg", int((sc < 0).sum()), "zero", int((sc == 0).sum()))
