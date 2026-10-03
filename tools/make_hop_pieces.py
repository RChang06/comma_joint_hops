# renderer wans body + carrier blob from a trainer state (moved renderer, refit carrier), for archive building
import sys
from pathlib import Path
import numpy as np
import torch
import wans_writer135 as WW
import carrier_writer135 as CW
from runtime.entropy import renderer_weight_codec as RW
from runtime import residual_archive as RA
from runtime.baseline import SEMANTIC_SCHEMA

st_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
st = torch.load(st_dir / "joint_state.pt", weights_only=False)
masters, logs = st["render_masters"], st.get("render_logs", {})
parts = RA.read_residual_archive(WW.SUB / "archive.zip")
recs = {r.schema.name: r for r in RW.decode_wans1(parts.semantic_blob)}
codes, fp16, scales, changed = {}, {}, {}, {}
for sch in SEMANTIC_SCHEMA:
    k, r = sch.name, recs[sch.name]
    m = masters[k].float()
    if r.codes is not None:
        c = m.round().clamp(-7, 7).numpy().astype(np.int8)
        codes[k] = c
        sc = logs[k].exp().reshape(-1).numpy() if k in logs else np.asarray(r.scales, dtype=np.float32)
        scales[k] = sc.astype("<f2").tobytes()
        changed[k] = (int((c != r.codes).sum()), int((np.frombuffer(scales[k], "<f2") != np.frombuffer(r.raw_scales, "<f2")).sum()))
    else:
        fp16[k] = m.reshape(-1).numpy().astype("<f2").tobytes()
        changed[k] = int((np.frombuffer(fp16[k], "<f2") != np.frombuffer(r.raw_fp16, "<f2")).sum())
body = WW.encode_body(codes, fp16, scales)
(out / "wans_body.bin").write_bytes(body)
print("renderer body", len(body), "B (fixed 36,040 expected by #135 F12 / #141 plain path)")
print("changed:", {k: v for k, v in changed.items() if v not in (0, (0, 0))})
coef = np.load(st_dir / "coef_codes_joint.npy")
basis = st["basis_codes"].numpy()
blob = CW.encode_carrier(coef, basis)
(out / "carrier.bin").write_bytes(blob)
print("carrier blob", len(blob), "B  gray", st["gray"], "amp", st["amp"])
