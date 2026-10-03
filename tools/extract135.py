# pull #135's renderer weights, pose carrier and frame-0 selector out of its archive, the same way its decoder
# does, so the map-training script can use them without the full inflate.
# output: work/extracted/pr135_components.pt  (tokens come from exact_cost135.py: pr135_cost/tokens_pr135.u8)
import sys, struct
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUB = HERE / "pr135/submissions/semantic-pose-HPAC_CPR1_polished"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from runtime.residual_archive import read_residual_archive
from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
from runtime.entropy.renderer_weight_codec import decode_wans1

parts = read_residual_archive(SUB / "archive.zip")
carrier, selector = split_frame0_selector_carrier(parts.carrier_blob)
canonical = materialize_cpr1(carrier, R)
marker = bytes(40_252)
_, basis_raw, coefficients = R.unpack_semantic_pose(
    struct.pack("<II", len(marker), len(canonical)) + marker + canonical)
semantic = R.SemanticTokenRenderer(96)
state = {r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32))
         for r in decode_wans1(parts.semantic_blob)}
semantic.load_state_dict(state, strict=True)

out = HERE / "extracted/pr135_components.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"renderer": {k: v.clone() for k, v in semantic.state_dict().items()},
            "basis_raw": basis_raw.float().clone(), "coefficients": coefficients.float().clone(),
            "selector": bytes(selector or b""), "table": parts.table.values.copy()}, out)
print("renderer params", sum(v.numel() for v in semantic.state_dict().values()),
      "basis", tuple(basis_raw.shape), "coefficients", tuple(coefficients.shape),
      "selector", len(selector or b""), "table", parts.table.values.shape, "->", out)
