# pull everything #141 decodes into one easy file, so experiments skip the 5-minute token decode.
#
# output work/extracted/pr141_components.pt:
#   tokens         uint8 [600, 384, 512]  decoded class maps (from the official decoder's checkpoint)
#   renderer       state_dict of SemanticTokenRenderer(96), float weights exactly as the decoder loads them
#   basis_raw      float [12, 3, 24, 32]   pose carrier patterns (before resize/normalise)
#   coefficients   float [600, 12]         per-pair carrier strengths
#   selector       bytes                   the 14-byte frame-0 selector (modes applied after the carrier)
#
# usage (from the repo root): python work/extract141.py
import sys, struct, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions/semantic_blocks"
sys.path.insert(0, str(SUB))
sys.path.insert(0, str(SUB / "cpr1"))
import inflate as R
from runtime.residual_archive import read_residual_archive
from runtime.carrier_repack import split_frame0_selector_carrier, materialize_cpr1
from runtime.entropy.renderer_weight_codec import decode_wans1

TOKENS = SUB / "inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
OUT = ROOT / "work/extracted/pr141_components.pt"

parts = read_residual_archive(SUB / "archive.zip")

# carrier: same path as f26_inflate (no compensation overlay in #141)
assert parts.compensation_blob is None
carrier, selector = split_frame0_selector_carrier(parts.carrier_blob)
canonical = materialize_cpr1(carrier, R)
marker = bytes(40_252)
_, basis_raw, coefficients = R.unpack_semantic_pose(
    struct.pack("<II", len(marker), len(canonical)) + marker + canonical
)

# renderer weights: same path as f26_inflate
semantic = R.SemanticTokenRenderer(96)
state = R.unpack_variant_semantic_or_none(parts.semantic_blob, semantic.state_dict())
if state is None:
    state = {r.schema.name: torch.from_numpy(np.ascontiguousarray(r.values, dtype=np.float32))
             for r in decode_wans1(parts.semantic_blob)}
semantic.load_state_dict(state, strict=True)

tokens = np.fromfile(TOKENS, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)

OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "tokens": torch.from_numpy(tokens.copy()),
    "renderer": {k: v.clone() for k, v in semantic.state_dict().items()},
    "basis_raw": basis_raw.float().clone(),
    "coefficients": coefficients.float().clone(),
    "selector": bytes(selector or b""),
    "source_archive_sha256": hashlib.sha256((SUB / "archive.zip").read_bytes()).hexdigest(),
}, OUT)
n_params = sum(v.numel() for v in semantic.state_dict().values())
print(f"tokens {tokens.shape} classes {np.bincount(tokens.ravel(), minlength=5).tolist()}")
print(f"renderer {n_params} params, basis {tuple(basis_raw.shape)}, coefficients {tuple(coefficients.shape)}, "
      f"selector {len(selector or b'')} B")
print("saved", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
