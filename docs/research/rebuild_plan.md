# Rebuild plan: joint_train135 state -> real #135-format archive.zip

Abbrev: S = `work/pr135/submissions/semantic-pose-HPAC_CPR1_polished` (copy it to our own submission dir before editing), EB = `work/research/repos/CommaVideoCompressionChallenge_ExperimentBook/src/cpr1_sub4`, FS = `work/research/repos/comma-ai-semantic-pose-hpac-cpr1/code`.
Inputs: `joint_run/tokens_joint.u8`, `joint_run/coef_joint.npy`, `joint_run/joint_state.pt`.
Proposed new script: `work/build_archive135.py` (one file, steps A-F below). Read-only audit so far; nothing below has been run.

## 0. Hard-coded decoder constants that WILL change (code edits in our submission copy, free)

| file:line | constant | why it changes |
|---|---|---|
| `S/runtime/residual_archive.py:39` | `IHS2_BYTES = 16_599` | IHS2 v3 with tightened rows + packed biases is variable length; new weights -> new length. The F24S parser slices the HPAC by this constant (no length field). |
| `S/runtime/residual_archive.py:41` | `WANS_BODY_BYTES = 36_040` | rANS streams change length with new int4 codes. Same slicing issue. |
| `S/inflate.py:14-15` | `ARCHIVE_SHA256`, `ARCHIVE_BYTES` | `_verify_input` refuses any other archive. |
| `S/verify_submission.py:15` | `EXPECTED_ARCHIVE_BYTES` | cosmetic, keep consistent. |
| `S/cpr1/inflate.py:29` | `CARRIER_AMPLITUDE = 64.0` | trained amp. |
| `S/cpr1/inflate.py:331` | `127.5 + CARRIER_AMPLITUDE * carrier` | trained gray: replace 127.5 by a `CARRIER_GRAY` constant. |
| `S/runtime/residual_archive.py:38` | `IHS2_HEADER = b"IHS2\x03\x31"` | only if we change IHS2 flags (optional, see A5). |

CAP1 body, selector, table, token stream are self-delimiting or fixed-size: no constant edits needed.

## A. HPAC -> IHS2 v3 blob (16,593-ish B body)

Decoder path: `S/runtime/residual_archive.py::_decode_models` takes `IHS2_HEADER + models[4:4+IHS2_BODY_BYTES]` -> `S/runtime/ihs2.py::materialize_ihs1` -> `S/runtime/ihs2_gate_a.py::decode_v3` -> IHS1 bytes -> `S/cpr1/integer_model_io.py::_deserialize_self_compressed` (called from `S/cpr1/inflate.py::load_hpac`).

Header flags `0x31` = frame format 1 (`FRAME_E0L0`, int4 nibbles, frame-major) | `FLAG_TIGHT_ROWS` (0x10) | `FLAG_PACK_BIASES` (0x20); exponents NOT 3-bit packed (raw int8).

Steps:
1. Build the shell exactly like `S/cpr1/inflate.py::load_hpac` (IntegerHPAC(N, 5, patch 64, delta 2, ch 64, frame_dim 8, norm none, relu, use_frame_scale, weight_bound 127, activation_bound 127, use_weight_scales, exponent_min -6, use_spm, no norm gates)), `load_state_dict(joint_state["hpac"])`.
2. Write IHS1 ourselves (no encoder for a non-self-compressed model exists). Template: `FS/pack_hpac_self_compress.py::serialize_self_compressed`, but it needs `bit_depth` params which our model lacks. Variant:
   - rows: for each IntegerConv2d/IntegerLinear in `model.modules()` order, `w = module.codes()[0]` (= round(clamp(w,-127,127))*mask); rows = masked entries per output channel (`_weight_rows` in `S/cpr1/integer_model_io.py`).
   - depths: `S/runtime/ihs2.py::minimum_signed_depth(row)` per row (517 rows; 0 for all-zero rows). Weight bound 127 -> depth <= 8.
   - tail, in `model.named_parameters()` order skipping compressed weights: bias -> round, clamp int16, `<i2`; exponent -> round, clamp [-6,0], `i1`; `frame_embed.weight` -> round, `i1`.
   - IHS1 = `b"IHS1" + pack_nibbles(depths) + bitpacked rows (LSB-first, `np.packbits(bitorder="little")`) + tail`.
   - Cross-check with `EB/ihs2.py::layout_from_runtime` + `parse_ihs1` (must parse with zero trailing bytes).
3. IHS2: `EB/ihs2_gate_a.py::encode_v3(ihs1, layout, frame_format=1, pack_exponents=False, tighten_rows=True, pack_biases=True)` (or `EB/ihs2.py::encode_ihs2_v3`). Original depths = the tight depths (allowed: decode only requires stored <= original; the duplicate 259 B nibble array LZMA-compresses to almost nothing).
4. Constraints to assert BEFORE encoding (training does not enforce them):
   - **frame_embed must be in [-8, 7] after rounding** (E0L0 int4). joint_train135 only clamps to +-127 (`frame_codes`). If violated: clamp and re-gate, or switch to FRAME_RAW (+2,400 B pre-LZMA, needs header edit). Add a clamp to the trainer for future runs.
   - exponents in [-6,0] (guaranteed by `codes()` clamp), biases fit signed 15 bits (|b| <= 16383, `_pack_biases` width <= 15), weights |w| <= 127.
5. Optional free bytes (decoder header edit): `pack_exponents=True` (517 B -> 194 B raw) and/or drop `tighten_rows` with original=tight depths (-259 B raw). Both pre-LZMA, measure after compression.
6. Gate A0: run step 2-3 on #135's own model (load from `RA.materialize_ihs1(parts.hpac_blob)`) -> blob must equal `parts.hpac_blob` byte-for-byte if we reuse #135's original depths (with original=tight it will differ only in the first depth array; then gate on `decode_v3` -> load -> identical state_dict instead).

## B. Boundary table -> RCF1 int6 (96 B in p)

Location: `p = rawLZMA2(models) + table(96 B) + rc64 stream`. Table = fp16 scale (2 B, `<f2`) + `pack_signed(codes, 6)` of 125 codes (94 B), magic `RCF1` elided. Decoder `S/runtime/residual_archive.py::_decode_fixed_table`: `values = codes.astype(f32) * float(fp16 scale)`, codes in [-31,31].
- Encoder: `EB/residual_calibration.py::quantize_table("boundary_predicted", values, 6)` (scale = fp16(max|v|/31), codes = clip(rint(v/scale), +-31)), then `EB/residual_archive.py::serialize_fixed_boundary_int6(table)[4:]`.
- **Gap:** joint_train135 trains and gates the table as unconstrained float (`exact_state` uses it unquantised). The quantised table changes token bytes slightly. Since corrected logits are re-rounded to 1/8 before softmax, try fp16 scales {max|v|/31, 0.125, 0.25, 0.0625} and keep the one with the lowest exact token bytes (cheap: rerun the encoder, or the trainer's `frame_bits` with the quantised table). Long-term: STE-quantise the table inside training.
- Use `_decode_fixed_table(b"RCF1"+body).values` as the table passed to the token encoder so encode and decode use bit-identical floats.

## C. Renderer (WANS1 -> F12 body) and carrier (CPR1 -> CAP1), selector kept

### C1. Renderer
- Records: start from `decode_wans1(parts.semantic_blob)` (#135's records, carries `schema`, `raw_scales`). For each w4 record: `codes = clamp(round(joint_state["render_codes"][name]), -7, 7).astype(int8)` (torch.round = half-even = training's ste_round); keep `raw_scales`/`scales` unchanged (not trained); `values = codes * scales` (reshape as in decode_wans1). For each fp16 record: `raw_fp16 = np.asarray(master, dtype="<f2").tobytes()` (matches training's `v.half()`); values = that fp16 as f32. Build `EB/baseline.py::TensorStorage(schema, "w4"/"fp16", values, scales, codes, raw_fp16=..., raw_scales=...)`.
- **Blocker for stock encode_wans1:** F12 elides the WANS header; the decoder reinserts the fixed `F11_FIXED_PREFIX = b"WANS\x01\xB7\xFD\x00\x00\x00\x00"` (`S/runtime/entropy/renderer_weight_codec.py`), i.e. the raw/rANS mode mask (0xFDB7) and all priors = 0 are frozen. `EB/entropy/renderer_weight_codec.py::encode_wans1(strategy="per_tensor")` picks modes/priors per tensor and `encode_f12_wans_body` raises if they differ. Write `encode_wans1_fixed(records)`: same as encode_wans1 but force mode_i = bit i of 0xFDB7 and prior 0: rANS streams via `EB/entropy/adaptive_ans.py::encode_adaptive(codes+8, prior_index=0)`, raw streams via `EB/bits.py::pack_signed(codes, 4)`. Then `EB/entropy/renderer_weight_codec.py::encode_f12_wans_body(blob, WANS_STREAM_ORDER=(1,15,4,0,11,5,2,9,3,6,10,12,14,7,8,13))`. Its length becomes the new `WANS_BODY_BYTES`.
- Constraints: codes never -8; every stream non-empty; total stream area < 65536.
- Do NOT use `EB/residual_archive.py::_models`/`build_residual_archive_bytes` for F24S: they hard-assert `len(hpac)==16,599`, `len(semantic_body)==F13_WANS_BODY_BYTES` and `encode_legacy_w4(decode_wans1(semantic)) == payload.semantic_blob` (i.e. the renderer is unchanged). Assemble by hand (step D).

### C2. Carrier
- Get #135's canonical CPR1: `S/runtime/carrier_repack.py::split_frame0_selector_carrier(parts.carrier_blob)` -> (cap1, selector); `materialize_cpr1(cap1, R)`; fields via `EB/carrier_repack.py::_parse_cpr1` (keep `scales` (96 B: 12 f4 basis scales + 12 f4 coefficient scales) unchanged).
- Coefficient codes: `k = rint(coef_joint / step)` with step = CPR1 coefficient scales (`EB/carrier_repack.py::cpr1_coefficient_scales`). Better: change the trainer to also save `round(codes)` as int16. **Assert all k in [-2048, 2047]** (the trainer never clamps; the int12 cumsum wraps mod 4096 silently).
- Basis codes: `b = clamp(round(joint_state["basis_codes"]), -15, 15)` flattened in (12,3,24,32) order.
- CPR1 bytes (canonical, since CAP1 decode re-derives CPR1 and checks it): header `struct.pack("<4sII", b"CPR1", basis_bits, coef_bits)` + scales + Huffman lengths (32 B) + Rice ks (12 B) + basis payload + coefficient payload, where
  - basis: `FS/carrier_codec.py::_encode_huffman(zigzag(b))` (MSB-first bits, canonical codes, max len 31, >=2 symbols); or the whole thing via `FS/carrier_codec.py::encode_compact_carrier(basis_scales, b, coef_scales, zigzag-delta codes)`;
  - coefficients: `EB/carrier_repack.py::_rice_encode(_zigzag(_predict_residuals(k, zeros(12), 0)), 1)` (guarantees the same k choice CAP1's decoder recomputes).
- CAP1: `EB/entropy/coefficient_ar1_codec.py::encode_cap1(cpr1, frames=600, dimensions=12)` (refits AR(1)+bias, self-checks decode == cpr1).
- Selector: keep #135's sparse F0E1 selector bytes unchanged (training applied it as-is).
- Gate C0: run C1+C2 on #135's own codes -> CAP1 and F12 body must equal #135's bytes.

## D. Assemble p and the zip

```
models = b"F24S" + ihs2[6:] + f12_wans_body
       + cap1[14:20]                                  # 6 B: u24 basis_bits, u24 residual_bits
       + scales(96) + predictor(36) + lengths(32) + ks(12) + basis + rice   # STORED_CAP_FIELDS order
       + selector[5:]                                 # strip b"F0E1\x01"
comp   = lzma.compress(models, format=lzma.FORMAT_RAW, filters=S/runtime/residual_archive.py::LZMA_FILTERS)
p      = comp + table_body(96) + rc64_stream
```
CAP1 blob layout = `CAP1 01 00 00 00` (8) + counts (6) + predictor (36) + scales (96) + lengths (32) + ks (12) + basis + rice; stored order swaps scales before predictor (`STORED_CAP_FIELDS`).
Token stream: modify `work/encode135.py` to take `--hpac-blob` (our IHS2 -> `RA.materialize_ihs1` -> `R.load_hpac`) and `--table` (our 96 B body -> `_decode_fixed_table().values`) instead of `parts.*`, and to write a full new p (not splice). Same loop otherwise (it already mirrors `decode_production_tokens`).
Zip: `EB/residual_archive.py::_zip_bytes(p)` or the block in encode135.py: one member `p`, ZIP_STORED, date (1980,1,1,0,0,0), create_system 3, external_attr 0o100644<<16, allowZip64=False (100 B overhead).
Gate D0: rebuild #135 from its own decoded state -> archive sha must be `12cf5d71...` (if liblzma output differs, accept identical decode instead; only size matters).
Then set the section 0 constants and copy the archive.zip into the submission dir (`S/inflate.py` reads `here/archive.zip`, and checks it matches `data_dir/p`).

## E. Gray / amplitude

The challenge rules (`D:/projects/comma_b/README.md` "rules") charge only archive.zip; code is free unless it carries large artifacts (neural nets etc.). Two scalars are fine (#135 already hard-codes 127.5, 64, schemas, the WANS prefix).
Edit `cpr1/inflate.py:29` to the trained amp and line 331 `127.5` -> trained gray. Write them as float32-exact literals (`float(np.float32(x))`, e.g. via `float.hex`) so the decoder's python-scalar-times-f32 matches the trainer's f32 tensors. Not guaranteed bit-exact anyway: render_video does the einsum in batches of 64 frames, training in 4 (cuBLAS may differ in the last ulp; matters only at .5 rounding boundaries). The real check is E2E below.

## F. Verification checklist (run with our edited decoder copy)

1. `read_residual_archive(new_zip)` parses; `len(compressed)+96+len(stream)+100 == zip size`.
2. HPAC: `R.load_hpac(materialize_ihs1(parts.hpac_blob))` state_dict == round/clamped trained state (every tensor exact; frame_embed in int4 range).
3. Table: `parts.table.codes/scale` == our quantised table.
4. Renderer: `decode_wans1(parts.semantic_blob)` codes == clamp(round(render_codes)); fp16 raw bytes equal.
5. Carrier: `decode_compact_carrier(materialize_cpr1(...))` -> basis codes == b, cumsum-decoded coefficient codes == k, scales unchanged; selector bytes == #135's.
6. Tokens: `decode_production_tokens(parts, R, S/cpr1, cuda)` -> `decoded_token_sha256` == sha of tokens_joint.u8, and `decoder_bit_position` consumed the whole stream.
7. E2E: `inflate.sh` into a fresh dir, then `evaluate.sh --submission-dir <ours> --device cuda`; compare seg/pose with joint_train135's last gate (differences = table quantisation + float batch effects) and the size with `best.json["archive_bytes"]` (the trainer's estimate ignores HPAC/WANS/LZMA size changes).
8. Runtime: the decode is unchanged in cost (same model), stays under the 30 min T4 limit.

## Blockers / missing pieces
- No IHS1 writer for a plain (non-self-compressed) IntegerHPAC: ~30 lines (A2).
- encode_wans1 cannot produce the frozen F11 mode/prior prefix: needs the forced-mode variant (C1).
- EB's F24S writer asserts unchanged HPAC/renderer: assemble p by hand (D).
- Decoder constants `IHS2_BYTES`, `WANS_BODY_BYTES`, `ARCHIVE_SHA256/BYTES` must be edited.
- Trainer does not enforce: frame_embed int4 range, coefficient int12 range, int6 table quantisation. Check all three on the saved state first; if violated, clamp and re-gate before building.
