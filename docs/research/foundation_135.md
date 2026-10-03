# Foundation research for building on PR #135 (F26)

Status: in progress (written incrementally). Legend: [READ] = read in source; [INF] = my inference.

Repos cloned (read-only) to `D:\projects\comma_b\work\research\repos\`:
- `CommaVideoCompressionChallenge_ExperimentBook` (codexblack, HEAD f229b26 "Finalize experiments") = the "separate frozen experiment repository" referenced in #135's PR body. [READ: its README names F26 as final candidate, same SHA 12cf5d71...]
- `comma-ai-semantic-pose-hpac-cpr1` (fesalfayed, #130 recipe, HEAD 2f94596)

Abbrev: EB = ExperimentBook repo root; S135 = `D:\projects\comma_b\work\pr135\submissions\semantic-pose-HPAC_CPR1_polished`.

---

## 1. Encoder side for #135's format

### 1.1 RC64 encoder: EXISTS, reusable directly [READ]

- `EB/src/cpr1_sub4/entropy/rc64.py`
  - `quantize_probabilities(probabilities[N,5] float32) -> uint32[N,5]`: `floor(float64(p) * 2^31)`, `max(.,1)`, balance `2^31 - sum` added to `argmax` (first max). Rejects p<=0, rows with |sum-1|>2e-5.
  - `ReferenceEncoder` / `ReferenceDecoder` (pure Python, slow) with `encode(symbols, probabilities)` / `finish() -> bytes`.
  - `NativeEncoder(library_path)` / `NativeDecoder`, ctypes wrappers; `compile_backend(output, compiler="cc")` compiles `rc64_backend.c` with `-std=c11 -O3 -fPIC -shared`.
- `EB/src/cpr1_sub4/entropy/rc64_backend.c` (373 lines) contains BOTH encoder (`rc64_encoder_create/encode/finish/data/size`) and decoder. S135's `runtime/entropy/rc64_backend.c` (178 lines) is the decoder half only, plus the fused `rc64_decoder_decode_probabilities` (float32 -> double -> `(uint64)(v*2^31)` floor, min 1, balance to first strict-max). Diffed: decoder logic identical except comments/includes.
- Coder: classic 63-bit interval arithmetic coder (`TOP=2^63-1, HALF=2^62, FIRST_QTR=2^61`), E3 pending-bit underflow handling, MSB-first bit packing, final flush `pending+=1; put(low<FIRST_QTR?0:1)`, last partial byte zero-padded. Decoder reads 63 bits initially, out-of-range bits read as 0. `width*cum >> 31` (encoder python uses `// TOTAL`, same).
- Overhead: 114,706 B actual vs 114,705.460 B ideal (0.54 B) per `EB/docs/F16_RC64.md`. So exact rate = sum -log2 p / 8 + <1 byte (+ the frequency floor rounding, negligible at 2^31).
- Self-test in `EB/scripts/audit_rc64_token_coder.py::_selftest` checks native == reference byte-identical.

### 1.2 Teacher-forced token-stream encoder: EXISTS but for the F15/F16 era inputs [READ]

`EB/scripts/audit_rc64_token_coder.py::_encode` (stage `encode`):
- loads frozen CPR1 runtime from `third_party/cpr1/code` (`_import_frozen_inflater`), tokens from `work/phase2/entropy_audit/tokens.uint8` (600x384x512 uint8), table from an RCL1 file, HPAC from `load_baseline(default_archive(root)).hpac_blob`.
- Per frame: `context = model.prepare_frame_context(idx, previous)`; boundary = `entropy_audit.boundary_buckets(previous)` (frame 0: all 4); per group (190 groups from `runtime.group_masks`): `selected = sparse.selected_logits(current, context, group)`, base = round(selected*8)/8, predicted = argmax(base), feature = boundary*5 + predicted, `corrected = base + table.values[feature]`, `probability = runtime.probability_table(corrected)` (re-rounds corrected*8 to int16, float64 softmax, cast float32), `encoder.encode(symbols, probability)`, then teacher-force `current[0,mask] = target[mask]`.
- HARD-CODED hash gates (TOKEN_SHA256, CORRECTED_LOGIT_SHA256, CDF_INPUT_SHA256, EXPECTED_NLL_BITS) will raise for a new map -> must be removed/parameterised for our use.
- `_decode` stage is an independent causal decode check.

=> To encode a NEW map for #135: copy this loop but take the HPAC blob and residual table from the #135 archive (`S135/runtime/residual_archive.py::read_residual_archive` gives `parts.hpac_blob` (IHS2, materialise with `runtime/ihs2.py::materialize_ihs1`), `parts.table`), and mirror S135's `decode_production_tokens` exactly (it uses `S135/runtime/residual_archive.py::_boundary_buckets` and `_probability_table`, numerically identical to EB's versions). Use EB `NativeEncoder` (compile EB's full `rc64_backend.c`). [INF: straightforward, ~80 lines.]

### 1.3 Archive layout and the writer [READ]

F24S `p` member layout (from S135 `read_residual_archive` + EB `build_residual_archive_bytes`):

```
p = rawLZMA2(models) + residual_table_body(96 B) + rc64_token_stream(rest of member)
models = b"F24S" + IHS2 body (16,593 B) + WANS F13 body (36,040 B) + stored CAP1 body (field order scales,predictor,lengths,ks,basis,rice) + F0E1 sparse selector body (prefix b"F0E1\x01" elided)
residual_table_body = fp16 scale (2 B) + 125 signed int6 codes bit-packed (94 B)   [RCF1 magic elided]
```
- The LZMA section end is found by `LZMADecompressor.unused_data` (filters: LZMA2 dict 64 KiB, lc0 lp1 pb0, nice_len 273, BT4, depth 0). The table is fixed size. The token stream is simply the remainder; no length field.
- ZIP: one member `p`, ZIP_STORED, `date_time=(1980,1,1,0,0,0)`, `create_system=3`, `external_attr=0o100644<<16`, no zip64 (`EB/src/cpr1_sub4/residual_archive.py::_zip_bytes`). Archive overhead = 100 B (186,724 - 186,624).
- Full writer: `EB/src/cpr1_sub4/residual_archive.py::build_residual_archive_bytes(payload, table, token_stream, schema="fixed_boundary_int6", fixed_wans_ar1_rc64_schema=True, model_compression="raw", lzma_filters=...)`. `_models(...)` builds the model section. Needs a `BaselinePayload` object (semantic/carrier/hpac blobs).

**Simplest exact route for a new map (renderer/HPAC/carrier/selector/table unchanged)** [INF, high confidence from the parser]: splice.
`new_p = old_p[: len(old_p) - 114_706] + new_rc64_stream`, then re-zip with `_zip_bytes` settings above. Nothing else references the stream length.
If the residual table is refit: replace the 96 bytes just before the stream (`old_p[-114706-96 : -114706]`) with `fp16(scale) + pack_signed(codes, 6)` (`S135/runtime/bits.py` / `EB/src/cpr1_sub4/bits.py::pack_signed`; `EB/src/cpr1_sub4/residual_archive.py::serialize_fixed_boundary_int6` gives the RCF1 form; strip the 4-byte magic).
If carrier coefficients are re-solved, the CAP1 body lives inside the LZMA section: decode models, rebuild CAP1 via EB `entropy/coefficient_ar1_codec.py` (encode side present in EB; S135 only has decode), re-concatenate, raw-LZMA2 recompress with the exact filters. EB's `build_residual_archive_bytes` does all of this.

### 1.4 How the 125-entry boundary table was fitted [READ]

`EB/docs/HPAC_RESIDUAL_CALIBRATION.md`, `EB/src/cpr1_sub4/residual_calibration.py`, `EB/scripts/evaluate_residual_calibration.py`:
- Feature `boundary_predicted` = boundary_bucket(previous frame, 0..4, saturating distance-to-class-edge, 4-neighbour dilation) * 5 + argmax(base logits). 25 states x 5 classes = 125 entries.
- Fit is ONE-SHOT, not iterative: accumulate `target[state,c]` = counts of true class, `expected[state,c]` = sum of frozen HPAC probabilities; `values = log((target+0.5)/(expected+0.5))`, then subtract row mean (`fit_log_ratio`, smoothing 0.5).
- `quantize_table(..., bits=6)`: `scale = max|v|/31` rounded to fp16; `codes = clip(rint(v/scale), -31, 31)`; deployed values = `codes*scale` (float32).
- Result (F15 era): int6 table 107 B (RCL1 form), stream 116,980 -> 116,828 B, net +45 B; int8 +30 B; predicted x margin int6 +16 B. Later the fixed RCF1 form costs 96 B in the p member.
- IMPORTANT [READ]: the corrected logit `base + codes*scale` is RE-ROUNDED to the 1/8 grid before softmax (`_probability_table` does `rint(corrected*8)` -> int16). So the effective table is quantised to 1/8-logit steps after addition; the log-ratio fit ignores this. [INF] A direct NLL-gradient fit of the 125 codes (with STE through the rounding) or a coordinate search over codes would beat the one-shot fit, cheaply, and should be redone after the map changes.

