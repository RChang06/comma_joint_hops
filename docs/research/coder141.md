# #141 token coder: how the class map is stored and decoded

Source: `submissions/semantic_blocks/` (PR #141 v3 = PR #140 content + BLK2/LAY1 repack). All paths below are
relative to that dir unless they start with `work/`.

## 1. Container (archive.zip -> member p, 179,791 B)

Dumped by `work/inspect141.py`.

```
p = "BLK2" | u32 models_len (71,739) | u8 block_count (4)
    | block0: codec 0 (raw)                 13,288 B
    | block1: codec 2 (brotli) + transpose:2 5,914 B stored
    | block2: codec 2 (brotli)              25,239 B stored
    | block3: codec 0 (raw)                 21,816 B
    | tail (113,507 B) = RCF1 table body (96 B) + RC64 token stream (113,411 B)
```

- Blocks are just byte spans of the concatenated `models` buffer (`runtime/block_container.py`); boundaries,
  codecs and the transpose stride are in the archive. `compress.py` + `recipe.json` rebuild them.
- `models` = RX1 header `<4sBBBBHHH>` = (`RX1M`, v1, codec 2, table_mode 0, reserved 0x18, 13515, 36202, 22008)
  then three sections:
  - HPAC: 13,515 B Brotli stream of an **IHS1** blob (17,952 B). Not IHS2 like #135.
  - renderer: 36,202 B **LAY1**-grouped (`runtime/segment_layout.py`) **SM3R** blob (36,130 B, row-pruned
    mixed precision, `cpr1/ddm_mp2_semantic_receiver.py`). Stored uncompressed inside models (block codecs
    compress it). Not WANS/F12 like #135.
  - carrier: 22,008 B body with reserved bits 0x08 (RR5 adaptive-arith basis) and 0x10 (DX2 CABAC
    coefficients) applied on top of packed CAP1 metadata, then the sparse F0E1 selector body (no compensation).
    Restored to `F0C1` blob of 22,316 B.
- In block mode `read_residual_archive` requires reserved bits 0..2 = 0 (no SZ1/CK2); RR5/DX2 optional.
- Tail: `section[:96]` = RCF1 body (fp16 scale + 125 int6 codes, magic elided) -> `parts.table`;
  `section[96:]` = token stream, **no length field, runs to the end of p**. Starts at p offset 66,380.

## 2. Token decode (runtime/residual_archive.py::decode_production_tokens)

Per frame f = 0..599, per group g = 0..189 (`cpr1/inflate.py::group_masks`, group(x,y) = (x&63) + 2(y&63),
same 64x64 tile wavefront as #135):

1. `context = model.prepare_frame_context(f, previous_frame)` (IntegerHPAC from IHS1, CUDA,
   `configure_cuda_reproducibility`, `optimize_sparse_evaluator`).
2. `boundary` = distance-to-edge bucket (0..4) of the previous decoded frame (`_boundary_buckets`), all 4 at f=0.
3. `base = sparse.selected_logits(current, context, g)`; `predicted = argmax(base)`.
4. `corrected = base + table.values[boundary*5 + predicted]` (RCF1 25x5 table, stored).
5. `probability = _probability_table(corrected, 8)` (round logits to 1/8, float64 softmax, float32). Up to
   here identical to #135.
6. NEW in #140/#141: the adaptive **free corrector** (`corrector.group_state(probability, predicted, positions)`
   then `row = corrector.coding_row(state)`) rewrites the row; RC64 decodes the group with `row`;
   `corrector.observe(state, symbols)`; after the frame `corrector.end_frame(tokens)`, and `begin_frame(boundary)`
   at the start of each frame.

The corrector is selected by `_rr8_select_corrector`: the C port (`runtime/f26_corrector_native.c` via
`native_free_corrector.py`, built by inflate.sh with `-ffp-contract=off -fno-fast-math`) when
`F26_CORRECTOR_NATIVE_LIBRARY` is set, else the python `free_corrector.FreeCorrector`. Both are meant to be
bit-identical; the C binding refuses if the python SHIPPED_CONFIG drifts.

### corrector stack (python class chain, all in runtime/)

`free_corrector.FreeCorrector` = `Ma1WithinMissCorrector` < `fx2.Fx2ModelAxisMixer` < `fx1.FixedPointLogisticMixer`
< `rr4.FreeCorrector`.

- rr4 (hit-event law): per position context = (predicted class, surprise bin ubin of 1-p_max on a 2^(-k/2)
  ladder, agree with prev1/prev2 frames, unchanged-run length 0..7, boundary bucket) = 51,200 cells. Per cell
  counts n, hits h, fixed-point sum of p_max. Odds multiplier m = KT ratio (observed hit odds / expected hit
  odds), only when n >= 32, clamped to [1/16, 16]. Coded row: argmax prob q = p m / (p m + 1-p), other
  columns scaled by (1-q)/(1-p).
- fx1/fx2 (logistic mixer): 23 count-table "members" (shipped_joint, temporal/spatial/surprise/boundary/run
  variants, fast-forgetting copies with halving at 256/4096, spatial4/homogeneity from 4 causal neighbours
  left/up/up-right/up-left, decode-step groupbin8, 32x32 patch, 64x64 tile). Their multipliers are blended as a
  weighted geometric mean `prod m_k^(w_k)` with fixed-point weights (20 frac bits, applied on a 1/64 grid via
  successive sqrt), weight set selected by mixer context cls x boundary x agree x homog x ubin8 (4,000 sets).
  Weights start at 1.0 for shipped_joint and 0 for the rest and are learned online by an integer gradient
  step on the hit log-loss (lr 2^-4). SSE stage off.
- ma1 (within-miss law): reweights the non-argmax columns by KT ratios M[cell, k] = (n+0.5)/(e+0.5),
  cell = (up, up-right, left neighbour classes, prev1 class) with UNKNOWN level (1,296 cells), min count 1,
  clamp [1/16, 16], mass-preserving.
- All state (count tables, mixer weights, prev1/prev2/run planes, current/known planes) starts at zero/identity
  for every decode and is updated only from already-decoded symbols in `observe` / `end_frame`. Only `+ - * /`,
  sqrt, comparisons: no libm transcendental on the decision path.

**Stored parameters for the token model: only the HPAC (IHS1) and the 96 B RCF1 table.** The correctors carry
zero archive bytes; their config (SHIPPED_CONFIG, constants) is frozen in source and compiled into the C.

### RC64

`runtime/entropy/rc64_backend.c` and `rc64.py` are byte-identical to #135's (they differ only in CRLF line
endings). Frequency quantization: floor(p * 2^31), min 1, balance to first argmax. So codexblack's encoder
(`ExperimentBook/src/cpr1_sub4/entropy/rc64.py` NativeEncoder + its rc64_backend.c, `quantize_probabilities`
does the same floor/argmax balance) is byte compatible, as it already was for #135.

## 3. Encoder (work/encode141.py)

Runs the same loop with known symbols: identical setup objects (`_load_renderer`, `load_hpac`,
`_sparse_class`, `_rr8_select_corrector`, `RA._probability_table`, `RA._boundary_buckets`), passes the
corrector's `coding_row` to `NativeEncoder.encode`, then `observe(state, known_symbols)`. Optional `--lockstep`
runs #141's own RC64 decoder on the stored stream with the same rows and reports the first mismatch.
Splice: `new_p = old_p[:-len(old_stream)] + new_stream`, zip with compress.py's settings (stored, 1980-01-01,
create_system 3, external_attr 0o644<<16).

## 4. Gate

PASS (2026-10-01, laptop T2000, native C corrector, full 600 frames, 613 s, ~1.0 s/frame):
stream 113,411 B identical to the stored one, lockstep decode never diverged (decoder bit position 907,349, same
as the local decode receipt), cdf-input sha 370a5e2a... equals the receipt, and the re-zipped archive is
byte-identical to #141's (sha 0e2d95c2...). Log: work/encode141_gate.log. The python corrector
(`--python-corrector`) also stays in lockstep over 5 frames (~3 s/frame).

## 5. Shipping a different map in #141's container

- Only the token stream changes if HPAC/table/renderer/carrier are kept: splice, exactly like encode135.
  Then update `inflate.py` ARCHIVE_SHA256 / ARCHIVE_BYTES (the only hard checks). No length field anywhere
  needs editing (the stream is the tail; block headers only cover models).
- Map-dependent stored parameters:
  - HPAC (IHS1): **differs from #135's** (same architecture, every tensor retrained; IHS1 17,952 B vs 20,179 B).
    It was trained on #140's (edited) map, so a new map would ideally get a refit/fine-tuned HPAC.
  - RCF1 table: also differs from #135's (scale 0.0496 vs 0.0751). Fitted offline; cheap to refit for a new map
    (same 96 B body format as #135, `work/hpac_writer135.py::encode_table` applies unchanged).
  - Correctors: nothing stored, nothing to refit; they adapt to whatever map is coded.
  - Carrier coefficients + frame-0 selector are not token-coded but were solved against renders of #141's
    map, so a materially different map may want a carrier re-solve (pose).
- Formats vs our #135 writers:
  - HPAC: #141 = Brotli(IHS1) in RX1 with a length field. Our hpac_writer135 emits IHS2 v3; we need its IHS1
    stage (IHS1 = magic + depth nibbles + bitpacked rows + tail), then brotli. IHS1 length is free (RX1 header).
  - Renderer: #141 = SM3R (row-pruned mixed precision) + LAY1 grouping; our wans_writer135 (F12 WANS) does
    not apply. A WANS1/SD1M/SM3R blob is accepted by f26_inflate; a plain WANS F12 body is only accepted on the
    non-tagged path with the fixed 36,040 B length. In BLK2 mode the renderer section is always passed through
    `unpack_segments`, so even a WANS body needs a (trivial) LAY1 wrapper. compress.py's `segment_semantic` does LAY1 grouping; an
    SM3R writer would have to be written as the inverse of `_decode_row_prune(_mixed)`.
  - Carrier: same CAP1 core as #135 (carrier_writer135 output), but stored as packed CAP1 metadata
    (inverse of `_restore_packed_cap1_metadata`: -40 B) with RR5 and DX2 riders on top. The encoders ship in
    the runtime: `rr5_arith_basis.apply_rider_to_carrier_body` and `dx2_cabac_coefficients.apply_cabac_to_carrier_body`.
    Packed CAP1 length is derived from its own bit counts (SA2), so a re-solved carrier parses.
  - Renderer constants: `cpr1/inflate.py` line 30 `CARRIER_AMPLITUDE = 64.0` and line 342
    `127.5 + CARRIER_AMPLITUDE * carrier` are the same gray/amp constants as #135 (cpr1/inflate.py 29-30/331).

## 6. Our c2 map in #141's container (2026-10-01)

Builder: `work/build_archive141.py` (defaults: renderer from `work/final_c2/sub_c2`, carrier
`work/cycle2/carrier_c2.bin`, stream `work/final_c2/stream141_c2map.bin`, out `work/final_c2/archive141_c2.zip`).

- Stream: `encode141.py` on `work/cycle2/c2_s2/tokens_joint.u8` (#141 HPAC + RCF1 table + correctors):
  110,039 B (685 s; log `work/encode141_c2map.log`, two identical copies of that run overlapped once).
- Renderer: sub_c2's F12 WANS body equals #135's original byte for byte (sha b0d41ec904ac); stored as a
  one-segment LAY1 (36,048 B). #141's reader decodes it to the same WANS1 blob / records.
- Carrier: carrier_c2.bin == sub_c2's carrier blob. Re-stored: CAP1 stored order -> packed metadata (-40 B,
  inverse of `_restore_packed_cap1_metadata`) -> dx2 `apply_cabac_to_carrier_body` -> rr5
  `apply_rider_to_carrier_body` (huffman table dropped), + selector body. 22,272 B blob -> 21,965 B section.
  Gate a: the same recipe applied to #141's own carrier reproduces its 22,008 B section byte for byte.
- RX1 header reserved 0x18 as in #141; hpac section (13,515 B brotli IHS1) and 96 B table copied verbatim.
- BLK2: small search over section-aligned cuts x {raw, brotli q11 lgwin 16-24, lzma2} x transpose {1,2,4};
  picked 2 blocks: hpac+renderer brotli transpose 47,789 B, carrier raw 21,965 B.
- Gate b (#141's `read_residual_archive` on the new archive): stream, table, hpac, carrier blob, renderer
  WANS1 + records all equal what went in; no compensation overlay.
- Archive 180,008 B (sha 8464c671...) vs 186,512 B for sub_c2: p = 9 (BLK2 hdr) + 69,773 blocks incl. headers
  ... see report; table 96, stream 110,039, zip overhead 100.
- Decoder copy `work/final_c2/sub141_c2/`: #141's tree minus archive/inflated, archive.zip replaced,
  ARCHIVE_SHA256/BYTES updated, cpr1/inflate.py CARRIER_AMPLITUDE = 63.97188949584961 and CARRIER_GRAY =
  127.44857025146484 in the carrier line; the parallel-render path in runtime/ddm_wc1_advisory_runtime.py had
  its own 127.5 and now uses renderer.CARRIER_GRAY too (only used when F26_ADVISORY_RENDER_WORKERS is set).
  .sh files already LF. Nothing in f26_inflate re-derives renderer/carrier from other data: renderer comes
  from the WANS1 blob (SM3R/SD1M path returns None), carrier from CAP1 + selector, compensation only if present.
