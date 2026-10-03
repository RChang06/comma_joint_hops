# #141 renderer storage: SM3R / SD1M + LAY1

Decoder: `submissions/semantic_blocks/cpr1/ddm_mp2_semantic_receiver.py::unpack_variant_semantic_or_none`, called
from `runtime/f26_inflate.py` (line ~482) before the WANS1 fallback. Grouping: `compress.py::segment_semantic` +
`runtime/segment_layout.py` (LAY1). Writer + gate: `work/sm3r_writer.py`.

## Template

`SemanticTokenRenderer(96).state_dict()` order = `runtime/baseline.py::SEMANTIC_SCHEMA` order (same 38 tensors
as WANS1). Tensors with ndim < 2 (biases, norm) are raw fp16. The 16 tensors with ndim >= 2 are quantized:
per-row fp16 scales (per column for `*embed.weight`, i.e. 96 scales for token_embed, 8 for frame_embed) and
signed codes, value = float32(code) * float32(fp16 scale). Same arithmetic as WANS1 `decode_wans1`, so equal
codes + equal raw scales give bit-identical float32 weights.

## Bit packing

`_unpack_signed_bits`: codes of `bits` width (2..8), two's complement, value bits LSB first, concatenated and
packed into bytes little-endian bit order (`np.unpackbits(bitorder="little")`), zero padded to a byte.
Unlike WANS1 there is no reserved -8: the full range [-2^(b-1), 2^(b-1)-1] is legal.

## Formats

- `SD1M`: `"SD1M" | u8 version 1 | u8 count 16 | depth nibbles (8 B, low nibble first) | per tensor in template
  order: fp16 raw, or scales + codes at its depth`. No pruning. Requires template order == SD1M_V1_NAMES.
- `SM3R` mode 5: `"SM3R" | u8 1 | u8 5 | u8 keep_percent (1..99) | u8 0 | u16 selection mask` then tensors,
  all depth 4.
- `SM3R` mode 6 (what #141 ships): mode 5 header + depth nibbles (8 B) after the selection mask.
- Selection mask must equal the bits of `ROW_PRUNE_NAMES` = {blocks.1.film, blocks.2.film, blocks.3.film}
  (fixed in the decoder, mask 0x4900). For each of those: a row bitmap (192 rows -> 24 B), then
  kept = max(1, round(192 * keep/100)) row scales, then kept*8 codes. Unselected rows decode to exact 0.
  keep < 100 is enforced, so **at least 2 rows of each of the three film tensors are always zeroed**
  (keep 99 -> 190 rows).

## #141's own blob (36,130 B, LAY1 36,202 B)

mode 6, keep 1% -> only 2 of 192 rows kept in blocks.1/2/3.film (190 zero rows each). Depths: frame_embed 3,
blocks.0.film 3, everything else 4. That pruning and the two 3-bit tensors are where its byte advantage comes
from, not the container: the raw-nibble code stream is less compact than WANS1's adaptive rANS.

## LAY1 grouping (segment_semantic)

Segments alternate metadata/codes: metadata = header (+depth table) + all fp16 tensors and scales (+ row
bitmap) since the previous code span; codes = one tensor's code bytes. LAY1 = "LAY1" | u16 n | u16 sizes[n] |
all even (metadata) segments | all odd (code) segments. So all fp16 metadata is contiguous first (good for a
transpose:2 block), then all code bytes. Overhead 6 + 2n B (n = 33 -> 72 B).

## Gate (work/sm3r_writer.py)

#141's blob: structural parse -> re-encode identical (36,130 B); codes re-derived from the decoded float
weights (round(w/scale)) -> identical; LAY1 of the re-encode == #141's renderer section (36,202 B). PASS.

## Our renderer

All 16 quantized tensors use codes -7..7 (4 bits needed), and no film row is all zero. So:
- SM3R can not hold it exactly (it must zero >= 2 rows in each of blocks.1-3.film).
- SD1M (depth 4 everywhere) holds it bit-exactly, but costs bytes vs the WANS body.

Archive sizes with `work/build_archive141_sm3r.py` (final_B pieces: same stream, carrier, hpac, table), all in
`work/final_sm3r/`:

| renderer storage | blk2 search | archive B | vs 181,176 | renderer |
|---|---|---|---|---|
| wans (1-seg lay1), control | default | 181,176 | 0 (sha 478be533, == final_B) | exact |
| sd1m depth 4 | default | 181,432 | +256 | exact |
| sd1m depth 4 | wide | 181,328 | +152 | exact |
| sm3r6 keep 99 (6 film rows zeroed) | default | 181,440 | +264 | lossy: 48 weights, max abs 0.0211 (blocks.1.film) |
| wans (1-seg lay1) | wide | **181,031** | **-145** | exact |

Conclusion: the tagged formats lose to WANS1 for our full renderer (adaptive rANS on codes beats brotli on
raw nibbles by ~150-250 B). #141's advantage is its pruned renderer content. The only exact win found is the
wider BLK2 search (brotli quality 10/11, mode 0/1, more lgwin values, extra cuts) over the old WANS storage:
one brotli block with transpose 2 over the whole models buffer.
