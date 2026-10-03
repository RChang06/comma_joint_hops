# fast proxy for refitting #141's rcf1 boundary table: pre-corrector cost sum -log2 p[target] with
# p = _probability_table(base + table[boundary*5 + predicted]), on a frame subset of a map.
# dump: run the hpac in known-symbol mode (true previous frame, true earlier groups) and keep base logits,
#   cell and target per position. search: the cost splits over the 25 cells, so each cell's 5 codes are
#   fitted independently by coordinate search over the int6 range at a fixed fp16 scale.
# the correctors recalibrate hit odds per (predicted, boundary, ...) afterwards, so proxy gains are an upper
# bound on what a full encode will show; check the winner with encode141x.py --table-body.
import argparse, sys, os, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SUB = HERE.parent / "submissions/semantic_blocks"

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", default=str(HERE / "rj_results/final_pose/tokens_joint.u8"))
ap.add_argument("--hpac-ihs1", default=None)
ap.add_argument("--every", type=int, default=12, help="use frames 0, k, 2k, ...")
ap.add_argument("--dump", default=str(HERE / "hpac141/proxy_dump.npz"))
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--scale", type=float, default=None, help="fp16 table scale (default #141's)")
ap.add_argument("--eval-table", action="append", default=[], help="96 B bodies to score with the proxy")
ap.add_argument("--out-table", default=str(HERE / "hpac141/table_proxy.bin"))
args = ap.parse_args()

sys.path.insert(0, str(SUB))
from runtime import residual_archive as RA
parts = RA.read_residual_archive(SUB / "archive.zip")
NC = RA.NUM_CLASSES


def dump():
    import torch
    from runtime.f26_inflate import _load_renderer
    from runtime.ihs2 import materialize_ihs1
    from runtime.hpac_inference import configure_cuda_reproducibility, optimize_sparse_evaluator
    renderer_dir = SUB / "cpr1"
    R = _load_renderer(renderer_dir)
    dev = torch.device("cuda")
    configure_cuda_reproducibility()
    blob = Path(args.hpac_ihs1).read_bytes() if args.hpac_ihs1 else parts.hpac_blob
    model = R.load_hpac(materialize_ihs1(blob, R), dev)
    sparse = RA._sparse_class(renderer_dir)(model, R.EVAL_H, R.EVAL_W)
    plans = []
    for m in R.group_masks(dev):
        fpos = np.flatnonzero(m.detach().cpu().numpy().reshape(-1))
        plans.append((torch.from_numpy(fpos).to(dev), fpos))
    target = np.fromfile(args.tokens, dtype=np.uint8).reshape(R.N, R.EVAL_H, R.EVAL_W)
    frames = list(range(0, R.N, args.every))
    B, C, T = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        for f in frames:
            prev_np = target[f - 1] if f else None
            previous = (torch.from_numpy(prev_np.astype(np.int64))[None].to(dev) if f
                        else torch.zeros((1, R.EVAL_H, R.EVAL_W), dtype=torch.long, device=dev))
            context = model.prepare_frame_context(torch.tensor([f], device=dev), previous)
            boundary = (RA._boundary_buckets(prev_np).reshape(-1) if f
                        else np.full(R.EVAL_H * R.EVAL_W, 4, dtype=np.uint8))
            flat = target[f].reshape(-1)
            current = torch.zeros_like(previous)
            for g, (dpos, fpos) in enumerate(plans):
                base = sparse.selected_logits(current, context, g).cpu().numpy()
                pred = base.argmax(axis=1)
                B.append(base.astype(np.float32))
                C.append((boundary[fpos].astype(np.int64) * NC + pred).astype(np.uint8))
                T.append(flat[fpos])
                current.reshape(-1)[dpos] = torch.from_numpy(flat[fpos].astype(np.int64)).to(dev)
            print(f"dump frame {f}  {time.time() - t0:.0f}s", flush=True)
    np.savez(args.dump, base=np.concatenate(B), cell=np.concatenate(C), target=np.concatenate(T))


def cell_cost(base, tgt, offs):
    p = RA._probability_table(base + offs[None, :], 8)
    return -np.log2(np.maximum(p[np.arange(len(tgt)), tgt], 1e-300)).sum()


def table_body(codes, scale16):
    acc = 0
    for i, v in enumerate(codes.reshape(-1).astype(np.int64)):
        acc |= (int(v) & 0x3F) << (6 * i)
    body = scale16.tobytes() + acc.to_bytes(94, "little")
    t = RA._decode_fixed_table(RA.FIXED_MAGIC + body)
    assert np.array_equal(t.codes, codes.astype(np.int8))
    return body


def search():
    d = np.load(args.dump)
    base, cell, tgt = d["base"], d["cell"], d["target"].astype(np.int64)
    scale16 = np.asarray([args.scale if args.scale is not None else parts.table.scale], dtype="<f2")
    s = float(scale16[0])
    print(f"{len(tgt):,} positions; scale {s!r}")
    groups = [np.flatnonzero(cell == c) for c in range(25)]

    def total(codes, sc):
        return sum(cell_cost(base[i], tgt[i], codes[c] * sc) for c, i in enumerate(groups) if i.size)

    for path in args.eval_table:
        t = RA._decode_fixed_table(RA.FIXED_MAGIC + Path(path).read_bytes())
        print(f"proxy {path}: {total(t.codes.astype(np.float64), t.scale) / 8:,.0f} B")
    zero = total(np.zeros((25, NC)), s)
    cur = parts.table.codes.astype(np.int64).copy()
    if float(parts.table.scale) != s:
        cur = np.clip(np.rint(parts.table.values / s), -32, 31).astype(np.int64)
    start = total(cur, s)
    print(f"proxy zero table {zero / 8:,.0f} B, start (#141 codes) {start / 8:,.0f} B")
    for c, i in enumerate(groups):
        if not i.size:
            continue
        b, t = base[i], tgt[i]
        best = cell_cost(b, t, cur[c] * s)
        for r in range(args.rounds):
            moved = False
            for k in range(NC):
                for v in range(-32, 32):
                    if v == cur[c, k]:
                        continue
                    trial = cur[c].copy(); trial[k] = v
                    e = cell_cost(b, t, trial * s)
                    if e < best - 1e-6:
                        best, cur[c] = e, trial; moved = True
            if not moved:
                break
        print(f"cell {c:2d} n {i.size:>9,}  codes {cur[c].tolist()}  {best / 8:,.0f} B", flush=True)
    fin = total(cur, s)
    print(f"proxy refit {fin / 8:,.0f} B (start {start / 8:,.0f}, delta {(fin - start) / 8:+,.0f} B on "
          f"{len(tgt):,} positions)")
    body = table_body(cur, scale16)
    Path(args.out_table).write_bytes(body)
    print("wrote", args.out_table, body.hex())


if not Path(args.dump).exists():
    dump()
search()
