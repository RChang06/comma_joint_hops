# merge joint4b checkpoints that cover disjoint frame ranges into one, for a full-video exact evaluation.
# usage: python merge_ckpt.py OUT.pt  A.pt a0:a1  B.pt b0:b1 ...
# the FIRST checkpoint supplies the shared parts (renderer, basis); maps and coefficient offsets are
# concatenated in frame order, so the ranges must be contiguous and non-overlapping.
import sys
import torch

out = sys.argv[1]
pairs = [(sys.argv[i], tuple(map(int, sys.argv[i + 1].split(":")))) for i in range(2, len(sys.argv), 2)]
pairs.sort(key=lambda p: p[1][0])
cks = [(torch.load(p, map_location="cpu", weights_only=False), r) for p, r in pairs]
for (_, (a0, a1)), (_, (b0, b1)) in zip(cks, cks[1:]):
    assert a1 == b0, "ranges must be contiguous: %d != %d" % (a1, b0)
shared = torch.load(sys.argv[2], map_location="cpu", weights_only=False)      # first ARGUMENT, not first range
merged = {
    "acc_map": torch.cat([ck["acc_map"] for ck, _ in cks], 0),
    "coff": [c for ck, _ in cks for c in ck["coff"]],
    "ren": shared["ren"],
    "bcode": shared["bcode"],
}
lo, hi = cks[0][1][0], cks[-1][1][1]
assert merged["acc_map"].shape[0] == hi - lo == len(merged["coff"])
torch.save(merged, out)
print("merged frames %d:%d (shared parts from %s) -> %s" % (lo, hi, sys.argv[2], out))
