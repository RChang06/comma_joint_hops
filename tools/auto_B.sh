#!/bin/bash
# unattended: wait for B hop 2, then chain stage B hops (renderer move on the full score -> near-pool re-search ->
# quick carrier refit) while each hop beats its start by > 1e-5; then a thorough pose stage on the best state.
# everything stays on the box; progress in auto_B.log
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f B2_done.flag ]; do sleep 30; done
cp B1_pose.log B2_pose.log
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
better() { python -c "import sys;sys.exit(0 if $1 < $2 - 1e-5 else 1)"; }
S=B1_pose
if better $(score B2_pose) $(score B1_pose); then S=B2_pose; fi
echo "$(date +%H:%M) after B2: best $S $(score $S)"
for n in 3 4 5 6; do
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 300 --window 8 --gate 300 \
    --lr-map 0 --w-pose 1 --w-rate 1 --render-bytes wans --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 \
    --lr-render-scale 1e-4 --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
    --save-final --out B${n}_move > B${n}_move.log 2>&1
  python exp2_train.py --tokens B${n}_move/unaccepted/tokens_joint.u8 --state B${n}_move/unaccepted/joint_state.pt \
    --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans \
    --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out B${n}_search > B${n}_search.log 2>&1
  python exp2_train.py --tokens B${n}_search/tokens_joint.u8 --state B${n}_search/joint_state.pt --steps 100 --window 8 \
    --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
    --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out B${n}_pose > B${n}_pose.log 2>&1
  echo "$(date +%H:%M) hop B$n: $(score B${n}_pose) vs start $(score $S)"
  if better $(score B${n}_pose) $(score $S); then S=B${n}_pose; else echo "hop B$n did not pay, stop hops"; break; fi
done
echo "$(date +%H:%M) final pose stage from $S"
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out final_pose > final_pose.log 2>&1
echo "$(date +%H:%M) final pose: $(score final_pose) (from $S $(score $S))"
tar czf /root/auto_B.tgz auto_B.log B*_move.log B*_search.log B*_pose.log B*_search B*_pose final_pose final_pose.log
echo DONE > auto_B_done.flag
