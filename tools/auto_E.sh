#!/bin/bash
# after the wide pass: stage B hop chain (renderer move on the full score -> near-pool re-search -> quick refit) while
# each hop beats its start by > 1e-5, max 4 hops, then a 600-step pose stage. folders E{n}_*, final E_pose.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_D_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
better() { python -c "import sys;sys.exit(0 if $1 < $2 - 1e-5 else 1)"; }
S=C_s4
if [ -f D_pose/best.json ] && better $(score D_pose) $(score C_s4); then S=D_pose; fi
echo "$(date +%H:%M) start $S $(score $S)"
for n in 1 2 3 4; do
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 300 --window 8 --gate 300 \
    --lr-map 0 --w-pose 1 --w-rate 1 --render-bytes wans --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 \
    --lr-render-scale 1e-4 --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
    --save-final --out E${n}_move > E${n}_move.log 2>&1
  python exp2_train.py --tokens E${n}_move/unaccepted/tokens_joint.u8 --state E${n}_move/unaccepted/joint_state.pt \
    --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans \
    --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out E${n}_search > E${n}_search.log 2>&1
  python exp2_train.py --tokens E${n}_search/tokens_joint.u8 --state E${n}_search/joint_state.pt --steps 100 --window 8 \
    --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
    --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out E${n}_pose > E${n}_pose.log 2>&1
  echo "$(date +%H:%M) hop E$n: $(score E${n}_pose) vs start $(score $S)"
  tar czf /root/E${n}.tgz E${n}_pose E${n}_*.log
  if better $(score E${n}_pose) $(score $S); then S=E${n}_pose; else echo "hop E$n did not pay, stop"; break; fi
done
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out E_pose > E_pose.log 2>&1
echo "$(date +%H:%M) final pose: $(score E_pose) (from $S $(score $S))"
tar czf /root/auto_E.tgz auto_E.log E*_pose E*.log
echo DONE > auto_E_done.flag
