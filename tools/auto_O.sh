#!/bin/bash
# after auto_N: renderer hops from N_pose (pruned rows pinned, max 3, stop at first fail), then a 4-batch wide search and a
# pose refit on the best state. bundles /root/O_*.tgz
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_N_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
better() { python -c "import sys;sys.exit(0 if $1 < $2 - 1e-5 else 1)"; }
P="--prune-film-keep 1 --render-bytes wans"
S=N_pose; echo "$(date +%H:%M) start $S $(score $S)"
for n in 1 2 3; do
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt $P --steps 300 --window 8 --gate 300 \
    --lr-map 0 --w-pose 1 --w-rate 1 --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 --lr-render-scale 1e-4 \
    --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
    --save-final --out O${n}_move > O${n}_move.log 2>&1
  python exp2_train.py --tokens O${n}_move/unaccepted/tokens_joint.u8 --state O${n}_move/unaccepted/joint_state.pt $P \
    --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --lr-render 1e-12 \
    --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out O${n}_search > O${n}_search.log 2>&1
  python exp2_train.py --tokens O${n}_search/tokens_joint.u8 --state O${n}_search/joint_state.pt $P --steps 100 --window 8 \
    --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
    --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out O${n}_pose > O${n}_pose.log 2>&1
  echo "$(date +%H:%M) hop O$n: $(score O${n}_pose) vs $(score $S)"
  tar czf /root/O${n}.tgz O${n}_pose O${n}_*.log auto_O.log
  if better $(score O${n}_pose) $(score $S); then S=O${n}_pose; else echo "hop O$n did not pay, stop"; break; fi
done
if [ "$S" != "N_pose" ]; then
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt $P --search-rounds 60 --search-k 32 \
    --search-pool all --search-batches 4 --w-pose 0 --w-rate 1 --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 \
    --carrier-polish 0 --out O_wide4 > O_wide4.log 2>&1
  echo "$(date +%H:%M) wide4: $(grep 'search total' O_wide4.log)"
  python exp2_train.py --tokens O_wide4/tokens_joint.u8 --state O_wide4/joint_state.pt $P --steps 300 --window 8 --gate 50 \
    --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
    --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out O_pose > O_pose.log 2>&1
  echo "$(date +%H:%M) final pose: $(score O_pose) (from $S $(score $S))"
  tar czf /root/O_final.tgz O_wide4 O_wide4.log O_pose O_pose.log auto_O.log
fi
echo DONE > auto_O_done.flag
