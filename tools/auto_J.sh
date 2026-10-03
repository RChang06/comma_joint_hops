#!/bin/bash
# unattended pruned-line playbook after I_pose: cycle 2 (map training -> wide x4 search -> pose stage), then renderer
# hops on the full score (pruned rows pinned) while each beats its start by > 1e-5 (max 4), then a final pose stage.
# every stage bundled to /root/J_<stage>.tgz as it finishes; progress in auto_J.log
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_I_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
better() { python -c "import sys;sys.exit(0 if $1 < $2 - 1e-5 else 1)"; }
P="--prune-film-keep 1 --render-bytes wans"
echo "$(date +%H:%M) start I_pose $(score I_pose)"
python exp2_train.py --tokens I_pose/tokens_joint.u8 --state I_pose/joint_state.pt $P --steps 1500 --window 8 --gate 150 \
  --map-mode soft --map-t0 0.5 --map-t1 0.1 --opt maxnorm --lr-map 1.0 --w-pose 0 --w-rate 1 --frame-accept \
  --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --out J_s1 > J_s1.log 2>&1
echo "$(date +%H:%M) c2 map training (seg+rate): $(grep 'gate 0:' J_s1.log | grep -o '"score": [0-9.]*') -> $(score J_s1)"
tar czf /root/J_s1.tgz J_s1 J_s1.log
python exp2_train.py --tokens J_s1/tokens_joint.u8 --state J_s1/joint_state.pt $P --search-rounds 60 --search-k 32 \
  --search-pool all --search-batches 4 --w-pose 0 --w-rate 1 --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 \
  --out J_wide4 > J_wide4.log 2>&1
echo "$(date +%H:%M) c2 wide4 search: $(grep 'search total' J_wide4.log)"
tar czf /root/J_wide4.tgz J_wide4 J_wide4.log
python exp2_train.py --tokens J_wide4/tokens_joint.u8 --state J_wide4/joint_state.pt $P --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out J_pose > J_pose.log 2>&1
echo "$(date +%H:%M) c2 pose stage: $(score J_pose) (cycle 1 end $(score I_pose))"
tar czf /root/J_pose.tgz J_pose J_pose.log auto_J.log
S=I_pose; if better $(score J_pose) $(score I_pose); then S=J_pose; fi
for n in 1 2 3 4; do
  python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt $P --steps 300 --window 8 --gate 300 \
    --lr-map 0 --w-pose 1 --w-rate 1 --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 --lr-render-scale 1e-4 \
    --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
    --save-final --out K${n}_move > K${n}_move.log 2>&1
  python exp2_train.py --tokens K${n}_move/unaccepted/tokens_joint.u8 --state K${n}_move/unaccepted/joint_state.pt $P \
    --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --lr-render 1e-12 \
    --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out K${n}_search > K${n}_search.log 2>&1
  python exp2_train.py --tokens K${n}_search/tokens_joint.u8 --state K${n}_search/joint_state.pt $P --steps 100 --window 8 \
    --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
    --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out K${n}_pose > K${n}_pose.log 2>&1
  echo "$(date +%H:%M) hop K$n: $(score K${n}_pose) vs start $(score $S)"
  tar czf /root/K${n}.tgz K${n}_pose K${n}_*.log auto_J.log
  if better $(score K${n}_pose) $(score $S); then S=K${n}_pose; else echo "hop K$n did not pay, stop"; break; fi
done
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt $P --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out K_final > K_final.log 2>&1
echo "$(date +%H:%M) final pose: $(score K_final) (from $S $(score $S))"
tar czf /root/K_final.tgz K_final K_final.log auto_J.log
echo DONE > auto_J_done.flag
