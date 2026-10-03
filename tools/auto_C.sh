#!/bin/bash
# unattended cycle after auto_B: map training (seg + bytes, softer settings) -> near-pool search -> hpac + table refit
# (per-row depth clamp) -> 600-step pose stage. starts from final_pose; progress in auto_C.log
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
while [ ! -f auto_B_done.flag ]; do sleep 30; done
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
S=final_pose
echo "$(date +%H:%M) cycle start from $S full $(score $S)"
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 1500 --window 8 --gate 150 \
  --map-mode soft --map-t0 0.5 --map-t1 0.1 --opt maxnorm --lr-map 1.0 --w-pose 0 --w-rate 1 --frame-accept \
  --render-bytes wans --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 \
  --carrier-gn 0 --carrier-polish 0 --out C_s1 > C_s1.log 2>&1
echo "$(date +%H:%M) step 1 (seg+rate): $(score C_s1)"
python exp2_train.py --tokens C_s1/tokens_joint.u8 --state C_s1/joint_state.pt --search-rounds 60 --search-k 32 \
  --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 --lr-basis 0 \
  --carrier-gn 0 --carrier-polish 0 --out C_s2 > C_s2.log 2>&1
echo "$(date +%H:%M) step 2 search: $(grep 'search total' C_s2.log)"
python exp2_train.py --tokens C_s2/tokens_joint.u8 --state C_s2/joint_state.pt --steps 1500 --window 8 --gate 150 \
  --shared-every 1 --lr-map 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0.002 \
  --carrier-gn 0 --carrier-polish 0 --w-pose 0 --w-rate 1 --render-bytes wans --out C_s3 > C_s3.log 2>&1
echo "$(date +%H:%M) step 3 hpac: $(score C_s3)"
python exp2_train.py --tokens C_s2/tokens_joint.u8 --state C_s3/joint_state.pt --steps 600 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out C_s4 > C_s4.log 2>&1
echo "$(date +%H:%M) step 4 pose: $(score C_s4) (cycle start $(score $S))"
tar czf /root/auto_C.tgz auto_C.log C_s*.log C_s1 C_s2 C_s3 C_s4
echo DONE > auto_C_done.flag
