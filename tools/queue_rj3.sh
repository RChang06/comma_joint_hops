#!/bin/bash
# three short joint map+renderer probes (seg + rate, pose off), each 300 steps, gate every 100
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
C="--tokens c2_s2/tokens_joint.u8 --state c2_s4/joint_state.pt --steps 300 --window 8 --gate 100 --map-mode soft --map-t0 0.5 \
  --map-t1 0.1 --opt maxnorm --lr-map 1.0 --w-pose 0 --w-rate 1 --render-bytes wans --shared-every 25 --lr-hpac 0 --lr-table 0 \
  --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0"
python exp2_train.py $C --lr-render 1e-12 --lr-render-fp16 1e-12 --lr-render-scale 1e-4 --out rj_e1 > rj_e1.log 2>&1
python exp2_train.py $C --kappa 0.1 --lr-render 0.02 --lr-render-fp16 2e-4 --lr-render-scale 1e-4 --out rj_e2 > rj_e2.log 2>&1
python exp2_train.py $C --tau 0.01 --lr-render 0.02 --lr-render-fp16 2e-4 --lr-render-scale 1e-4 --out rj_e3 > rj_e3.log 2>&1
echo DONE > rj3_done.flag
