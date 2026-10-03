#!/bin/bash
# joint map + renderer smoke 2: no per-frame accept (it reverted the map's re-adaptation), renderer steps once per
# 600-frame sweep, gates every 300 steps so each frame's map gets ~4 visits per renderer move
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens c2_s2/tokens_joint.u8 --state c2_s4/joint_state.pt --steps 900 --window 8 --gate 300 \
  --map-mode soft --map-t0 0.5 --map-t1 0.1 --opt maxnorm --lr-map 1.0 --w-pose 0 --w-rate 1 \
  --lr-render 0.02 --lr-render-fp16 2e-4 --lr-render-scale 1e-4 --render-bytes wans --shared-every 75 \
  --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --out rj_smoke2 > rj_smoke2.log 2>&1
echo DONE > rj_smoke2_done.flag
