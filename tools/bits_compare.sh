#!/bin/bash
# per-frame bits of two maps (cycle-2 map and current map) under the trainer's coder and under #141's real coder
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
for m in c2_s2 final_pose; do
  python exp2_train.py --tokens $m/tokens_joint.u8 --state final_pose/joint_state.pt --steps 0 --lr-render 0 --lr-basis 0 \
    --carrier-gn 0 --carrier-polish 0 --w-pose 0 --w-rate 1 --out bc_t_$m > bc_t_$m.log 2>&1
  (cd /root/cb && python work/encode141_bits.py --tokens work/$m/tokens_joint.u8 --bits-out work/bc_141_$m.npy > work/bc_141_$m.log 2>&1)
done
echo DONE > bc_done.flag
