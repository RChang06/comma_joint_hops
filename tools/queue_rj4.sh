#!/bin/bash
# big renderer move (map frozen, 300 steps, sigmoid seg loss, pose off) then full near-pool map search to re-adapt;
# compare the searched seg + rate with the stage start 0.140019
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens c2_s2/tokens_joint.u8 --state c2_s4/joint_state.pt --steps 300 --window 8 --gate 300 \
  --lr-map 0 --w-pose 0 --w-rate 1 --render-bytes wans --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 \
  --lr-render-scale 1e-4 --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --save-final --out rj_move > rj_move.log 2>&1
python exp2_train.py --tokens rj_move/unaccepted/tokens_joint.u8 --state rj_move/unaccepted/joint_state.pt \
  --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans \
  --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out rj_move_search > rj_move_search.log 2>&1
echo DONE > rj4_done.flag
