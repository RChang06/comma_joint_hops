#!/bin/bash
# stage B hop 2: renderer move on the FULL score (seg + pose + rate; carrier frozen during the move), full near-pool map
# re-search (seg + rate), quick carrier refit (strengths gn + gray/amp). judge the full score vs the start state.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
S=B1_pose
python exp2_train.py --tokens $S/tokens_joint.u8 --state $S/joint_state.pt --steps 300 --window 8 --gate 300 \
  --lr-map 0 --w-pose 1 --w-rate 1 --render-bytes wans --shared-every 25 --lr-render 0.02 --lr-render-fp16 2e-4 \
  --lr-render-scale 1e-4 --lr-hpac 0 --lr-table 0 --lr-coef 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 \
  --save-final --out B2_move > B2_move.log 2>&1
python exp2_train.py --tokens B2_move/unaccepted/tokens_joint.u8 --state B2_move/unaccepted/joint_state.pt \
  --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 --w-pose 0 --w-rate 1 --render-bytes wans \
  --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out B2_search > B2_search.log 2>&1
python exp2_train.py --tokens B2_search/tokens_joint.u8 --state B2_search/joint_state.pt --steps 100 --window 8 \
  --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --render-bytes wans --out B2_pose > B1_pose.log 2>&1
echo DONE > B2_done.flag
