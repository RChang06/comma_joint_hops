#!/bin/bash
# search style on the pruned state (F_search), pairs 200:240
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
C="--tokens F_search/tokens_joint.u8 --state F_search/joint_state.pt --search-rounds 60 --search-frames 200:240 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0"
python exp2_train.py $C --search-pool near --search-k 32 --search-batches 1 --out d3_near > d3_near.log 2>&1
python exp2_train.py $C --search-pool near --search-k 32 --search-batches 4 --out d3_near4 > d3_near4.log 2>&1
python exp2_train.py $C --search-pool all --search-k 32 --search-batches 1 --out d3_wide > d3_wide.log 2>&1
python exp2_train.py $C --search-pool all --search-k 32 --search-batches 4 --out d3_wide4 > d3_wide4.log 2>&1
echo DONE > d3_done.flag
