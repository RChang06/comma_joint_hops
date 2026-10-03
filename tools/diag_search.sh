#!/bin/bash
# step-2 search variants on the same 40 pairs from step 1's map; compare net gain and time
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
C="--tokens e2_train/tokens_joint.u8 --search-rounds 60 --search-k 32 --search-frames 200:240 --w-pose 1 --w-rate 1 --lr-render 0 --lr-basis 0 --carrier-gn 0 --carrier-polish 0"
python exp2_train.py $C --search-pool all --search-batches 1 --out d_all1 > d_all1.log 2>&1
python exp2_train.py $C --search-pool near --search-batches 1 --out d_near1 > d_near1.log 2>&1
python exp2_train.py $C --search-pool all --search-batches 4 --out d_all4 > d_all4.log 2>&1
echo DONE > diag_done.flag
