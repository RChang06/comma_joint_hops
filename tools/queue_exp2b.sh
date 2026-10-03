#!/bin/bash
# step 2 rerun with the near-error pool (diagnostic winner), from step 1's map; pose left to step 5
set -e
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens e2_train/tokens_joint.u8 --search-rounds 60 --search-k 32 --search-pool near --search-batches 1 \
  --w-pose 1 --w-rate 1 --lr-render 0 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out e2b_search > e2b_2_search.log 2>&1
python encode135.py --tokens e2b_search/tokens_joint.u8 --out e2b_search/archive.zip > e2b_3_encode.log 2>&1
python score_map135.py --tokens e2b_search/tokens_joint.u8 --archive e2b_search/archive.zip > e2b_4_score.log 2>&1
echo DONE > e2b_done.flag
