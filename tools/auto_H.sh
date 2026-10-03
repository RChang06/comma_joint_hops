#!/bin/bash
# full wide-pool search with 4 candidate batches per round on the pruned + map-trained state (pose later)
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens G_s1/tokens_joint.u8 --state G_s1/joint_state.pt --prune-film-keep 1 --search-rounds 60 \
  --search-k 32 --search-pool all --search-batches 4 --w-pose 0 --w-rate 1 --render-bytes wans --lr-render 1e-12 \
  --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out H_wide4 > H_wide4.log 2>&1
echo "$(date +%H:%M) wide4 search: $(grep 'search total' H_wide4.log)"
tar czf /root/auto_H.tgz auto_H.log H_wide4 H_wide4.log
echo DONE > auto_H_done.flag
