#!/bin/bash
# fine-tune #141's hpac on the final map (K_final), rows clamped to their stored bit depths, table kept; then write it in
# #141's format and encode the map with #141's full coder (correctors) to judge stream + hpac section for real.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
python exp2_train.py --tokens K_final/tokens_joint.u8 --state K_final/joint_state.pt --hpac-ihs1 hpac141/ihs1_141.bin \
  --table-body hpac141/table_141.bin --prune-film-keep 1 --render-bytes wans --steps 1500 --window 8 --gate 150 \
  --shared-every 1 --lr-map 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0 \
  --carrier-gn 0 --carrier-polish 0 --w-pose 0 --w-rate 1 --out M_hpac > M_hpac.log 2>&1
echo "$(date +%H:%M) trainer: $(grep 'gate 0:' M_hpac.log | grep -o '"token_bytes": [0-9.]*') -> $(grep -o '"token_bytes": [0-9.]*' M_hpac/best.json)"
python hpac_writer141.py --state M_hpac/joint_state.pt --out-ihs1 M_hpac/hpac.ihs1 --out-section M_hpac/hpac.sec > M_writer.log 2>&1
echo "$(date +%H:%M) writer: $(tail -2 M_writer.log | tr '\n' ' ')"
(cd /root/cb && python work/encode141x.py --tokens work/K_final/tokens_joint.u8 --hpac-ihs1 work/M_hpac/hpac.ihs1 \
  --stream-out work/M_hpac/stream141.bin > work/M_encode.log 2>&1)
echo "$(date +%H:%M) stream $(stat -c %s M_hpac/stream141.bin) B  section $(stat -c %s M_hpac/hpac.sec) B"
tar czf /root/M.tgz M_hpac M_*.log auto_M.log
echo DONE > auto_M_done.flag
