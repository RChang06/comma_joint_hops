#!/bin/bash
# rerun of the #141 hpac fine-tune with the gate-weight fix (rate-only gates added the pose term), then real encodes
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
P="--prune-film-keep 1 --render-bytes wans"; B=N_pose
rm -rf N_hpac
python exp2_train.py --tokens $B/tokens_joint.u8 --state $B/joint_state.pt --hpac-ihs1 hpac141/ihs1_141.bin \
  --table-body hpac141/table_141.bin $P --steps 1500 --window 8 --gate 150 --shared-every 1 --lr-map 0 --lr-coef 0 \
  --lr-render 0 --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0 --carrier-gn 0 --carrier-polish 0 --w-pose 0 \
  --w-rate 1 --out N_hpac > N_hpac.log 2>&1
echo "$(date +%H:%M) hpac trainer (fixed gates): $(grep 'gate 0:' N_hpac.log | grep -o '"token_bytes": [0-9.]*') -> $(grep -o '"token_bytes": [0-9.]*' N_hpac/best.json)" >> auto_N.log
python hpac_writer141.py --state N_hpac/joint_state.pt --out-ihs1 N_hpac/hpac.ihs1 --out-section N_hpac/hpac.sec > N_writer.log 2>&1
tar czf /root/N_hpac.tgz N_hpac N_hpac.log N_writer.log auto_N.log
(cd /root/cb && python work/encode141x.py --tokens work/$B/tokens_joint.u8 --stream-out work/N_hpac/base141.bin > work/N_base.log 2>&1)
(cd /root/cb && python work/encode141x.py --tokens work/$B/tokens_joint.u8 --hpac-ihs1 work/N_hpac/hpac.ihs1 \
  --stream-out work/N_hpac/stream141.bin > work/N_encode.log 2>&1)
echo "$(date +%H:%M) #141 hpac: stream $(stat -c %s N_hpac/base141.bin) + 13515 | tuned: stream $(stat -c %s N_hpac/stream141.bin) + section $(stat -c %s N_hpac/hpac.sec)" >> auto_N.log
tar czf /root/N_final.tgz N_hpac N_*.log auto_N.log
echo DONE > auto_N_done.flag
