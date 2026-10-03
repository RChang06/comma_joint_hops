#!/bin/bash
# (1) longer #141-hpac fine-tune from the tuned hpac, table trained too; (2) 4-batch wide search priced with the tuned
# hpac, pose refit, short hpac refit on the new map. every judged by real #141 encodes. bundles /root/P_*.tgz
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
P="--prune-film-keep 1 --render-bytes wans"
HP="--shared-every 1 --lr-map 0 --lr-coef 0 --lr-render 0 --lr-basis 0 --lr-const 0 --carrier-gn 0 --carrier-polish 0 --w-pose 0 --w-rate 1"
enc() { (cd /root/cb && python work/encode141x.py --tokens work/$1 --hpac-ihs1 work/$2 --table-body work/$3 --stream-out work/$4 > work/$5 2>&1); }
# --- 1. longer fine-tune, table too
python exp2_train.py --tokens N_pose/tokens_joint.u8 --state N_pose/joint_state.pt --hpac-ihs1 N_hpac/hpac.ihs1 \
  --table-body hpac141/table_141.bin $P $HP --steps 2000 --window 8 --gate 100 --lr-hpac 0.0005 --lr-table 0.0005 \
  --out P_hpac > P_hpac.log 2>&1
python hpac_writer141.py --state P_hpac/joint_state.pt --out-ihs1 P_hpac/hpac.ihs1 --out-section P_hpac/hpac.sec > P_writer.log 2>&1
python make_table_body.py P_hpac/joint_state.pt P_hpac/table.bin >> P_writer.log 2>&1
enc N_pose/tokens_joint.u8 P_hpac/hpac.ihs1 P_hpac/table.bin P_hpac/stream141.bin P_encode.log
echo "$(date +%H:%M) step1: stream $(stat -c %s P_hpac/stream141.bin) + section $(stat -c %s P_hpac/hpac.sec)  (tuned1: 110502 + 13486)" >> auto_P.log
tar czf /root/P_hpac.tgz P_hpac P_hpac.log P_writer.log P_encode.log auto_P.log
# --- 2. search priced with the tuned hpac + table
python exp2_train.py --tokens N_pose/tokens_joint.u8 --state N_pose/joint_state.pt --hpac-ihs1 P_hpac/hpac.ihs1 \
  --table-body P_hpac/table.bin $P --search-rounds 60 --search-k 32 --search-pool all --search-batches 4 --w-pose 0 \
  --w-rate 1 --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 --out P_wide4 > P_wide4.log 2>&1
echo "$(date +%H:%M) step2 search: $(grep 'search total' P_wide4.log)" >> auto_P.log
tar czf /root/P_wide4.tgz P_wide4 P_wide4.log auto_P.log
python exp2_train.py --tokens P_wide4/tokens_joint.u8 --state P_wide4/joint_state.pt --hpac-ihs1 P_hpac/hpac.ihs1 \
  --table-body P_hpac/table.bin $P --steps 300 --window 8 --gate 50 --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 \
  --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 \
  --out P_pose > P_pose.log 2>&1
echo "$(date +%H:%M) pose: $(score P_pose)" >> auto_P.log
tar czf /root/P_pose.tgz P_pose P_pose.log auto_P.log
python exp2_train.py --tokens P_pose/tokens_joint.u8 --state P_pose/joint_state.pt --hpac-ihs1 P_hpac/hpac.ihs1 \
  --table-body P_hpac/table.bin $P $HP --steps 1500 --window 8 --gate 150 --lr-hpac 0.001 --lr-table 0.001 \
  --out P_hpac2 > P_hpac2.log 2>&1
python hpac_writer141.py --state P_hpac2/joint_state.pt --out-ihs1 P_hpac2/hpac.ihs1 --out-section P_hpac2/hpac.sec > P_writer2.log 2>&1
python make_table_body.py P_hpac2/joint_state.pt P_hpac2/table.bin >> P_writer2.log 2>&1
enc P_pose/tokens_joint.u8 P_hpac/hpac.ihs1 P_hpac/table.bin P_hpac2/stream_h1.bin P_encode_h1.log
enc P_pose/tokens_joint.u8 P_hpac2/hpac.ihs1 P_hpac2/table.bin P_hpac2/stream141.bin P_encode2.log
echo "$(date +%H:%M) step2 final: map P_pose, hpac1 stream $(stat -c %s P_hpac2/stream_h1.bin)+$(stat -c %s P_hpac/hpac.sec), hpac2 stream $(stat -c %s P_hpac2/stream141.bin)+$(stat -c %s P_hpac2/hpac.sec)" >> auto_P.log
tar czf /root/P_final.tgz P_hpac2 P_pose P_*.log auto_P.log
echo DONE > auto_P_done.flag
