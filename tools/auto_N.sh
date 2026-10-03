#!/bin/bash
# final squeeze from K_final: 4-batch wide search -> pose refit -> #141 hpac fine-tune on the final map, judged by a
# real #141 encode. every stage bundled to /root/N_<stage>.tgz the moment it ends.
source /root/venv/bin/activate
cd /root/cb/work
export CPR1_RC64_LIBRARY=/tmp/rc64_135.so PYTHONUNBUFFERED=1
score() { python -c "import json;print(json.load(open('$1/best.json'))['score'])"; }
P="--prune-film-keep 1 --render-bytes wans"
python exp2_train.py --tokens K_final/tokens_joint.u8 --state K_final/joint_state.pt $P --search-rounds 60 --search-k 32 \
  --search-pool all --search-batches 4 --w-pose 0 --w-rate 1 --lr-render 1e-12 --lr-basis 0 --carrier-gn 0 --carrier-polish 0 \
  --out N_wide4 > N_wide4.log 2>&1
echo "$(date +%H:%M) wide4: $(grep 'search total' N_wide4.log)"; tar czf /root/N_wide4.tgz N_wide4 N_wide4.log auto_N.log
python exp2_train.py --tokens N_wide4/tokens_joint.u8 --state N_wide4/joint_state.pt $P --steps 300 --window 8 --gate 50 \
  --shared-every 10 --lr-map 0 --lr-hpac 0 --lr-table 0 --lr-render 0 --lr-coef 0 --lr-basis 0.02 --lr-const 0.01 \
  --carrier-gn 8 --carrier-polish 1 --w-pose 1 --w-rate 1 --out N_pose > N_pose.log 2>&1
echo "$(date +%H:%M) pose: $(score N_pose) (K_final $(score K_final))"; tar czf /root/N_pose.tgz N_pose N_pose.log auto_N.log
B=N_pose; python -c "import sys;sys.exit(0 if $(score N_pose) < $(score K_final) else 1)" || B=K_final
echo "$(date +%H:%M) map for hpac: $B"
python exp2_train.py --tokens $B/tokens_joint.u8 --state $B/joint_state.pt --hpac-ihs1 hpac141/ihs1_141.bin \
  --table-body hpac141/table_141.bin $P --steps 1500 --window 8 --gate 150 --shared-every 1 --lr-map 0 --lr-coef 0 \
  --lr-render 0 --lr-basis 0 --lr-const 0 --lr-hpac 0.002 --lr-table 0 --carrier-gn 0 --carrier-polish 0 --w-pose 0 \
  --w-rate 1 --out N_hpac > N_hpac.log 2>&1
echo "$(date +%H:%M) hpac trainer: $(grep 'gate 0:' N_hpac.log | grep -o '"token_bytes": [0-9.]*') -> $(grep -o '"token_bytes": [0-9.]*' N_hpac/best.json)"
python hpac_writer141.py --state N_hpac/joint_state.pt --out-ihs1 N_hpac/hpac.ihs1 --out-section N_hpac/hpac.sec > N_writer.log 2>&1
tar czf /root/N_hpac.tgz N_hpac N_hpac.log N_writer.log auto_N.log
(cd /root/cb && python work/encode141x.py --tokens work/$B/tokens_joint.u8 --stream-out work/N_hpac/base141.bin > work/N_base.log 2>&1)
(cd /root/cb && python work/encode141x.py --tokens work/$B/tokens_joint.u8 --hpac-ihs1 work/N_hpac/hpac.ihs1 \
  --stream-out work/N_hpac/stream141.bin > work/N_encode.log 2>&1)
echo "$(date +%H:%M) #141 hpac: stream $(stat -c %s N_hpac/base141.bin) + 13515 | tuned: stream $(stat -c %s N_hpac/stream141.bin) + section $(stat -c %s N_hpac/hpac.sec)"
tar czf /root/N_final.tgz N_hpac N_*.log auto_N.log
echo DONE > auto_N_done.flag
