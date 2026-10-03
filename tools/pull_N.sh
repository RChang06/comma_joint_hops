#!/bin/bash
# copy each finished stage bundle off the box the moment it appears
K="-i $HOME/.ssh/id_ed25519_personal -o ConnectTimeout=20"
D=/d/projects/comma_b/work/rj_results; mkdir -p $D
for i in $(seq 1 110); do
  for f in N_wide4 N_pose N_hpac N_final; do
    [ -f $D/$f.tgz ] && continue
    if ssh $K -p 34468 root@ssh6.vast.ai "test -f /root/$f.tgz" 2>/dev/null; then
      scp $K -P 34468 -q root@ssh6.vast.ai:/root/$f.tgz $D/ && (cd $D && tar xzf $f.tgz) && echo "$(date +%H:%M) pulled $f"
    fi
  done
  [ -f $D/N_final.tgz ] && break
  sleep 60
done
tail -6 $D/auto_N.log
