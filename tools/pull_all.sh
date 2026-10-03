#!/bin/bash
# copy every new stage bundle (/root/N_*.tgz, /root/O*.tgz) off the box as it appears
K="-i $HOME/.ssh/id_ed25519_personal -o ConnectTimeout=20"
D=/d/projects/comma_b/work/rj_results; mkdir -p $D
for i in $(seq 1 55); do
  for f in $(ssh $K -p 34468 root@ssh6.vast.ai 'ls /root/N_*.tgz /root/O*.tgz 2>/dev/null' 2>/dev/null); do
    b=$(basename $f); [ -f $D/$b ] && continue
    scp $K -P 34468 -q root@ssh6.vast.ai:$f $D/ && (cd $D && tar xzf $b) && echo "$(date +%H:%M) pulled $b"
  done
  sleep 60
done
