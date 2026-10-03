#!/bin/bash
# every 10 min: pack the box's cycle result folders + logs and pull them to work/cycle_sync/
P=${1:-21476}; H=${2:-ssh6.vast.ai}
K="-i $HOME/.ssh/id_ed25519_personal -o ConnectTimeout=20"
mkdir -p /d/projects/comma_b/work/cycle_sync
while true; do
  ssh $K -p $P root@$H 'cd /root/cb/work && tar czf /root/sync.tgz --exclude="*.u8" c1_* c2_* *.log *.flag 2>/dev/null; cd /root/cb/work && tar czf /root/sync_maps.tgz $(ls -d c*_*/tokens_joint.u8 2>/dev/null) 2>/dev/null' >/dev/null 2>&1
  scp $K -P $P -q root@$H:/root/sync.tgz /d/projects/comma_b/work/cycle_sync/sync.tgz 2>/dev/null && (cd /d/projects/comma_b/work/cycle_sync && tar xzf sync.tgz 2>/dev/null)
  scp $K -P $P -q root@$H:/root/sync_maps.tgz /d/projects/comma_b/work/cycle_sync/sync_maps.tgz 2>/dev/null
  echo "$(date +%H:%M) synced" >> /d/projects/comma_b/work/cycle_sync/sync.log
  sleep 600
done
