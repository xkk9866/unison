#!/bin/bash
# v11 repair queue: after queue_v11 ends, re-runs every job whose summary.json is missing (OOM when a
# co-tenant grows during engine start). gpu_util is set from the memory actually free at launch.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs; R=/data/yaokang/evishare/results
Q=$L/queue_v11r.log; J=$L/jobs_v11.txt
U0="--ssn /data/yaokang/evishare/results/ssn/law.pt --qsplit test --note dfp"
: > $Q
missing() { while read -r tag args; do ds=$(echo "$args" | sed 's/.*--ds \([a-z]*\).*/\1/')
  [ -f $R/$tag/$ds/summary.json ] || echo "$tag $args"; done < $J; }
missing > $L/jobs_v11r.txt; NJ=$(wc -l < $L/jobs_v11r.txt); : > $L/jobs_v11r.claimed
echo "REPAIR $NJ jobs $(date +%T)" >> $Q
claim() { exec 9>$L/jobs_v11r.lock; flock 9; n=$(( $(wc -l < $L/jobs_v11r.claimed) + 1 ))
  [ $n -le $NJ ] && echo $n >> $L/jobs_v11r.claimed; flock -u 9; [ $n -le $NJ ] && echo $n; }
free() { nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits -i $1; }
worker() { g=$1
  while n=$(claim); do line=$(sed -n "${n}p" $L/jobs_v11r.txt); tag=${line%% *}; args=${line#* }
    for attempt in 1 2 3 4 5 6 7 8; do
      while :; do read f t < <(free $g | tr -d ','); [ $f -gt 14000 ] && break; sleep 60; done
      u=0.90  # vLLM budgets total*util including co-tenant memory
      echo "START $tag gpu$g try$attempt util$u $args $(date +%T)" >> $Q
      if CUDA_VISIBLE_DEVICES=$g $PY scripts/run_exp.py --tag $tag $args $U0 --gpu_util $u > $L/v11r_${tag}_$(date +%s%N).log 2>&1; then
        echo "DONE $tag gpu$g $(date +%T)" >> $Q; break; fi
      echo "FAIL $tag gpu$g try$attempt $(date +%T)" >> $Q; sleep 120
    done; done; }
worker 3 & worker 7 & wait
echo QUEUE_V11R_DONE >> $Q
