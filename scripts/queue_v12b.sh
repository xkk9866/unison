#!/bin/bash
# v12b queue (GPU 3 only): same job list as queue_v12.sh, adaptive to a co-tenant on GPU 3.
# A job starts when GPU 3 has >= 14 GiB free; gpu_util is the whole-card budget (vLLM V1 counts other
# processes): util = (used_by_others + 13.5 GiB) / total, capped at 0.90. Failed jobs go to the back of the
# queue (up to 6 attempts). Co-tenant memory is logged per job and per method (gpu_others_mib).
# Progress: logs/queue_v12.log (shared with v12), ends with QUEUE_V12B_DONE.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs; Q=$L/queue_v12.log; RES=/data/yaokang/evishare/results
while pgrep -f "run_exp.py --tag v12_" > /dev/null; do sleep 30; done   # let the running v12 job finish
source <(sed -n '/^M=/,/^r()/p' scripts/queue_v12.sh)
eval "$(sed -n '/^jobs()/,/^}/p' scripts/queue_v12.sh)"
jobs > $L/jobs_v12b.txt
declare -A tries
mapfile -t todo < $L/jobs_v12b.txt
while [ ${#todo[@]} -gt 0 ]; do
  line=${todo[0]}; todo=("${todo[@]:1}")
  tag=${line%% *}; args=${line#* }; ds=$(echo "$args" | sed -E 's/.*--ds ([a-z]+).*/\1/')
  [ -f $RES/$tag/$ds/summary.json ] && continue
  k="$tag/$ds"; tries[$k]=$(( ${tries[$k]:-0} + 1 ))
  [ ${tries[$k]} -gt 6 ] && { echo "GIVEUP $k" >> $Q; continue; }
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    [ $used -lt 9800 ] && break; sleep 30; done
  util=$(python3 -c "print(min(0.90, round(($used + 13800) / 24120, 2)))")
  echo "START $tag try${tries[$k]} $ds util=$util others=$used $(date +%T)" >> $Q
  if $PY scripts/run_exp.py --tag $tag $args --gpu_util $util --qsplit test --note dfp > $L/v12_${tag}_${ds}_$(date +%s).log 2>&1; then
    echo "DONE $tag $ds $(date +%T)" >> $Q
  else
    echo "FAIL $tag $ds try${tries[$k]} $(date +%T)" >> $Q; todo+=("$line"); sleep 20
  fi
done
echo QUEUE_V12B_DONE >> $Q
