#!/bin/bash
# v10 queue: grounded decision-first notes (evidence pointer first), decide-then-lookup baseline, per-source
# error counts, randomized system order. Two workers (GPU 3 and GPU 7) pull jobs from one list; a worker
# starts a job only when its GPU has room for an engine (co-tenants hold memory on both GPUs).
# Progress: logs/queue_v10.log, ends with QUEUE_V10_DONE when every job is done or failed twice.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
Q=$L/queue_v10.log
J=$L/jobs_v10.txt
U="--gpu_util 0.90 --ssn /data/yaokang/evishare/results/ssn/law.pt --qsplit test --note gdf"
M="noshare@cot,exact+suffix@cot,noshare,dbc-lazy,exact+suffix,semS0.90,semVS0.90,semcrcS-0.05,law-e0.05@cot,dbc-bstop,dbc,dbc-defer"
A="exact+suffix,dbc-nodraft,dbc-single,dbc-note,dbc-bstop,dbc@dfirst,dbc"
G="noshare,exact+suffix,semS0.90,dbc-bstop,dbc,dbc-defer"
W="exact+suffix,semS0.90,semVS0.90,dbc-bstop,dbc,dbc-defer"
if [ ! -f $J ]; then
  {
    for seed in 0 1 2; do for ds in hotpotqa musique; do
      echo "v10_main_s$seed --ds $ds --methods $M --n 200 --rate 1.2 --mu 0.1 --seed $seed"; done; done
    for ds in hotpotqa musique; do echo "v10_abl --ds $ds --methods $A --n 200 --rate 1.2 --mu 0.1 --seed 0"; done
    for m in qwen3-4b qwen2.5-7b; do echo "v10_model_$m --ds hotpotqa --model $m --methods $G --n 200 --rate 1.2 --mu 0.1 --seed 0"; done
    B="--ds hotpotqa --methods $W --n 200 --seed 0"
    echo "v10_ver_mu0.5 $B --rate 1.2 --mu 0.5"
    echo "v10_ver_mu2.0 $B --rate 1.2 --mu 2.0"
    echo "v10_zipf_0 $B --rate 1.2 --mu 0.1 --zipf 0"
    echo "v10_zipf_1.1 $B --rate 1.2 --mu 0.1 --zipf 1.1"
    echo "v10_team_1 $B --rate 1.2 --mu 0.1 --team 1"
    echo "v10_team_5 $B --rate 1.2 --mu 0.1 --team 5"
    echo "v10_load_r0.6 $B --rate 0.6 --mu 0.1"
    echo "v10_load_r2.4 $B --rate 2.4 --mu 0.1"
    echo "v10_azure $B --rate 1.2 --mu 0.1 --arrival azure"
    echo "v10_lowred $B --rate 1.2 --mu 0.1 --zipf 0 --team 1"
  } > $J
  : > $L/jobs_v10.claimed
fi
NJ=$(wc -l < $J)

claim() {  # next unclaimed job line number, atomically
  exec 9>$L/jobs_v10.lock; flock 9
  n=$(( $(wc -l < $L/jobs_v10.claimed) + 1 ))
  [ $n -le $NJ ] && echo $n >> $L/jobs_v10.claimed
  flock -u 9; [ $n -le $NJ ] && echo $n; }

room() { [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1) -lt 9000 ]; }

worker() { g=$1
  while true; do
    while ! room $g; do  # claim only when this GPU can host an engine
      [ $(wc -l < $L/jobs_v10.claimed) -ge $NJ ] && return; sleep 60; done
    n=$(claim) || break
    line=$(sed -n "${n}p" $J); tag=${line%% *}; args=${line#* }
    for attempt in 1 2; do
      while ! room $g; do sleep 60; done
      echo "START $tag gpu$g try$attempt $args $(date +%T)" >> $Q
      if CUDA_VISIBLE_DEVICES=$g $PY scripts/run_exp.py --tag $tag $args $U > $L/v10_${tag}_$(date +%s%N).log 2>&1; then
        echo "DONE $tag gpu$g $(date +%T)" >> $Q; break; fi
      echo "FAIL $tag gpu$g try$attempt $(date +%T)" >> $Q; sleep 60
    done
  done; }

worker 3 & worker 7 & wait
echo QUEUE_V10_DONE >> $Q
