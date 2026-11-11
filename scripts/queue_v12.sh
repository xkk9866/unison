#!/bin/bash
# v12 queue (GPU 3 only, one engine at a time, isolated): decision-keyed provenance, the Self-Boundary+Defer
# baseline, vCache-style baseline, background accounting, nondeterminism floor.
# A job starts only when GPU 3 holds < 1 GiB of other processes' memory (isolation); the co-tenant memory
# during each method is recorded in summary.json (gpu_others_mib). Failed jobs are retried up to 3 times.
# Progress: logs/queue_v12.log, ends with QUEUE_V12_DONE.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
Q=$L/queue_v12.log
U="--gpu_util 0.90 --qsplit test --note dfp"
M="noshare@cot,exact+suffix@cot,exact+suffix,semS0.90,vcacheS,dbc-bstop,dbc-sbd,dbc-canon"
A="exact+suffix,dbc-lazy,dbc,dbc-sbd,dbc-sbdcanon,dbc-cont,dbc-canonnofg,dbc-canon"
W="exact+suffix,semS0.90,dbc-bstop,dbc-sbd,dbc-canon"
r() { [ $1 = musique ] && echo 0.6 || echo 1.2; }
jobs() {
  for seed in 0 1 2; do for ds in hotpotqa musique; do
    echo "v12_main_s$seed --ds $ds --methods $M --n 200 --rate $(r $ds) --mu 0.1 --seed $seed --replay2 1"; done; done
  for seed in 0 1; do for ds in hotpotqa musique; do
    echo "v12_abl_s$seed --ds $ds --methods $A --n 200 --rate $(r $ds) --mu 0.1 --seed $seed"; done; done
  B="--ds hotpotqa --methods $W --n 200 --seed 0"
  echo "v12_load_r0.6 $B --rate 0.6 --mu 0.1"
  echo "v12_load_r1.8 $B --rate 1.8 --mu 0.1"
  echo "v12_load_r2.4 $B --rate 2.4 --mu 0.1"
  echo "v12_muload_r1.2 --ds musique --methods $W --n 200 --seed 0 --rate 1.2 --mu 0.1"
  echo "v12_ver_mu0.5 $B --rate 1.2 --mu 0.5"
  echo "v12_ver_mu2.0 $B --rate 1.2 --mu 2.0"
  echo "v12_zipf_0 $B --rate 1.2 --mu 0.1 --zipf 0"
  echo "v12_zipf_1.1 $B --rate 1.2 --mu 0.1 --zipf 1.1"
  echo "v12_team_1 $B --rate 1.2 --mu 0.1 --team 1"
  echo "v12_team_5 $B --rate 1.2 --mu 0.1 --team 5"
  echo "v12_azure $B --rate 1.2 --mu 0.1 --arrival azure"
  echo "v12_lowred $B --rate 1.2 --mu 0.1 --zipf 0 --team 1"
  for m in qwen3-4b qwen2.5-7b; do
    echo "v12_model_$m --ds hotpotqa --model $m --methods $W --n 200 --rate 1.2 --mu 0.1 --seed 0"; done
  for ds in hotpotqa musique; do  # seed 2 reruns (the original s2 runs overlapped a co-tenant)
    echo "v12_main_s2r --ds $ds --methods $M --n 200 --rate $(r $ds) --mu 0.1 --seed 2 --replay2 1"; done
}
room() { [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3) -lt 1024 ]; }
jobs | while read -r line; do
  tag=${line%% *}; args=${line#* }; ds=$(echo "$args" | sed -E 's/.*--ds ([a-z]+).*/\1/')
  [ -f /data/yaokang/evishare/results/$tag/$ds/summary.json ] && { echo "SKIP $tag $ds" >> $Q; continue; }
  for attempt in 1 2 3; do
    w=0; while ! room && [ $w -lt 360 ]; do sleep 10; w=$((w+1)); done
    echo "START $tag try$attempt $args $(date +%T) others=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)" >> $Q
    if $PY scripts/run_exp.py --tag $tag $args $U > $L/v12_${tag}_${ds}_$(date +%s).log 2>&1; then
      echo "DONE $tag $ds $(date +%T)" >> $Q; break; fi
    echo "FAIL $tag $ds try$attempt $(date +%T)" >> $Q; sleep 30
  done
done
echo QUEUE_V12_DONE >> $Q
