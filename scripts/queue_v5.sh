#!/bin/bash
# After queue_v4: re-run the phase-2 runs that completed under gpu_util 0.42 so that every sweep uses the
# same engine configuration (gpu_util 0.65; another user's job holds 7 GB of GPU 3). Progress: logs/queue_v5.log
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v5.log
until grep -q QUEUE_V4_DONE $L/queue_v4.log; do sleep 30; done
run() { tag=$1; shift
  for attempt in 1 2; do
    echo "START $tag try$attempt $* $(date +%T)" >> $Q
    if $PY scripts/run_exp.py --tag $tag "$@" > $L/v5_${tag}_$(date +%s%N).log 2>&1; then echo "DONE $tag $(date +%T)" >> $Q; return; fi
    echo "FAIL $tag try$attempt $(date +%T)" >> $Q
  done; }
R=1.2
G="--ssn $S/law.pt --gpu_util 0.65"
SWEEP="noshare,exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.1"
ABL="exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.05-latest,unison-e0.05-nolaw"
run v3_abl --ds hotpotqa --methods noshare,$ABL --n 150 --rate $R --mu 0.5 $G
run v3_load_r0.6 --ds hotpotqa --methods $SWEEP --n 150 --rate 0.6 --mu 0.1 $G
run v3_load_r1.2 --ds hotpotqa --methods $SWEEP --n 150 --rate 1.2 --mu 0.1 $G
run v3_azure --ds hotpotqa --methods $SWEEP --n 200 --rate $R --arrival azure --mu 0.1 $G
echo QUEUE_V5_DONE >> $Q
