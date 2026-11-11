#!/bin/bash
# Second revision (GPU 3 only):
#  A. fair baselines: semantic caches whose misses use the same SuffixDecoding speculator (S), incl.
#     version-keyed and Learn-then-Test-calibrated variants, in the same streams as Unison (3 seeds x 2 datasets);
#  B. cross-version stress: shadow mode at a high edit rate collects many cross-version decisions.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v8.log
run() { tag=$1; shift
  for attempt in 1 2; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3)
    echo "START $tag try$attempt gpu3_util=$u $* $(date +%T)" >> $Q
    if $PY scripts/run_exp.py --tag $tag "$@" > $L/v8_${tag}_$(date +%s%N).log 2>&1; then echo "DONE $tag $(date +%T)" >> $Q; return; fi
    echo "FAIL $tag try$attempt $(date +%T)" >> $Q; sleep 20
  done; }
U="--gpu_util 0.90 --ssn $S/law.pt"
M="noshare,exact+suffix,semS0.90,semS0.95,semVS0.90,semcrcS-0.05,unison-e0.05"
for seed in 0 1 2; do
  for ds in hotpotqa musique; do
    run v8_fair_s$seed --ds $ds --qsplit test --methods $M --n 200 --rate 1.2 --mu 0.1 --seed $seed $U
  done
done
for ds in hotpotqa musique; do
  run v8_xver --ds $ds --qsplit test --methods unison-shadow --n 300 --rate 1.2 --mu 2.0 --seed 7 --replay 1 $U
done
echo QUEUE_V8_DONE >> $Q
