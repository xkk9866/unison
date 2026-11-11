#!/bin/bash
# Revision queue (GPU 3 only). Answers the review:
#  A. shadow calibration on the *calibration half* of each evaluation workload (same distribution,
#     disjoint questions; strict decision equality), for all three models;
#  B. main comparison on the disjoint *test half*, three independent seeds, incl. the
#     draft-without-commit control (unison-shadow) and quality-matched baselines;
#  C. model generality on the test half.
# Before every run we wait (<= 20 min) for the co-located tenant on GPU 3 to be idle and log the
# utilisation we observed. Progress: logs/queue_v7.log
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v7.log
idle() { for i in $(seq 1 6); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3);
         [ "$u" -lt 15 ] && break; sleep 30; done; echo "  gpu3_util_before=$u $(date +%T)" >> $Q; }
run() { tag=$1; shift
  for attempt in 1 2; do
    idle
    echo "START $tag try$attempt $* $(date +%T)" >> $Q
    if $PY scripts/run_exp.py --tag $tag "$@" > $L/v7_${tag}_$(date +%s%N).log 2>&1; then echo "DONE $tag $(date +%T)" >> $Q; return; fi
    echo "FAIL $tag try$attempt $(date +%T)" >> $Q; sleep 20
  done; }
U="--gpu_util 0.90"
# ---- A: shadow calibration on the calibration half
for ds in hotpotqa musique; do
  run v7_shadow --ds $ds --qsplit calib --methods unison-shadow --n 400 --rate 1.2 --mu 0.1 --seed 100 --replay 0 --ssn $S/law.pt $U
  $PY scripts/calib_shadow.py --runs v7_shadow/$ds --method unison-shadow --ssn $S/law.pt --workload $ds > $L/v7_cal_$ds.log 2>&1
done
for m in qwen3-4b qwen2.5-7b; do
  run v7_shadow_$m --ds hotpotqa --model $m --qsplit calib --methods unison-shadow --n 400 --rate 1.2 --mu 0.1 --seed 100 --replay 0 --ssn $S/law_$m.pt $U
  $PY scripts/calib_shadow.py --runs v7_shadow_$m/hotpotqa --method unison-shadow --ssn $S/law_$m.pt --workload hotpotqa > $L/v7_cal_$m.log 2>&1
done
echo CALIB_DONE >> $Q
# ---- B: main comparison, test half, three seeds
M="noshare,exact+suffix,sem0.90,sem0.95,semcrc-0.05,osc-e0.05,unison-shadow,unison-e0.02,unison-e0.05,unison-e0.1"
for seed in 0 1 2; do
  for ds in hotpotqa musique; do
    run v7_main_s$seed --ds $ds --qsplit test --methods $M --n 200 --rate 1.2 --mu 0.1 --seed $seed --ssn $S/law.pt $U
  done
done
# ---- C: models
for m in qwen3-4b qwen2.5-7b; do
  run v7_model_$m --ds hotpotqa --model $m --qsplit test --methods noshare,exact+suffix,sem0.95,unison-e0.02,unison-e0.05 --n 200 --rate 1.2 --mu 0.1 --seed 0 --ssn $S/law_$m.pt $U
done
echo QUEUE_V7_DONE >> $Q
