#!/bin/bash
# Deployment (shadow) calibration on the calib pools: epsilon = 0 runs that draft candidates and log
# the in-pass law value; the consumer's own output labels each pass. GPU 3 only.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
for ds in hotpotqa musique; do
  $PY scripts/run_exp.py --ds $ds --split calib --methods unison-shadow --n 300 --rate 1.6 --zipf 0.8 --mu 0.5 --replay 0 --tag calibshadow > $L/calibshadow_$ds.log 2>&1 || { echo SHADOW_FAILED; exit 1; }
done
$PY scripts/calib_shadow.py --runs calibshadow/hotpotqa,calibshadow/musique --method unison-shadow --ssn $S/ssn.pt > $L/calib_shadow.log 2>&1 || { echo CALSHADOW_FAILED; exit 1; }
for m in qwen3-4b qwen2.5-7b; do
  $PY scripts/run_exp.py --ds hotpotqa --split calib --model $m --ssn $S/ssn_$m.pt --methods unison-shadow --n 300 --rate 1.6 --zipf 0.8 --mu 0.5 --replay 0 --tag calibshadow_$m > $L/calibshadow_$m.log 2>&1 || { echo SHADOW_FAILED $m; exit 1; }
  $PY scripts/calib_shadow.py --runs calibshadow_$m/hotpotqa --method unison-shadow --ssn $S/ssn_$m.pt > $L/calib_shadow_$m.log 2>&1 || { echo CALSHADOW_FAILED $m; exit 1; }
done
echo CAL_ALL_DONE
