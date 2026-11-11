#!/bin/bash
# GPU 3 only. Train + calibrate the SSN (qwen3-8b), pilot the new policy, then calibrate the same
# (frozen) law on the other two models with their own calibration logs.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
O=/data/yaokang/evishare/results/ssn
until grep -q REPLAY_DATA_DONE $L/replay_data.log; do sleep 10; done
$PY scripts/train_ssn.py > $L/train_ssn.log 2>&1 || { echo TRAIN_FAILED; exit 1; }
$PY scripts/run_exp.py --ds hotpotqa --methods noshare,exact+suffix,unison,unison-e0.05 --n 60 --rate 1.2 --mu 0.3 --tag pilot3 > $L/pilot3.log 2>&1
echo PILOT3_DONE
for m in qwen3-4b qwen2.5-7b; do
  $PY scripts/run_exp.py --ds hotpotqa --split calib --model $m --methods noshare --n 300 --rate 2.0 --zipf 0.8 --mu 0.5 --replay 0 --tag calibdata_$m > $L/calibdata_$m.log 2>&1 || { echo CALIBDATA_FAILED $m; exit 1; }
  $PY scripts/build_replay.py --tag calibdata_$m --split calib --ds hotpotqa --model $m --out $O/calib_$m.pkl.gz > $L/replay_$m.log 2>&1 || { echo REPLAY_FAILED $m; exit 1; }
  $PY scripts/train_ssn.py --load $O/ssn.pt --calib $O/calib_$m.pkl.gz --out $O/ssn_$m.pt > $L/calib_ssn_$m.log 2>&1 || { echo CAL_FAILED $m; exit 1; }
done
echo SSN_ALL_DONE
