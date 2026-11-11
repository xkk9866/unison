#!/bin/bash
# waits for the training-data collection, then builds pairs, trains the AHN and runs a pilot
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
until grep -q ALLDONE /data/yaokang/evishare/logs/ahndata.log; do sleep 10; done
$PY scripts/build_pairs.py > /data/yaokang/evishare/logs/pairs.log 2>&1 || { echo PAIRS_FAILED; exit 1; }
$PY scripts/train_ahn.py > /data/yaokang/evishare/logs/train_ahn.log 2>&1 || { echo TRAIN_FAILED; exit 1; }
$PY scripts/run_exp.py --ds hotpotqa --methods noshare,exact+suffix,semcrc-0.05,unison,unison-e0.05 --n 40 --rate 2.0 --mu 0.2 --tag pilot > /data/yaokang/evishare/logs/pilot.log 2>&1
echo PIPE_DONE
