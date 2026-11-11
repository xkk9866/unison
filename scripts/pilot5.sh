#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison,unison-e0.05,unison-e0.05-peerdraft,unison-e0.1 --n 60 --rate 1.2 --mu 0.3 --tag pilot5 --replay 0 > $L/pilot5.log 2>&1
echo PILOT5_DONE >> $L/pilot5.log
