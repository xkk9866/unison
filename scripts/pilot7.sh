#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison-shadow,unison-e0.05 --n 100 --rate 2.0 --mu 0.3 --tag pilot7 --replay 0 > $L/pilot7.log 2>&1
echo PILOT7_DONE >> $L/pilot7.log
