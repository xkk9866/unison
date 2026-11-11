#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,sem0.90,unison-e0.05,unison-e0.1 --n 150 --rate 1.2 --mu 0.3 --tag pilot9 --replay 1 > $L/pilot9.log 2>&1
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison-e0.05 --n 100 --rate 2.0 --mu 0.3 --seed 1 --tag pilot9b --replay 0 >> $L/pilot9.log 2>&1
echo PILOT9_DONE >> $L/pilot9.log
