#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
until grep -qE 'SSN_ALL_DONE|FAILED' $L/pipe_ssn.log; do sleep 10; done
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison,unison-e0.05,unison-e0.1 --n 60 --rate 1.2 --mu 0.3 --tag pilot4 > $L/pilot4.log 2>&1
echo PILOT4_DONE >> $L/pilot4.log
