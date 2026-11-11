#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
$PY scripts/run_exp.py --ds hotpotqa --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05,unison-e0.1 --n 100 --rate 2.0 --mu 0.3 --tag pilot6 --replay 1 > $L/pilot6.log 2>&1
echo PILOT6_DONE >> $L/pilot6.log
