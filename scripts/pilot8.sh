#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
$PY scripts/fix_meta.py $S/ssn.pt $S/ssn_qwen3-4b.pt $S/ssn_qwen2.5-7b.pt > $L/pilot8.log 2>&1
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison,unison-e0.05 --n 100 --rate 2.0 --mu 0.3 --tag pilot8 >> $L/pilot8.log 2>&1
echo PILOT8_DONE >> $L/pilot8.log
