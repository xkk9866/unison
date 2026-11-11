#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
O=/data/yaokang/evishare/results/ssn
$PY scripts/build_replay.py --tag ahndata --split train --out $O/train_qwen3-8b.pkl.gz || exit 1
$PY scripts/build_replay.py --tag calibdata --split calib --text_baselines 1 --out $O/calib_qwen3-8b.pkl.gz || exit 1
echo REPLAY_DATA_DONE
