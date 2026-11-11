#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
$PY scripts/prep_data.py && $PY scripts/gen_versions.py && $PY scripts/build_index.py && $PY scripts/gen_paraphrase.py && echo PREP_DONE
