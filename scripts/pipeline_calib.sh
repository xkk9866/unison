#!/bin/bash
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 PYTHONUNBUFFERED=1 SPLITS=calib
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
$PY scripts/prep_calib.py > $L/calib_prep.log 2>&1 && $PY scripts/gen_versions.py >> $L/calib_prep.log 2>&1 && \
$PY scripts/build_index.py >> $L/calib_prep.log 2>&1 && $PY scripts/gen_paraphrase.py >> $L/calib_prep.log 2>&1 || { echo CALIB_PREP_FAILED; exit 1; }
for ds in hotpotqa musique; do
  $PY scripts/run_exp.py --ds $ds --split calib --methods noshare --n 300 --rate 2.0 --zipf 0.8 --mu 0.5 --replay 0 --tag calibdata > $L/calibdata_$ds.log 2>&1 || { echo CALIBDATA_FAILED; exit 1; }
done
CALIB_TAG=calibdata CALIB_SPLIT=calib $PY scripts/calibrate_deploy.py > $L/calib_deploy.log 2>&1 || { echo CALIBRATE_FAILED; exit 1; }
$PY scripts/run_exp.py --ds hotpotqa --methods noshare,exact+suffix,semcrc-0.1,unison,unison-e0.02,unison-e0.05 --n 40 --rate 2.0 --mu 0.2 --tag pilot2 > $L/pilot2.log 2>&1
echo PIPE2_DONE
