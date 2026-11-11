#!/bin/bash
# GPU 3 only: train the anticipatory law, transfer-calibrate it to the other models, pilot it,
# then run deployment (shadow) calibration for all three models.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
$PY scripts/train_law.py --out $S/law.pt > $L/train_law.log 2>&1 || { echo LAW_FAILED; exit 1; }
H=$($PY -c "import json;print(json.load(open('$S/law_meta.json'))['horizon'])")
for m in qwen3-4b qwen2.5-7b; do
  $PY scripts/train_law.py --load $S/law.pt --calib $S/calib_$m.pkl.gz --out $S/law_$m.pt --horizon $H > $L/law_$m.log 2>&1 || { echo LAWM_FAILED; exit 1; }
done
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison-e0.05,unison-e0.1 --n 100 --rate 1.2 --mu 0.3 --tag pilot10 --ssn $S/law.pt > $L/pilot10.log 2>&1
echo PILOT10_DONE >> $L/pilot10.log
for ds in hotpotqa musique; do
  $PY scripts/run_exp.py --ds $ds --split calib --methods unison-shadow --ssn $S/law.pt --n 300 --rate 1.6 --zipf 0.8 --mu 0.5 --replay 0 --tag calibshadow > $L/calibshadow_$ds.log 2>&1 || { echo SHADOW_FAILED; exit 1; }
done
$PY scripts/calib_shadow.py --runs calibshadow/hotpotqa,calibshadow/musique --method unison-shadow --ssn $S/law.pt > $L/calib_shadow.log 2>&1 || { echo CALSHADOW_FAILED; exit 1; }
for m in qwen3-4b qwen2.5-7b; do
  $PY scripts/run_exp.py --ds hotpotqa --split calib --model $m --ssn $S/law_$m.pt --methods unison-shadow --n 300 --rate 1.6 --zipf 0.8 --mu 0.5 --replay 0 --tag calibshadow_$m > $L/calibshadow_$m.log 2>&1 || { echo SHADOW_FAILED $m; exit 1; }
  $PY scripts/calib_shadow.py --runs calibshadow_$m/hotpotqa --method unison-shadow --ssn $S/law_$m.pt > $L/calib_shadow_$m.log 2>&1 || { echo CALSHADOW_FAILED $m; exit 1; }
done
echo LAW_ALL_DONE
