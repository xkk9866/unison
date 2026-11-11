#!/bin/bash
# Per-workload shadow recalibration (decision-equivalence labels), validation pilot, then the full queue.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
for ds in hotpotqa musique; do
  $PY scripts/calib_shadow.py --runs calibshadow/$ds --method unison-shadow --ssn $S/law.pt --workload $ds > $L/recal_$ds.log 2>&1 || { echo RECAL_FAILED; exit 1; }
done
for m in qwen3-4b qwen2.5-7b; do
  $PY scripts/calib_shadow.py --runs calibshadow_$m/hotpotqa --method unison-shadow --ssn $S/law_$m.pt --workload hotpotqa > $L/recal_$m.log 2>&1 || { echo RECAL_FAILED; exit 1; }
done
echo RECAL_DONE
$PY scripts/run_exp.py --ds hotpotqa --methods exact+suffix,unison-e0.05,unison-e0.1 --n 100 --rate 1.2 --mu 0.3 --tag pilot12 --ssn $S/law.pt > $L/pilot12.log 2>&1
echo PILOT12_DONE >> $L/pilot12.log
bash scripts/queue_v3.sh
