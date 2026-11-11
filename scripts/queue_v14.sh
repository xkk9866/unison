#!/bin/bash
# v14 queue (GPU 3 only, single seed). Exactly the runs that appear in paper/unison14:
#   smoke  -> check_smoke.py must print SMOKE_OK, else the queue stops
#   main   (Table 2)  7 baselines + Unison + answer-last reference, 2 datasets, + non-determinism floor
#   abl    (Table 3)  2x2 factorization + 2 component removals + decision-only bound, 2 datasets
#   load   (Fig. 3, Table 4)  HotpotQA at 0.6/1.8/2.4/3.0 q/s (1.2 = main)
#   edit   (Table 5)  decisive edits (versions_edit.jsonl), 1 edit/s
#   audit  (Table 6)  offline judge on v14_abl provenance
#   model  (Table 7, optional: MODELS=1)  Qwen2.5-7B and Llama-3.1-8B
# A job starts only when GPU 3 is compute-idle (utilization < 10% over 60 s while we hold nothing);
# jobs with a summary.json are skipped (resume-safe). Log: logs/queue_v14.log, end marker QUEUE_V14_DONE.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 UNISON_PRIO=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs; Q=$L/queue_v14.log; RES=/data/yaokang/evishare/results
MAIN="noshare,pld,suffix,gptcache,vcacheoff,lacache,u14-sbd,u14,noshare@cot"
ABL="u14-lazy,u14-sbd,u14-coop,u14-keyed,u14,u14-nolive,u14-noprio"
LOADM="suffix,gptcache,u14-sbd,u14"
EDITM="gptcache,vcacheoff,lacache,u14-sbd,u14"
COMMON="--n 200 --seed 0 --qsplit test --note dfp"
r() { [ $1 = musique ] && echo 0.6 || echo 1.2; }
jobs() {
  echo "v14_smoke --ds hotpotqa --methods u14,u14-sbd,u14-noprio,lacache --n 20 --rate 1.2 --mu 0.1 --seed 0 --qsplit test --note dfp"
  for ds in hotpotqa musique; do
    echo "v14_main --ds $ds --methods $MAIN --rate $(r $ds) --mu 0.1 --replay2 1 $COMMON"; done
  for ds in hotpotqa musique; do
    echo "v14_abl --ds $ds --methods $ABL --rate $(r $ds) --mu 0.1 $COMMON"; done
  for rate in 0.6 1.8 2.4 3.0; do
    echo "v14_load_r$rate --ds hotpotqa --methods $LOADM --rate $rate --mu 0.1 $COMMON"; done
  echo "v14_edit --ds hotpotqa --methods $EDITM --rate 1.2 --mu 1.0 --versions versions_edit.jsonl $COMMON"
  if [ "$MODELS" = 1 ]; then for m in qwen2.5-7b llama3.1-8b; do
    echo "v14_model_$m --ds hotpotqa --model $m --methods $LOADM --rate 1.2 --mu 0.1 $COMMON"; done; fi
}
[ "$1" = list ] && { jobs; exit; }
gpu_idle() {  # 60 s of utilization samples on GPU 3 below 10% (co-tenant memory alone is tolerated)
  for i in $(seq 1 12); do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3)
    [ "$u" -ge 10 ] && return 1; sleep 5; done; return 0; }
for ds in hotpotqa musique; do  # decisive-edit versions for the stress test (idempotent)
  [ -f /data/yaokang/evishare/data/$ds/eval/versions_edit.jsonl ] || $PY scripts/gen_edits.py $ds >> $Q 2>&1; done
jobs > $L/jobs_v14.txt
declare -A tries
mapfile -t todo < $L/jobs_v14.txt
while [ ${#todo[@]} -gt 0 ]; do
  line=${todo[0]}; todo=("${todo[@]:1}")
  tag=${line%% *}; args=${line#* }; ds=$(echo "$args" | sed -E 's/.*--ds ([a-z]+).*/\1/')
  [ -f $RES/$tag/$ds/summary.json ] && continue
  k="$tag/$ds"; tries[$k]=$(( ${tries[$k]:-0} + 1 ))
  [ ${tries[$k]} -gt 6 ] && { echo "GIVEUP $k" >> $Q; continue; }
  until gpu_idle; do echo "WAIT busy GPU3 $(date +%T)" >> $Q; sleep 120; done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  util=$(python3 -c "print(min(0.90, round(($used + 13800) / 24120, 2)))")
  log=$L/${tag}_${ds}_$(date +%s).log
  echo "START $tag try${tries[$k]} $ds util=$util others=$used $(date +%T) log=$log" >> $Q
  if $PY scripts/run_exp.py --tag $tag $args --gpu_util $util > $log 2>&1; then
    echo "DONE $tag $ds $(date +%T)" >> $Q
    if [ $tag = v14_smoke ]; then
      if ! $PY scripts/check_smoke.py $RES/v14_smoke/hotpotqa/summary.json $log >> $Q 2>&1; then
        echo "SMOKE_FAILED (see $log)" >> $Q; echo QUEUE_V14_DONE >> $Q; exit 1; fi; fi
  else
    echo "FAIL $tag $ds try${tries[$k]} $(date +%T)" >> $Q
    [ $tag = v14_smoke ] && { echo "SMOKE_FAILED (see $log)" >> $Q; echo QUEUE_V14_DONE >> $Q; exit 1; }
    todo+=("$line"); sleep 30
  fi
done
for ds in hotpotqa musique; do  # audit quality of provenance (offline judge, different model generation)
  out=$RES/v14_abl/$ds/audit_u14_vs_u14-sbd.json
  [ -f $out ] || { until gpu_idle; do sleep 120; done
    $PY scripts/audit_eval.py --tag v14_abl --ds $ds --a u14 --b u14-sbd --n 400 > $L/v14_audit_$ds.log 2>&1 \
      && echo "DONE audit $ds" >> $Q || echo "FAIL audit $ds" >> $Q; }
done
echo QUEUE_V14_DONE >> $Q
