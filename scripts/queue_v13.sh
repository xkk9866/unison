#!/bin/bash
# v13 queue (GPU 3 only). Main comparison against published systems only, each run locally from its
# official code: vLLM (APC), Prompt Lookup Decoding (vLLM n-gram), SuffixDecoding (arctic_inference),
# GPTCache (gptcache 0.1.44), vCache (vcache-project/vCache master). Our own design variants appear only in
# the ablation (v12_abl_*). Adaptive to the GPU-3 co-tenant like queue_v12b.sh. Ends with QUEUE_V13_DONE.
cd /data/yaokang/evishare/code
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs; Q=$L/queue_v13.log; RES=/data/yaokang/evishare/results
B13="noshare@cot,pld@cot,suffix@cot,gptcache@cot,vcacheoff@cot,dbc-canon"
W13="suffix@cot,gptcache@cot,vcacheoff@cot,dbc-canon"
A="exact+suffix,dbc-lazy,dbc,dbc-sbd,dbc-sbdcanon,dbc-cont,dbc-canonnofg,dbc-canon"
r() { [ $1 = musique ] && echo 0.6 || echo 1.2; }
jobs() {
  echo "v13_smoke --ds hotpotqa --methods gptcache@cot,vcacheoff@cot --n 20 --rate 1.2 --mu 0.1 --seed 0"
  for seed in 0 1 2; do for ds in hotpotqa musique; do
    echo "v13_main_s$seed --ds $ds --methods $B13 --n 200 --rate $(r $ds) --mu 0.1 --seed $seed --replay2 1"; done; done
  for ds in hotpotqa musique; do
    echo "v12_abl_s1 --ds $ds --methods $A --n 200 --rate $(r $ds) --mu 0.1 --seed 1"; done
  B="--ds hotpotqa --methods $W13 --n 200 --seed 0"
  echo "v13_load_r0.6 $B --rate 0.6 --mu 0.1"
  echo "v13_load_r1.8 $B --rate 1.8 --mu 0.1"
  echo "v13_load_r2.4 $B --rate 2.4 --mu 0.1"
  echo "v13_muload_r1.2 --ds musique --methods $W13 --n 200 --seed 0 --rate 1.2 --mu 0.1"
  echo "v13_ver_mu0.5 $B --rate 1.2 --mu 0.5"
  echo "v13_ver_mu2.0 $B --rate 1.2 --mu 2.0"
  echo "v13_zipf_0 $B --rate 1.2 --mu 0.1 --zipf 0"
  echo "v13_zipf_1.1 $B --rate 1.2 --mu 0.1 --zipf 1.1"
  echo "v13_team_1 $B --rate 1.2 --mu 0.1 --team 1"
  echo "v13_team_5 $B --rate 1.2 --mu 0.1 --team 5"
  echo "v13_azure $B --rate 1.2 --mu 0.1 --arrival azure"
  echo "v13_lowred $B --rate 1.2 --mu 0.1 --zipf 0 --team 1"
  for ds in hotpotqa musique; do  # ablation seed 0 rerun (v12_abl_s0 overlapped the co-tenant)
    echo "v13_abl_s0r --ds $ds --methods $A --n 200 --rate $(r $ds) --mu 0.1 --seed 0"; done
  for m in qwen3-4b qwen2.5-7b; do
    echo "v13_model_$m --ds hotpotqa --model $m --methods $W13 --n 200 --rate 1.2 --mu 0.1 --seed 0"; done
}
[ "$1" = list ] && { jobs; exit; }
while pgrep -f "run_exp.py --tag v1" > /dev/null; do sleep 30; done
jobs > $L/jobs_v13.txt
declare -A tries
mapfile -t todo < $L/jobs_v13.txt
while [ ${#todo[@]} -gt 0 ]; do
  line=${todo[0]}; todo=("${todo[@]:1}")
  tag=${line%% *}; args=${line#* }; ds=$(echo "$args" | sed -E 's/.*--ds ([a-z]+).*/\1/')
  [ -f $RES/$tag/$ds/summary.json ] && continue
  k="$tag/$ds"; tries[$k]=$(( ${tries[$k]:-0} + 1 ))
  [ ${tries[$k]} -gt 10 ] && { echo "GIVEUP $k" >> $Q; continue; }
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
    if [ $used -lt 9800 ]; then sleep 60; u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
      [ $u2 -lt 9800 ] && { used=$(( u2 > used ? u2 : used )); break; }; fi; sleep 30; done
  util=$(python3 -c "print(min(0.90, round(($used + 13800) / 24120, 2)))")
  echo "START $tag try${tries[$k]} $ds util=$util others=$used $(date +%T)" >> $Q
  if $PY scripts/run_exp.py --tag $tag $args --gpu_util $util --qsplit test --note dfp > $L/${tag}_${ds}_$(date +%s).log 2>&1; then
    echo "DONE $tag $ds $(date +%T)" >> $Q
  else
    echo "FAIL $tag $ds try${tries[$k]} $(date +%T)" >> $Q
    [ $tag = v13_smoke ] && { echo "SMOKE_FAILED" >> $Q; echo QUEUE_V13_DONE >> $Q; exit 1; }
    todo+=("$line"); sleep 20
  fi
done
echo QUEUE_V13${QSUF}_DONE >> $Q
