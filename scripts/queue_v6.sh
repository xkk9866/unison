#!/bin/bash
# Consolidated re-run of every sweep, sequential on GPU 3. vLLM 0.8.5 counts other processes' memory
# against gpu_memory_utilization; with another user's ~7 GB job resident on GPU 3, 0.90 leaves our
# engine ~7 GB of KV cache (0.42 / 0.65 left too little: init failure or the cache_full_blocks assert).
# One automatic retry per run. Progress: logs/queue_v6.log
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v6.log
run() { tag=$1; shift
  for attempt in 1 2; do
    echo "START $tag try$attempt $* $(date +%T)" >> $Q
    if $PY scripts/run_exp.py --tag $tag "$@" > $L/v6_${tag}_$(date +%s%N).log 2>&1; then echo "DONE $tag $(date +%T)" >> $Q; return; fi
    echo "FAIL $tag try$attempt $(date +%T)" >> $Q
    sleep 20
  done; }
R=1.2
U="--gpu_util 0.90"
G="--ssn $S/law.pt $U"
SWEEP="noshare,exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.1"
ABL="exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.05-latest,unison-e0.05-nolaw"
run v3_abl --ds hotpotqa --methods noshare,$ABL --n 150 --rate $R --mu 0.5 $G
run v3_abl --ds musique --methods noshare,$ABL --n 150 --rate $R --mu 0.5 $G
for r in 0.6 1.2 1.8 2.4; do run v3_load_r$r --ds hotpotqa --methods $SWEEP --n 150 --rate $r --mu 0.1 $G; done
for mu in 0 0.5 1.0 2.0; do run v3_ver_mu$mu --ds hotpotqa --methods noshare,exact,sem0.90,semV0.90,osc-e0.05,unison-e0.05 --n 150 --rate $R --mu $mu $G; done
for m in qwen3-4b qwen2.5-7b; do run v3_model_$m --ds hotpotqa --model $m --methods noshare,exact+suffix,sem0.90,unison-e0.05,unison-e0.1 --n 150 --rate $R --mu 0.1 --ssn $S/law_$m.pt $U; done
for z in 0 0.5 1.1; do run v3_zipf_$z --ds musique --methods noshare,exact+suffix,sem0.90,unison-e0.05 --n 150 --rate $R --zipf $z --mu 0.1 $G; done
for t in 1 5; do run v3_team_$t --ds hotpotqa --methods noshare,exact+suffix,sem0.90,unison-e0.05 --n 150 --rate $R --team $t --mu 0.1 $G; done
run v3_azure --ds hotpotqa --methods $SWEEP --n 200 --rate $R --arrival azure --mu 0.1 $G
echo QUEUE_V6_DONE >> $Q
