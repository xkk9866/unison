#!/bin/bash
# Full experiment queue on GPU 2 (sequential). Progress lines go to logs/queue_progress.log.
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=2 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
R=1.2
run() { tag=$1; shift; echo "START $tag $(date +%T)" >> $L/queue_progress.log
        $PY scripts/run_exp.py --tag $tag "$@" > $L/q_${tag}_$(date +%s).log 2>&1 && echo "DONE $tag $(date +%T)" >> $L/queue_progress.log || echo "FAIL $tag $(date +%T)" >> $L/queue_progress.log; }
MAIN="noshare,exact,pld,suffix,exact+suffix,unison,sem0.90,sem0.95,semcrc-0.05,semcrc-0.1,unison-e0.02,unison-e0.05,unison-e0.1"
SWEEP="noshare,exact+suffix,sem0.90,semcrc-0.1,unison,unison-e0.05"
run main --ds hotpotqa --methods $MAIN --n 200 --rate $R --mu 0.1
run main --ds musique --methods $MAIN --n 200 --rate $R --mu 0.1
ABL="unison-e0.05,unison-e0.05-cosine,unison-e0.05-nospec,unison-e0.05-noinflight,unison-e0.05-nojoin,unison-e0.05-fulldraft,unison-e0.05-nodelta"
run abl --ds hotpotqa --methods noshare,$ABL --n 150 --rate $R --mu 0.5
run abl --ds musique --methods noshare,$ABL --n 150 --rate $R --mu 0.5
for r in 0.6 1.8 2.4; do run load_r$r --ds hotpotqa --methods $SWEEP --n 150 --rate $r --mu 0.1; done
for mu in 0 0.5 1.0 2.0; do run ver_mu$mu --ds hotpotqa --methods noshare,exact,sem0.90,semV0.90,unison,unison-e0.05 --n 150 --rate $R --mu $mu; done
for z in 0 0.5 1.1; do run zipf_$z --ds musique --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --zipf $z --mu 0.1; done
for t in 1 5; do run team_$t --ds hotpotqa --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --team $t --mu 0.1; done
run azure --ds hotpotqa --methods $SWEEP --n 200 --rate $R --arrival azure --mu 0.1
run model_qwen3-4b --ds hotpotqa --model qwen3-4b --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --mu 0.1
run model_qwen2.5-7b --ds hotpotqa --model qwen2.5-7b --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --mu 0.1
echo QUEUE_ALL_DONE >> $L/queue_progress.log
