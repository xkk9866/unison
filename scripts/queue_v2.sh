#!/bin/bash
# Full experiment queue for consumer-conditioned replay (CCR). GPU 3 only, sequential so that latency
# measurements are not perturbed by co-located jobs. Progress: logs/queue_v2.log
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
R=1.2
until grep -q SSN_ALL_DONE $L/pipe_ssn.log; do sleep 20; done
run() { tag=$1; shift; echo "START $tag $(date +%T)" >> $L/queue_v2.log
        $PY scripts/run_exp.py --tag $tag "$@" > $L/v2_${tag}_$(date +%s).log 2>&1 && echo "DONE $tag $(date +%T)" >> $L/queue_v2.log || echo "FAIL $tag $(date +%T)" >> $L/queue_v2.log; }
MAIN="noshare,exact,suffix,exact+suffix,sem0.90,semcrc-0.05,osc-e0.05,unison,unison-e0.02,unison-e0.05,unison-e0.1"
run v2_main --ds hotpotqa --methods pld,$MAIN --n 200 --rate $R --mu 0.1
run v2_main --ds musique --methods $MAIN --n 200 --rate $R --mu 0.1
ABL="unison-e0.05,unison-e0.05-noreplay,unison-e0.05-norace,unison-e0.05-onecand,unison-e0.05-nolaw,unison-e0.05-nospec,osc-e0.05"
run v2_abl --ds hotpotqa --methods noshare,$ABL --n 150 --rate $R --mu 0.5
run v2_abl --ds musique --methods noshare,$ABL --n 150 --rate $R --mu 0.5
SWEEP="noshare,exact+suffix,sem0.90,osc-e0.05,unison,unison-e0.05"
for mu in 0 0.5 1.0 2.0; do run v2_ver_mu$mu --ds hotpotqa --methods noshare,exact,sem0.90,semV0.90,osc-e0.05,unison,unison-e0.05 --n 150 --rate $R --mu $mu; done
for r in 0.6 1.8 2.4; do run v2_load_r$r --ds hotpotqa --methods $SWEEP --n 150 --rate $r --mu 0.1; done
for m in qwen3-4b qwen2.5-7b; do run v2_model_$m --ds hotpotqa --model $m --ssn $S/ssn_$m.pt --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --mu 0.1; done
for z in 0 0.5 1.1; do run v2_zipf_$z --ds musique --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --zipf $z --mu 0.1; done
for t in 1 5; do run v2_team_$t --ds hotpotqa --methods noshare,exact+suffix,sem0.90,unison,unison-e0.05 --n 150 --rate $R --team $t --mu 0.1; done
run v2_azure --ds hotpotqa --methods $SWEEP --n 200 --rate $R --arrival azure --mu 0.1
echo QUEUE_V2_DONE >> $L/queue_v2.log
