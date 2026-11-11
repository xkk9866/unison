#!/bin/bash
# Decision-boundary certification (DBC) queue, GPU 3 only, one engine at a time (a co-located tenant
# holds ~8 GB of GPU 3). Progress: logs/queue_v9.log
#  A. main comparison: 3 seeds x 2 datasets, test half, all baselines with the same speculator,
#     both note formats for the lossless baselines, and the earlier estimated-legality design (prefix law)
#  B. ablations of the mechanism
#  C. models
#  D. workload sweeps (edit rate, popularity skew, team size, load, public arrival trace)
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v9.log
run() { tag=$1; shift
  for attempt in 1 2; do
    u=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
    echo "START $tag try$attempt gpu3=$u $* $(date +%T)" >> $Q
    if $PY scripts/run_exp.py --tag $tag "$@" > $L/v9_${tag}_$(date +%s%N).log 2>&1; then echo "DONE $tag $(date +%T)" >> $Q; return; fi
    echo "FAIL $tag try$attempt $(date +%T)" >> $Q; sleep 30
  done; }
U="--gpu_util ${GPU_UTIL:-0.62} --ssn $S/law.pt --qsplit test"
# ---- A
M="noshare@cot,noshare,exact+suffix@cot,exact+suffix,semS0.90,semS0.95,semVS0.90,semcrcS-0.05,law-e0.05@cot,dbc"
for seed in 0 1 2; do
  for ds in hotpotqa musique; do
    run v9_main_s$seed --ds $ds --methods $M --n 200 --rate 1.2 --mu 0.1 --seed $seed $U
  done
done
echo MAIN_DONE >> $Q
# ---- B
for ds in hotpotqa musique; do
  run v9_abl --ds $ds --methods exact+suffix,dbc-nodraft,dbc-single,dbc-note,dbc@cot,dbc --n 200 --rate 1.2 --mu 0.1 --seed 0 $U
done
# ---- C
for m in qwen3-4b qwen2.5-7b; do
  run v9_model_$m --ds hotpotqa --model $m --methods noshare,exact+suffix,semS0.90,semS0.95,dbc --n 200 --rate 1.2 --mu 0.1 --seed 0 $U
done
# ---- D
W="exact+suffix,semS0.90,semVS0.90,dbc"
for mu in 0.5 2.0; do run v9_ver_mu$mu --ds hotpotqa --methods $W --n 200 --rate 1.2 --mu $mu --seed 0 $U; done
for z in 0 1.1; do run v9_zipf_$z --ds hotpotqa --methods $W --n 200 --rate 1.2 --mu 0.1 --zipf $z --seed 0 $U; done
for t in 1 5; do run v9_team_$t --ds hotpotqa --methods $W --n 200 --rate 1.2 --mu 0.1 --team $t --seed 0 $U; done
for r in 0.6 2.4; do run v9_load_r$r --ds hotpotqa --methods $W --n 200 --rate $r --mu 0.1 --seed 0 $U; done
run v9_azure --ds hotpotqa --methods $W --n 200 --rate 1.2 --mu 0.1 --arrival azure --seed 0 $U
echo QUEUE_V9_DONE >> $Q
