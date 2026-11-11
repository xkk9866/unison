#!/bin/bash
# Final experiment queue (GPU 3 only).
#  Phase 1 (solo, gpu_util 0.78): headline main table on both datasets -- latency measured without co-runners.
#  Phase 2 (two concurrent lanes on GPU 3, gpu_util 0.42 each): ablations and sweeps. Latencies inside one
#          run are comparable (all methods of a run share the same co-runner regime); compare across
#          lanes only through ratios to the in-run NoShare / Exact+Suffix.
# Progress: logs/queue_v3.log ; results: results/<tag>/<ds>/summary.json
cd /data/yaokang/evishare/code
export CUDA_VISIBLE_DEVICES=3 VLLM_USE_V1=1 PYTHONUNBUFFERED=1
PY=/data/yaokang/evishare/.venv/bin/python
L=/data/yaokang/evishare/logs
S=/data/yaokang/evishare/results/ssn
Q=$L/queue_v3.log
run() { tag=$1; shift; echo "START $tag $* $(date +%T)" >> $Q
        $PY scripts/run_exp.py --tag $tag "$@" > $L/v3_${tag}_$(date +%s%N).log 2>&1 && echo "DONE $tag $(date +%T)" >> $Q || echo "FAIL $tag $(date +%T)" >> $Q; }
R=1.2
MAIN="noshare,exact,suffix,exact+suffix,sem0.90,sem0.95,semcrc-0.05,osc-e0.05,unison-e0.02,unison-e0.05,unison-e0.1"
# ---------------- phase 1
run v3_main --ds hotpotqa --methods pld,$MAIN --n 200 --rate $R --mu 0.1 --ssn $S/law.pt
run v3_main --ds musique --methods $MAIN --n 200 --rate $R --mu 0.1 --ssn $S/law.pt
echo PHASE1_DONE >> $Q
# ---------------- phase 2
G="--gpu_util 0.42 --ssn $S/law.pt"
SWEEP="noshare,exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.1"
laneA() {
  ABL="exact+suffix,sem0.90,osc-e0.05,unison-e0.05,unison-e0.05-latest,unison-e0.05-nolaw"
  run v3_abl --ds hotpotqa --methods noshare,$ABL --n 150 --rate $R --mu 0.5 $G
  run v3_abl --ds musique --methods noshare,$ABL --n 150 --rate $R --mu 0.5 $G
  for r in 0.6 1.8 2.4; do run v3_load_r$r --ds hotpotqa --methods $SWEEP --n 150 --rate $r --mu 0.1 $G; done
  run v3_azure --ds hotpotqa --methods $SWEEP --n 200 --rate $R --arrival azure --mu 0.1 $G
  echo LANE_A_DONE >> $Q
}
laneB() {
  for mu in 0 0.5 1.0 2.0; do run v3_ver_mu$mu --ds hotpotqa --methods noshare,exact,sem0.90,semV0.90,osc-e0.05,unison-e0.05 --n 150 --rate $R --mu $mu $G; done
  for m in qwen3-4b qwen2.5-7b; do run v3_model_$m --ds hotpotqa --model $m --methods noshare,exact+suffix,sem0.90,unison-e0.05,unison-e0.1 --n 150 --rate $R --mu 0.1 --gpu_util 0.42 --ssn $S/law_$m.pt; done
  for z in 0 0.5 1.1; do run v3_zipf_$z --ds musique --methods noshare,exact+suffix,sem0.90,unison-e0.05 --n 150 --rate $R --zipf $z --mu 0.1 $G; done
  for t in 1 5; do run v3_team_$t --ds hotpotqa --methods noshare,exact+suffix,sem0.90,unison-e0.05 --n 150 --rate $R --team $t --mu 0.1 $G; done
  echo LANE_B_DONE >> $Q
}
laneA &
sleep 90
laneB &
wait
echo QUEUE_V3_DONE >> $Q
