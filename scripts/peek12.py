"""Condensed view of a run log (FINAL/FLOOR lines) or summary.json files. usage: peek12.py <log|summary.json>..."""
import json
import sys

K = [("lat", "lat_mean"), ("p99", "lat_p99"), ("vlat", "vlat_mean"), ("em", "em"), ("busy", "engine_busy"),
     ("busyfg", "busy_fg"), ("gen", "gen_tokens"), ("draft", "draft_tokens"), ("provR", "prov_requests"),
     ("provE", "prov_engine"), ("provT", "prov_tokens"), ("plag", "prov_lag_mean"), ("drain", "drain_s"),
     ("dcert", "src_dcert"), ("exact", "src_exact"), ("cert", "src_certified"), ("sem", "src_sem"),
     ("err", "nerr_reused"), ("stale", "stale_reuse"), ("dz", "stale_decisive"), ("dzerr", "stale_decisive_err"),
     ("nd", "nd_decision_diff"), ("ndn", "nd_prompts"), ("oth", "gpu_others_mib"), ("sv", "support_valid")]


def show(d):
    out = [d["method"][:22].ljust(22)]
    for a, k in K:
        v = d.get(k)
        if v is None:
            continue
        out.append(f"{a}={v:.3f}" if isinstance(v, float) else f"{a}={v}")
    print(" ".join(out))


for p in sys.argv[1:]:
    print("==", p)
    if p.endswith(".json"):
        for d in json.load(open(p)):
            show(d)
        continue
    for l in open(p):
        if l.startswith("FINAL "):
            show(json.loads(l[6:]))
        elif l.startswith("FLOOR "):
            print(l.strip())
