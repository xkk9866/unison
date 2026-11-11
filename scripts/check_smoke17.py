"""Invariants of the v17 smoke run (u17 and its variants on 20 queries). Prints SMOKE_OK or exits 1.
usage: python check_smoke17.py <summary.json> <run log>"""
import json
import sys

S = {r["method"]: r for r in json.load(open(sys.argv[1]))}
log = open(sys.argv[2], errors="replace").read()
bad = []


def need(cond, msg):
    if not cond:
        bad.append(msg)


FLOOR = 0.97

for m in ("u17", "u17-nopre", "u17-nopeer", "u17-sbd", "u17-noid"):
    need(m in S and S[m]["n_queries"] == 20, f"{m}: missing or incomplete")
    r = S.get(m, {})
    need(r.get("prov_requests", 0) > 0, f"{m}: no provenance produced")
    need(r.get("n_acc", 0) > 0, f"{m}: no draft accounting from the plugin (acc log missing)")
    need(r.get("nerr_dcert", 0) == 0, f"{m}: certified decision differs from re-execution")
    need(r.get("n_df_mismatch", 0) == 0, f"{m}: determined team answer differs from the full aggregation")
    # exactness: every delivered sequence is a prefix of the call's independent re-execution (a key ends at its
    # boundary). Two executions of one prompt in different batches differ in a small share of calls on this engine
    # (floating-point non-determinism, measured by the replay floor of the main runs), so the gate is not 1.0;
    # the drafted systems must be as close to the re-execution as the SuffixDecoding variant of the same run.
    need(r.get("fidelity_prefix_verify", 0) >= FLOOR and r.get("fidelity_prefix_plan", 0) >= FLOOR,
         f"{m}: delivered tokens are not a prefix of the independent re-execution "
         f"(verify {r.get('fidelity_prefix_verify')}, plan {r.get('fidelity_prefix_plan')})")
u, s, n, p = S.get("u17", {}), S.get("u17-sbd", {}), S.get("u17-noid", {}), S.get("u17-nopre", {})
need(u.get("acc_nd", 0) + u.get("acc_pre", 0) > 0, "u17: no verification key token accepted from the consensus drafter")
need(u.get("pacc_nd", 0) > 0, "u17: no plan token accepted from the consensus drafter")
need(u.get("acc_pre", 0) > 0, "u17: no verification key token accepted from a draft made before the first pass")
need(u.get("key_one_pass", 0) > 0, "u17: no verification call completed in one pass")
need(p.get("acc_pre", 0) + p.get("pacc_pre", 0) == 0, "u17-nopre: drafts made before the first pass")
need(p.get("acc_nd", 0) > 0, "u17-nopre: no key token accepted from the consensus drafter")
need(s.get("acc_nd", 0) + s.get("acc_pre", 0) == 0, "u17-sbd: consensus drafts used by the self-key baseline")
need(u.get("prov_engine", 0) <= u.get("prov_requests", 0), "u17: provenance accounting")
need(n.get("src_exact", 0) == 0 and n.get("n_plan_exact", 0) == 0, "u17-noid: identity sharing still active")
need(u.get("src_exact", 0) > 0, "u17: no identity-shared verification call")
need(abs(u.get("em", 0) - s.get("em", 0)) <= 0.25, "u17 vs u17-sbd: EM differ by more than noise on 20 queries")
for pat in ("[unison] priority error", "[unison] draft error", "[unison] consensus draft error",
            "[unison] pre-draft error", "Traceback"):
    need(pat not in log, f"log contains '{pat}'")
if bad:
    print("SMOKE_BAD", *bad, sep="\n  ")
print("SMOKE_OK" if not bad else "SMOKE_VALUES",
      json.dumps({m: dict(lat=round(r["lat_mean"], 2), p95=round(r.get("lat_p95", 0.0), 2), iters=r.get("engine_iters"),
                          busy=round(r.get("engine_busy", 0.0), 1),
                          key_passes=round(r.get("key_passes", 0.0), 2), key_one=round(r.get("key_one_pass", 0.0), 3),
                          plan_passes=round(r.get("plan_passes", 0.0), 2), plan_one=round(r.get("plan_one_pass", 0.0), 3),
                          pre=r.get("acc_pre"), nd=r.get("acc_nd"), ppre=r.get("pacc_pre"), pnd=r.get("pacc_nd"),
                          drafted=r.get("acc_drafted"), pdrafted=r.get("pacc_drafted"),
                          plan_exact=r.get("n_plan_exact"), exact=r.get("src_exact"),
                          fid=r.get("fidelity_prefix_verify"), fid_plan=r.get("fidelity_prefix_plan"),
                          em=r.get("em"), nd_s=round(r.get("nd_seconds", 0.0), 2), price=round(r.get("nd_price", 0.0), 3),
                          sprice=round(r.get("nd_sprice", 0.0), 3), iter_ms=round(r.get("nd_iter_ms", 0.0), 1),
                          others=round(r.get("gpu_others_sm_mean", 0.0)))
                  for m, r in S.items()}))
sys.exit(1 if bad else 0)
