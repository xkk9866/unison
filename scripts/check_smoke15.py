"""Invariants of the v15 smoke run (u15, u15-sbd, u15-noid, lacache on 20 queries). Prints SMOKE_OK or exits 1.
usage: python check_smoke15.py <summary.json> <run log>"""
import json
import sys

S = {r["method"]: r for r in json.load(open(sys.argv[1]))}
log = open(sys.argv[2], errors="replace").read()
bad = []


def need(cond, msg):
    if not cond:
        bad.append(msg)


for m in ("u15", "u15-sbd", "u15-noid", "lacache"):
    need(m in S and S[m]["n_queries"] == 20, f"{m}: missing or incomplete")
for m in ("u15", "u15-sbd", "u15-noid"):
    r = S.get(m, {})
    need(r.get("prov_requests", 0) > 0, f"{m}: no provenance produced")
    need(r.get("n_acc", 0) > 0, f"{m}: no draft accounting from the plugin (acc log missing)")
    need(r.get("nerr_dcert", 0) == 0, f"{m}: certified decision differs from re-execution")
    need(r.get("n_df_mismatch", 0) == 0, f"{m}: determined team answer differs from the full aggregation")
    need(r.get("df_boundaries", 0) == 1, f"{m}: dataflow boundaries not active")
u, b, n = S.get("u15", {}), S.get("u15-sbd", {}), S.get("u15-noid", {})
need(u.get("acc_trie", 0) + u.get("acc_live", 0) > 0, "u15: no key token was accepted from a peer draft")
need(b.get("acc_trie", 0) + b.get("acc_live", 0) == 0, "u15-sbd: peer drafts used by the self-key baseline")
need(u.get("prov_engine", 0) <= u.get("prov_requests", 0), "u15: provenance accounting")
need(n.get("src_exact", 0) == 0 and n.get("n_plan_exact", 0) == 0, "u15-noid: identity sharing still active")
need(u.get("src_exact", 0) > 0, "u15: no identity-shared verification call")
need(abs(u.get("em", 0) - b.get("em", 0)) <= 0.25, "u15 vs u15-sbd: EM differ by more than noise on 20 queries")
for pat in ("[unison] priority error", "[unison] draft error", "Traceback"):
    need(pat not in log, f"log contains '{pat}'")
if bad:
    print("SMOKE_BAD", *bad, sep="\n  ")
    sys.exit(1)
print("SMOKE_OK", json.dumps({m: dict(lat=round(r["lat_mean"], 2), lat_all=round(r.get("lat_all_mean", 0), 2),
                                      prov=r.get("prov_requests"), prov_eng=r.get("prov_engine"),
                                      live=r.get("acc_live"), trie=r.get("acc_trie"), plan_exact=r.get("n_plan_exact"),
                                      team_early=r.get("n_team_early"), round_early=r.get("n_round_early"))
                              for m, r in S.items()}))
