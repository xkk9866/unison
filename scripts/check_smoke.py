"""Invariants of the v14 smoke run (u14, u14-sbd, u14-noprio, lacache on 20 queries). Prints SMOKE_OK or exits 1.
usage: python check_smoke.py <summary.json> <run log>"""
import json
import sys

S = {r["method"]: r for r in json.load(open(sys.argv[1]))}
log = open(sys.argv[2], errors="replace").read()
bad = []


def need(cond, msg):
    if not cond:
        bad.append(msg)


for m in ("u14", "u14-sbd", "u14-noprio", "lacache"):
    need(m in S and S[m]["n_queries"] == 20, f"{m}: missing or incomplete")
for m in ("u14", "u14-sbd", "u14-noprio"):
    r = S.get(m, {})
    need(r.get("prov_requests", 0) > 0, f"{m}: no provenance produced")
    need(r.get("n_acc", 0) > 0, f"{m}: no draft accounting from the plugin (acc log missing)")
    need(r.get("nerr_dcert", 0) == 0, f"{m}: certified decision differs from re-execution")
u, b = S.get("u14", {}), S.get("u14-sbd", {})
need(u.get("acc_trie", 0) + u.get("acc_live", 0) > 0, "u14: no key token was accepted from a peer draft")
need(b.get("acc_trie", 0) + b.get("acc_live", 0) == 0, "u14-sbd: peer drafts used by the self-key baseline")
need(u.get("prov_engine", 0) <= u.get("prov_requests", 0), "u14: provenance accounting")
for pat in ("[unison] priority error", "[unison] draft error", "Traceback"):
    need(pat not in log, f"log contains '{pat}'")
if bad:
    print("SMOKE_BAD", *bad, sep="\n  ")
    sys.exit(1)
print("SMOKE_OK", json.dumps({m: dict(lat=round(r["lat_mean"], 2), prov=r.get("prov_requests"),
                                      prov_eng=r.get("prov_engine"), live=r.get("acc_live"), trie=r.get("acc_trie"))
                              for m, r in S.items()}))
