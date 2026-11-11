"""Add thresholds_by to already-trained SSN checkpoints."""
import sys
import torch
for p in sys.argv[1:]:
    st = torch.load(p, map_location="cpu")
    m = st["__meta__"]
    m.setdefault("thresholds_by", {"offline": m["thresholds"], "survival": m["alt_thresholds"]["survival"],
                                   "deploy": m["thresholds"]})
    torch.save(st, p)
    print(p, m["thresholds_by"])
