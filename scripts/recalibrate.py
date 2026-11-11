"""Re-run risk calibration from saved calibration scores and update the AHN checkpoint metadata."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from train_ahn import EPS, calibrate
from unison.common import RESULTS

d = np.load(os.path.join(RESULTS, "ahn", "calib_scores.npz"))
q, cos, agree = d["q"], d["cos"], d["agree"]
thr = {f"{e:.3f}": calibrate(q, agree, e) for e in EPS}
cthr = {f"{e:.3f}": calibrate(cos, agree, e) for e in EPS}
cover = {k: dict(ahn=float((q >= thr[k]).mean()), cos=float((cos >= cthr[k]).mean()),
                 ahn_err=float(1 - agree[q >= thr[k]].mean()) if (q >= thr[k]).any() else 0.0,
                 cos_err=float(1 - agree[cos >= cthr[k]].mean()) if (cos >= cthr[k]).any() else 0.0) for k in thr}
ck = os.path.join(RESULTS, "ahn", "ahn.pt")
state = torch.load(ck, map_location="cpu")
state["__meta__"]["thresholds"], state["__meta__"]["cos_thresholds"] = thr, cthr
state["__meta__"]["metrics"]["coverage"] = cover
torch.save(state, ck)
m = json.load(open(os.path.join(RESULTS, "ahn", "metrics.json")))
m.update(thresholds=thr, cos_thresholds=cthr, coverage=cover)
json.dump(m, open(os.path.join(RESULTS, "ahn", "metrics.json"), "w"), indent=1)
print(json.dumps(dict(thresholds=thr, cos_thresholds=cthr, coverage=cover), indent=1))
