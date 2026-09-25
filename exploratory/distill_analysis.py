"""Post hoc analysis of the distilled CNN (M10) on the test set: error counts, silent misses, student-teacher agreement.
Exploratory: not part of protocol v2.2.0. Needs D:/XrVit/results_distill."""
import glob, json
import numpy as np
from scipy.special import expit

CT = json.load(open("D:/XrVit/results_distill/calibration_thresholds_distill.json"))
RUNS = {"CNN alone": glob.glob("D:/XrVit/results_s42/runs/*M08*")[0], "RAD-DINO alone": glob.glob("D:/XrVit/results_s42/runs/*M09*")[0],
        "Distilled CNN": glob.glob("D:/XrVit/results_distill/*M10*")[0]}
P, Z = {}, {}
for k, d in RUNS.items():
    z = np.load(f"{d}/logits_test.npz")
    Z[k], y = z["logits"], z["labels"].astype(bool)
    P[k] = expit(np.array(CT["calibration"][k]["a"]) * z["logits"] + np.array(CT["calibration"][k]["b"]))
P["Average CNN+RAD-DINO"] = (P["CNN alone"] + P["RAD-DINO alone"]) / 2
pred = {k: P[k] >= np.array(CT["thresholds"][k]["thresholds"]) for k in P}

print(f"true findings per image {y.sum(1).mean():.3f}")
for k in pred:
    print(f"{k:22s} predicted/image {pred[k].sum(1).mean():.3f}  TP {int((pred[k] & y).sum()):6d}  "
          f"FP {int((pred[k] & ~y).sum()):6d}  FN {int((~pred[k] & y).sum()):6d}")

miss = y & ~pred["CNN alone"]                                  # positives the CNN called No
rd_only = miss & pred["RAD-DINO alone"]
print(f"\nCNN-missed positives {miss.sum()}: found by RAD-DINO {int((miss & pred['RAD-DINO alone']).sum())}, "
      f"distilled {int((miss & pred['Distilled CNN']).sum())}, average {int((miss & pred['Average CNN+RAD-DINO']).sum())}")
print(f"found by RAD-DINO but missed by CNN: {rd_only.sum()}; distilled catches {int((rd_only & pred['Distilled CNN']).sum())}")

cc = lambda a, b: np.mean([np.corrcoef(a[:, c], b[:, c])[0, 1] for c in range(a.shape[1])])
print(f"\nmean per-class logit correlation: distilled~CNN {cc(Z['Distilled CNN'], Z['CNN alone']):.3f}, "
      f"distilled~RAD-DINO {cc(Z['Distilled CNN'], Z['RAD-DINO alone']):.3f}, CNN~RAD-DINO {cc(Z['CNN alone'], Z['RAD-DINO alone']):.3f}")
