import numpy as np, json, glob
from scipy.special import expit
from sklearn.metrics import roc_auc_score

def label_order_of(z):
    lo = z["label_order"]
    return json.loads(str(lo)) if lo.shape == () else list(lo)

R = "D:/XrVit/"
CT = json.load(open(R + "results_distill/calibration_thresholds_distill.json"))

def load(path):
    z = np.load(path, allow_pickle=True)
    return z

cnn_z = load(glob.glob(R + "results_s42/runs/*M08*/logits_test.npz")[0])
rd_z  = load(glob.glob(R + "results_s42/runs/*M09*/logits_test.npz")[0])
st_z  = load(R + "results_distill/NIH_CXR14__NIH_OFFICIAL_PV__M10_CONVNEXT_TINY_KD_384__s42/logits_test.npz")

findings = label_order_of(cnn_z)
y = cnn_z["labels"].astype(int)
pid = cnn_z["patient_id"]
assert (cnn_z["image_index"] == rd_z["image_index"]).all()
assert (cnn_z["image_index"] == st_z["image_index"]).all()
assert label_order_of(rd_z) == findings and label_order_of(st_z) == findings

def calib(z, key):
    a = np.array(CT["calibration"][key]["a"]); b = np.array(CT["calibration"][key]["b"])
    return expit(a * z["logits"] + b)

p_cnn = calib(cnn_z, "CNN alone")
p_rd  = calib(rd_z, "RAD-DINO alone")
p_st  = calib(st_z, "Distilled CNN")
p_avg = (p_cnn + p_rd) / 2

def per_finding_auc(p, idx):
    return np.array([roc_auc_score(y[idx, j], p[idx, j]) for j in range(len(findings))])

full_idx = np.arange(len(y))
point = {
    "CNN": per_finding_auc(p_cnn, full_idx),
    "RAD-DINO": per_finding_auc(p_rd, full_idx),
    "Average": per_finding_auc(p_avg, full_idx),
    "Student": per_finding_auc(p_st, full_idx),
}

# patient-level cluster bootstrap, matching protocol: 1000 resamples of test patients, seed 20260922
unique_patients = np.unique(pid)
n_patients = len(unique_patients)
patient_to_idx = {pt: np.where(pid == pt)[0] for pt in unique_patients}
rng = np.random.RandomState(20260922)

n_boot = 1000
deltas = np.zeros((n_boot, len(findings)))
for b in range(n_boot):
    sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
    idx = np.concatenate([patient_to_idx[pt] for pt in sampled_patients])
    auc_cnn = per_finding_auc(p_cnn, idx)
    auc_st  = per_finding_auc(p_st, idx)
    deltas[b] = auc_st - auc_cnn

lo = np.percentile(deltas, 2.5, axis=0)
hi = np.percentile(deltas, 97.5, axis=0)
delta_point = point["Student"] - point["CNN"]

print(f"{'Finding':<20}{'CNN':>8}{'RAD-DINO':>10}{'Average':>10}{'Student':>10}{'Delta':>10}{'CI_lo':>10}{'CI_hi':>10}  excl0")
rows = []
for j, f in enumerate(findings):
    excl0 = "yes" if (lo[j] > 0 or hi[j] < 0) else "no"
    print(f"{f:<20}{point['CNN'][j]:>8.4f}{point['RAD-DINO'][j]:>10.4f}{point['Average'][j]:>10.4f}{point['Student'][j]:>10.4f}{delta_point[j]:>+10.4f}{lo[j]:>+10.4f}{hi[j]:>+10.4f}  {excl0}")
    rows.append(dict(finding=f, cnn=point['CNN'][j], raddino=point['RAD-DINO'][j], average=point['Average'][j],
                      student=point['Student'][j], delta=delta_point[j], ci_lo=lo[j], ci_hi=hi[j], excludes_zero=excl0))

import csv
with open(R + "results_distill/T_perfinding_ci.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("\nwrote results_distill/T_perfinding_ci.csv")
