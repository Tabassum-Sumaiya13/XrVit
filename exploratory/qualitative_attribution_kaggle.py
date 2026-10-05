"""Create a hard-positive attribution figure on Kaggle.

This script does not train or modify model weights. It selects cases from the
saved test predictions, then reruns only those images for attribution.
"""

import csv
import glob
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy.special import expit

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
N_CLASSES = len(CLASSES)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_ROOT = Path(os.environ.get("CXR_INPUT_DIR", "/kaggle/input"))
OUTPUT_DIR = Path(os.environ.get("CXR_ATTRIBUTION_OUT", "/kaggle/working/qualitative_attribution"))
N_CASES = 4
PREFERRED_FINDINGS = ["Pneumonia", "Infiltration", "Mass", "Cardiomegaly"]

CNN_CFG = dict(
    kind="cnn", timm_name="convnext_tiny.fb_in22k_ft_in1k", img_size=384,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), drop_path=0.1,
)
VIT_CFG = dict(
    kind="raddino", hf_name="microsoft/rad-dino",
    hf_revision="110cbc18d5133582e320b43d53bf5c44e410c936", img_size=518,
    mean=(0.5307,) * 3, std=(0.2583,) * 3, drop_path=0.1,
)


def find_one(pattern, roots):
    hits = []
    for root in roots:
        hits.extend(glob.glob(str(root / pattern), recursive=True))
    hits = sorted(set(hits))
    if not hits:
        raise FileNotFoundError(f"Could not find {pattern!r} below {[str(r) for r in roots]}")
    if len(hits) > 1:
        exact = [p for p in hits if "results_distill" in p or "results_s42" in p]
        if len(exact) == 1:
            return Path(exact[0])
        raise RuntimeError(f"Multiple matches for {pattern}: {hits}")
    return Path(hits[0])


def locate_inputs():
    roots = [INPUT_ROOT, Path("/kaggle/working")]
    cal_distill = find_one("**/calibration_thresholds_distill.json", roots)
    results_s42 = find_one("**/calibration.json", roots).parent
    m08_logits = find_one("**/*M08*/*logits_test.npz", roots)
    m09_logits = find_one("**/*M09*/*logits_test.npz", roots)
    m10_logits = find_one("**/*M10*/*logits_test.npz", roots)
    m08_ckpt = find_one("**/*M08*/checkpoint_best.pt", roots)
    m09_ckpt = find_one("**/*M09*/checkpoint_best.pt", roots)
    m10_ckpt = find_one("**/*M10*/checkpoint_best.pt", roots)
    return dict(
        results_s42=results_s42, cal_distill=cal_distill,
        m08_logits=m08_logits, m09_logits=m09_logits, m10_logits=m10_logits,
        m08_ckpt=m08_ckpt, m09_ckpt=m09_ckpt, m10_ckpt=m10_ckpt, roots=roots,
    )


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def label_order(z):
    value = z["label_order"]
    if np.asarray(value).shape == ():
        return json.loads(str(value))
    return list(value)


def load_prediction_data(paths):
    arrays = [np.load(paths[key], allow_pickle=True) for key in ("m08_logits", "m09_logits", "m10_logits")]
    ids = [np.asarray(z["image_index"]).astype(str) for z in arrays]
    if not (np.array_equal(ids[0], ids[1]) and np.array_equal(ids[0], ids[2])):
        raise ValueError("The three logits files do not have identical test-image order")
    if any(label_order(z) != CLASSES for z in arrays):
        raise ValueError("The logits label order does not match the protocol class order")
    labels = arrays[0]["labels"].astype(bool)
    if not (np.array_equal(labels, arrays[1]["labels"]) and np.array_equal(labels, arrays[2]["labels"])):
        raise ValueError("The three logits files do not have identical labels")
    return ids[0], labels, arrays


def calibrated_probabilities(arrays, cal):
    keys = ["CNN alone", "RAD-DINO alone", "Distilled CNN"]
    probs = {}
    for key, z in zip(keys, arrays):
        c = cal["calibration"][key]
        probs[key] = expit(np.asarray(c["a"]) * z["logits"] + np.asarray(c["b"]))
    thresholds = {key: np.asarray(cal["thresholds"][key]["thresholds"]) for key in keys}
    return probs, thresholds


def select_cases(ids, labels, probs, thresholds):
    pred = {key: probs[key] >= thresholds[key] for key in probs}
    recovery = labels & ~pred["CNN alone"] & pred["RAD-DINO alone"] & pred["Distilled CNN"]
    margins = {}
    for j, finding in enumerate(CLASSES):
        margins[finding] = np.minimum(
            thresholds["CNN alone"][j] - probs["CNN alone"][:, j],
            np.minimum(
                probs["RAD-DINO alone"][:, j] - thresholds["RAD-DINO alone"][j],
                probs["Distilled CNN"][:, j] - thresholds["Distilled CNN"][j],
            ),
        )

    chosen, used = [], set()
    for finding in PREFERRED_FINDINGS:
        j = CLASSES.index(finding)
        candidates = [i for i in np.flatnonzero(recovery[:, j]) if i not in used]
        if candidates:
            best = max(candidates, key=lambda i: (margins[finding][i], ids[i]))
            chosen.append((best, j))
            used.add(best)

    remaining = []
    for j, finding in enumerate(CLASSES):
        for i in np.flatnonzero(recovery[:, j]):
            if i not in used:
                remaining.append((margins[finding][i], i, j))
    remaining.sort(reverse=True)
    for _, i, j in remaining:
        if len(chosen) >= N_CASES:
            break
        chosen.append((i, j))
        used.add(i)
    if len(chosen) < N_CASES:
        raise RuntimeError(f"Only {len(chosen)} qualifying cases were available")

    rows = []
    for row_number, (i, j) in enumerate(chosen[:N_CASES], 1):
        finding = CLASSES[j]
        row = dict(
            row=row_number, image_index=ids[i], target_finding=finding,
            ground_truth=int(labels[i, j]),
            selection_rule="GT=1, baseline=0, RAD-DINO=1, distilled=1",
        )
        for key in ("CNN alone", "Distilled CNN", "RAD-DINO alone"):
            row[f"{key}_probability"] = float(probs[key][i, j])
            row[f"{key}_threshold"] = float(thresholds[key][j])
            row[f"{key}_prediction"] = int(pred[key][i, j])
        rows.append(row)
    return rows


def find_image_paths(ids, roots):
    wanted, found = set(ids), {}
    for root in roots:
        for path in root.rglob("*.png"):
            if path.name in wanted:
                found[path.name] = path
        if len(found) == len(wanted):
            break
    missing = sorted(wanted - set(found))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected PNGs; examples: {missing[:5]}")
    return found


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            CNN_CFG["timm_name"], pretrained=False, num_classes=0,
            drop_path_rate=CNN_CFG["drop_path"],
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.backbone.num_features, N_CLASSES))

    def forward(self, x):
        return self.head(self.backbone(x))


class RadDinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            VIT_CFG["hf_name"], revision=VIT_CFG["hf_revision"],
            drop_path_rate=VIT_CFG["drop_path"],
        )
        self.n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * self.backbone.config.hidden_size, N_CLASSES))

    def forward(self, x):
        h = self.backbone(pixel_values=x).last_hidden_state
        return self.head(torch.cat([h[:, 0], h[:, 1 + self.n_reg:].mean(1)], dim=1))


def load_state(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def transform_for(cfg):
    return T.Compose([
        T.Resize((cfg["img_size"], cfg["img_size"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"]),
    ])


def gradcam(model, x, target_class):
    activation, gradient = {}, {}
    layer = model.backbone.stages[-1]

    def save_activation(_, __, output):
        activation["value"] = output

    def save_gradient(_, __, grad_output):
        gradient["value"] = grad_output[0]

    handles = [layer.register_forward_hook(save_activation), layer.register_full_backward_hook(save_gradient)]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, target_class].backward()
        act, grad = activation["value"].detach(), gradient["value"].detach()
    finally:
        for handle in handles:
            handle.remove()
    if act.ndim != 4:
        raise RuntimeError(f"Unexpected ConvNeXt target-layer shape: {tuple(act.shape)}")
    if not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0:
        raise RuntimeError("ConvNeXt Grad-CAM received no finite, nonzero target-layer gradient")
    if act.shape[1] <= act.shape[-1] and act.shape[1] < 64:
        act, grad = act.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(1).relu()[0]
    return normalize_map(cam.cpu().numpy())


def raddino_grad_activation(model, x, target_class):
    model.zero_grad(set_to_none=True)
    h = model.backbone(pixel_values=x).last_hidden_state
    h.retain_grad()
    pooled = torch.cat([h[:, 0], h[:, 1 + model.n_reg:].mean(1)], dim=1)
    model.head(pooled)[0, target_class].backward()
    if h.grad is None or not torch.isfinite(h.grad).all() or float(h.grad.abs().sum()) == 0:
        raise RuntimeError("RAD-DINO attribution received no finite, nonzero patch-token gradient")
    tokens = h.grad[:, 1 + model.n_reg:] * h[:, 1 + model.n_reg:]
    attribution = tokens.abs().mean(-1)[0]
    side = int(math.sqrt(attribution.numel()))
    if side * side != attribution.numel():
        raise RuntimeError(f"RAD-DINO patch count {attribution.numel()} is not square")
    return normalize_map(attribution.reshape(side, side).detach().cpu().numpy())


def normalize_map(values):
    values = np.maximum(values, 0).astype(np.float32)
    lo, hi = np.percentile(values, [1, 99])
    return np.clip((values - lo) / max(hi - lo, 1e-8), 0, 1)


def resize_map(values, size):
    image = Image.fromarray((values * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255


def save_rows(rows):
    with (OUTPUT_DIR / "selected_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT_DIR / "selected_cases.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def make_figure(rows, panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(len(rows), 4, figsize=(7.2, 2.0 * len(rows)), squeeze=False)
    titles = ["Original CXR", "CNN Grad-CAM", "Distilled CNN Grad-CAM", "RAD-DINO\nGrad x Activation"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9, pad=5)
    for r, row in enumerate(rows):
        image, cnn_cam, student_cam, vit_cam = panels[row["image_index"]]
        image_array = np.asarray(image, dtype=np.float32) / 255
        for c, heat in enumerate((None, cnn_cam, student_cam, vit_cam)):
            ax = axes[r, c]
            ax.imshow(image_array, cmap="gray", vmin=0, vmax=1)
            if heat is not None:
                ax.imshow(heat, cmap="magma", vmin=0, vmax=1, alpha=0.45)
            ax.axis("off")
        axes[r, 0].set_ylabel(row["target_finding"], fontsize=9, labelpad=8, rotation=90, va="center")
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01, label="Attribution intensity")
    fig.subplots_adjust(wspace=0.03, hspace=0.12, left=0.08, right=0.94, top=0.94, bottom=0.03)
    fig.savefig(OUTPUT_DIR / "hard_positive_attribution.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "hard_positive_attribution.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = locate_inputs()
    ids, labels, arrays = load_prediction_data(paths)
    probs, thresholds = calibrated_probabilities(arrays, load_json(paths["cal_distill"]))
    rows = select_cases(ids, labels, probs, thresholds)
    image_paths = find_image_paths([row["image_index"] for row in rows], paths["roots"])
    save_rows(rows)

    cnn = load_state(CNNClassifier(), paths["m08_ckpt"])
    student = load_state(CNNClassifier(), paths["m10_ckpt"])
    raddino = load_state(RadDinoClassifier(), paths["m09_ckpt"])
    cnn_tfm, vit_tfm = transform_for(CNN_CFG), transform_for(VIT_CFG)
    panels = {}
    for row in rows:
        image = Image.open(image_paths[row["image_index"]]).convert("L")
        target_class = CLASSES.index(row["target_finding"])
        cnn_x = cnn_tfm(image).repeat(1, 3, 1, 1).to(DEVICE)
        vit_x = vit_tfm(image).repeat(1, 3, 1, 1).to(DEVICE)
        cnn_cam = resize_map(gradcam(cnn, cnn_x, target_class), image.size)
        student_cam = resize_map(gradcam(student, cnn_x, target_class), image.size)
        vit_cam = resize_map(raddino_grad_activation(raddino, vit_x, target_class), image.size)
        panels[row["image_index"]] = (image, cnn_cam, student_cam, vit_cam)
        for name, panel in (("cnn", cnn_cam), ("distilled", student_cam), ("raddino", vit_cam)):
            Image.fromarray((panel * 255).astype(np.uint8)).save(OUTPUT_DIR / f"{row['image_index']}_{row['target_finding']}_{name}_attribution.png")
    make_figure(rows, panels)
    print(f"Wrote {len(rows)} cases and figure files to {OUTPUT_DIR}")
    print("RAD-DINO method: gradient x activation over final patch tokens, not standard Grad-CAM.")


if __name__ == "__main__":
    main()