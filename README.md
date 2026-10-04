# XrVit: RAD-DINO accuracy at CNN cost on chest X-rays
Dataset: https://www.kaggle.com/datasets/nih-chest-xrays/data

https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

Multi-label classification of 14 findings on the NIH ChestX-ray14 **official test split**, with two trained models:
a fast CNN (**ConvNeXt-Tiny**, 384 px) and an X-ray-pretrained transformer (**RAD-DINO**, ViT-B/14, 518 px).
The question: can we get RAD-DINO's accuracy without paying RAD-DINO's cost on every image?

1. **Distillation.** RAD-DINO teaches a new ConvNeXt-Tiny during training and is not run at test time. **Helped**: the CNN kept its cost and gained about 75% of RAD-DINO's AUROC advantage.


## Headline results (official test set, 25,596 images)

| System | Macro AUROC [95% CI] | Macro F1 | Cost vs RAD-DINO (batch 32) |
|---|---|---:|---:|
| CNN alone | 0.8263 [0.8200, 0.8317] | 0.3539 | 0.18 |
| **Distilled CNN** | **0.8339** [0.8283, 0.8387] | **0.3702** | **0.18** |
| RAD-DINO alone | 0.8364 [0.8303, 0.8415] | 0.3788 | 1.00 |
| Average of both models | 0.8428 [0.8373, 0.8478] | 0.3818 | 1.18 |

One training seed (42), no test-time augmentation, per-class Platt calibration and F1 thresholds fitted on validation, CIs from 1,000 patient-level bootstrap resamples. The distilled CNN is significantly better than the CNN (+0.0076 AUROC) but did not meet the pre-registered "matches RAD-DINO" rule (lower CI bound −0.0056 vs −0.005).

<!-- ## What is in this repository

| Path | Contents |
|---|---|
| [RESULTS.md](RESULTS.md) | Full report: cascade study, why its gate cannot be fixed, distillation study, recommendations |
| [METHODOLOGY.md](METHODOLOGY.md) | Every mechanism of the final (distilled) model: data, preprocessing, teachers, loss, calibration, thresholds, distillation, evaluation, QC, with references |
| [evaluation_protocol.yaml](evaluation_protocol.yaml) | Rules fixed before each experiment (v2.0.0 cascade, v2.1.0 layer depth, v2.2.0 distillation) |
| [2-staged_official_split.ipynb](2-staged_official_split.ipynb) | Cascade study: training both models, calibration, gate sweep, test metrics, compute |
| [3-raddino_layer_cut.ipynb](3-raddino_layer_cut.ipynb) | Layer-depth experiment: linear probes on each of RAD-DINO's 12 blocks (not part of RESULTS.md) |
| [4-distillation.ipynb](4-distillation.ipynb) | Distillation study: teacher soft labels and go/no-go (D1), student training (D2), evaluation (D3) |
| [exploratory/](exploratory/) | Post hoc analysis scripts behind RESULTS.md sections 7 and 8.4 (see its README) |
| [DM_Workbook.xlsx](DM_Workbook.xlsx) | Results workbook; sheet 7 = distillation |
| `1610.02391v4.pdf` | Reference: Grad-CAM (Selvaraju et al.) |
| `2511.19920v1.pdf` | Reference: image search with visual large models (Wang et al., 2025) |

**Not in git** (see [.gitignore](.gitignore)): the result folders and zip files. They are produced by the notebooks and kept locally for future runs:

- `results_s42/`: cascade study tables, logits, and both trained teachers (`runs/*/checkpoint_best.pt`, 106 MB and 330 MB).
- `results_distill/`: distillation tables, the student checkpoint, the teachers' training-set logits and the soft labels (reusable without a GPU).
- `results_layers/`: layer-depth tables.

The official split lists (`train_val_list.txt`, `test_list.txt`) come with the NIH dataset on Kaggle; every notebook checks their SHA-256. -->
<!-- 
## How to reproduce

All notebooks run on Kaggle with a Tesla T4 GPU and Internet on.

1. Attach the dataset `nih-chest-xrays/data` (images, `Data_Entry_2017.csv`, official list files).
<!-- 2. **Cascade** ([2-staged_official_split.ipynb](2-staged_official_split.ipynb)): set one flag per session in the config cell. Sessions A and B train the CNN (about 2.8 h) and RAD-DINO (about 6.6 h), session T measures latency, and session C (CPU) runs calibration, the gate sweep and all test metrics. Outputs go to `results_s42/`. -->
<!-- 2. **Distillation** ([4-distillation.ipynb](4-distillation.ipynb)): upload `results_s42/` as a Kaggle dataset (it already holds both `checkpoint_best.pt` files inside their run folders; the notebook checks their SHA-256), then run sessions D1+D2 on GPU (about 3.5 h) and D3 on CPU (about 15 min). Outputs go to `results_distill/`.
3. Every notebook checks the split hash (`b9f2fbc9…`), patient separation and identical evaluation images before reporting anything. --> 

## Limitations

One training seed, so confidence intervals cover test-set sampling only. Labels are text-mined from radiology reports (noisy). One dataset, no external test set. This is research code, not a clinical tool.
