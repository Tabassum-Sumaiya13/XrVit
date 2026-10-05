# XrVit onboarding guide

This repository studies whether an X-ray-pretrained transformer can provide
better chest X-ray classification than a small CNN without paying the
transformer's inference cost on every image.

The task is 14-label classification on the NIH ChestX-ray14 dataset. The main
models are:

- **ConvNeXt-Tiny**, resized to 384 pixels.
- **RAD-DINO (ViT-B/14)**, using its native 518-pixel input size.
- **Distilled ConvNeXt-Tiny**, trained from the teacher models' soft
  predictions.

This is research code, not a clinical product.

## Repository map

| Path | Purpose |
| --- | --- |
| [`README.md`](./README.md) | Project summary, headline metrics, and limitations |
| [`evaluation_protocol.yaml`](./evaluation_protocol.yaml) | Pre-registered experiment rules and acceptance criteria |
| [`rad-dino_official-split.ipynb`](./rad-dino_official-split.ipynb) | Train/evaluate RAD-DINO on the official NIH split |
| [`4-distillation-ipynb.ipynb`](./4-distillation-ipynb.ipynb) | Produce and evaluate the distilled CNN |
| [`gradcam.ipynb`](./gradcam.ipynb) | Grad-CAM/attribution exploration |
| [`generate_plots.py`](./generate_plots.py) | Generate publication-style comparison figures from fixed result values |
| [`exploratory/`](./exploratory/) | Post-hoc analyses; see [`exploratory/README.md`](./exploratory/README.md) |

Several notebooks and result archives are intentionally ignored by Git. Do
not assume that a clean clone contains trained checkpoints or result folders.

## Prerequisites

The supported execution environment is **Kaggle notebooks**:

- Attach the NIH ChestX-ray14 dataset, usually exposed as
  `nih-chest-xrays/data`.
- Enable a GPU for model training and RAD-DINO inference.
- Enable Internet when the notebook needs to obtain model dependencies or
  pretrained weights.
- Use a Kaggle session with enough disk space for the dataset, checkpoints,
  logits, and generated result files.

The notebooks install or import their Python dependencies in notebook cells.
There is no repository-level `requirements.txt`; use the versions and setup
cells in the notebook rather than inventing a separate local environment.

## Quick start

1. Clone or open the repository.
2. Open the notebook you want to run in Kaggle or Jupyter.
3. Attach the NIH dataset and confirm that it contains:
   - `Data_Entry_2017.csv` (or the supported equivalent CSV),
   - `train_val_list.txt`,
   - `test_list.txt`,
   - the referenced image files.
4. Run the setup and configuration cells first.
5. Keep the default seed (`42`) unless you are deliberately running a new
   experiment.
6. Let the notebook's data-validation cells complete before starting model
   training.
7. Save or download the generated result directory before the Kaggle session
   expires.
