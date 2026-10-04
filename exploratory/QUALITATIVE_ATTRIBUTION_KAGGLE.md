# Kaggle qualitative attribution figure

The script [qualitative_attribution_kaggle.py](qualitative_attribution_kaggle.py) runs the complete analysis without retraining.

Attach the NIH ChestX-ray14 dataset and the result dataset containing `results_s42/` and `results_distill/`. Enable a GPU, then run:

```python
!pip install -q timm transformers
!python /kaggle/input/<your-code-dataset>/qualitative_attribution_kaggle.py
```

The script searches recursively below `/kaggle/input` and `/kaggle/working`, so the result dataset does not need a particular mount name.

Outputs are written to `/kaggle/working/qualitative_attribution/`:

- `hard_positive_attribution.png` at 400 dpi;
- `hard_positive_attribution.pdf`;
- `selected_cases.csv` and `selected_cases.json`;
- individual CNN, distilled-CNN, and RAD-DINO attribution panels.

Cases are selected from saved test predictions using the fixed rule `ground truth positive, baseline negative, RAD-DINO positive, distilled CNN positive`. The selection score uses only prediction margins and is computed before any image or attribution map is inspected. Preferred findings are attempted in the requested order, with deterministic margin-based fallback when needed.

The CNN maps use Grad-CAM at the final ConvNeXt stage. RAD-DINO uses gradient times activation over final patch tokens and is labeled accordingly; it is not called Grad-CAM. The maps are intended only for qualitative comparison.

## Caption draft

Qualitative attribution analysis on representative hard-positive ChestX-ray14 cases recovered by the distilled CNN. Each row shows the original chest radiograph and attribution maps for the baseline ConvNeXt-Tiny, distilled ConvNeXt-Tiny, and RAD-DINO. The selected cases were positive for the indicated finding, missed by the baseline CNN, and correctly identified by both the distilled CNN and RAD-DINO. Grad-CAM is used for the CNN-based models, while gradient times activation over final patch tokens is used for RAD-DINO. The maps are intended for qualitative comparison and do not constitute quantitative localization evaluation.