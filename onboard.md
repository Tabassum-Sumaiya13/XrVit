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
