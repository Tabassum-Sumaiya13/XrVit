# Getting RAD-DINO's accuracy at a CNN's cost: a cascade and a distillation on chest X-rays

**Results on the official NIH ChestX-ray14 test set**

*Protocol v2.0.0 (cascade) and v2.2.0 (distillation) · split `NIH_OFFICIAL_PV` (SHA-256 `b9f2fbc9…`) · one training seed (42) · no test-time augmentation · 25 September 2026*

---

## Summary

We have two trained models that label 14 findings on chest X-rays: a small, fast CNN (ConvNeXt-Tiny) and a larger, X-ray-pretrained transformer (RAD-DINO). RAD-DINO is more accurate, but in batch processing it costs about 5.7 times as much per image. We tested two ways to get RAD-DINO's accuracy at a lower cost.

**1. A two-stage cascade did not work** (Sections 1–6). The CNN reads every image and passes the images it is unsure about to RAD-DINO.

- **The compute saving mostly disappeared.** The gate was tuned on validation, where it sent 44% of images to RAD-DINO. On the test set it sent **78%**. The saving fell from 38% to **5%** at batch size 32. One image at a time, the cascade was **29% slower** than RAD-DINO alone.
- **It was also less accurate than RAD-DINO alone**: macro AUROC −0.004, macro F1 −0.008, both significant.
- **Main reason:** the test patients are sicker than the validation patients (1.7× more findings per image), and the CNN is less sure about test images in general.

**2. A better gate would not have rescued it** (Section 7, exploratory). The CNN's mistakes are silent: when it misses a finding, it gives it a low score, just like a true negative. Gates built from the CNN's scores found the images where RAD-DINO helps barely better than chance. At equal cost, they did about as well as sending random images to RAD-DINO.

**3. Using RAD-DINO as a teacher worked** (Section 8). We trained a new ConvNeXt-Tiny on the soft predictions of the CNN + RAD-DINO average. RAD-DINO is used only during training. At test time the new CNN runs alone.

- **It is clearly better than the original CNN:** macro AUROC **0.8339 vs 0.8263** (+0.0076, 95% CI +0.0058 to +0.0096), at the **same cost** (0.18× RAD-DINO).
- **It closes about 75% of the AUROC gap** between the CNN and RAD-DINO, and about two thirds of the AUPRC and macro-F1 gaps. Its macro recall equals RAD-DINO's.
- **It does not formally match RAD-DINO.** The rule fixed in advance needed the lower CI bound of (Distilled − RAD-DINO) macro AUROC to be above −0.005. It came out at −0.0056. The official verdict is **"helps"**.
- **It makes the cascade pointless:** the same accuracy (AUROC 0.834 vs 0.833, macro F1 0.370 vs 0.371) at about **one fifth of the cost**.

**What to use** (Section 10): the distilled CNN when compute is limited; RAD-DINO alone for the best single model; the average of both models when accuracy matters more than cost.

---

## 1. What was compared

Four systems were evaluated on the same images:

| System | What it does | Share of images RAD-DINO reads |
|---|---|---|
| **CNN alone** | ConvNeXt-Tiny at 384 px (27.8 M parameters) | 0% |
| **RAD-DINO alone** | RAD-DINO ViT-B/14 at 518 px (86.6 M parameters) | 100% |
| **Average** | Mean of both models' calibrated probabilities on every image | 100% |
| **Cascade** | CNN first. An image goes to RAD-DINO if *any* of its 14 findings is unsure (calibrated probability between 0.2 and 0.8). RAD-DINO then decides all 14 findings for that image. | Depends on the data |

**How decisions were fixed.** Every choice was made on the validation set before test labels were used:

- Each model's scores were calibrated per finding (Platt scaling: a small logistic fit that turns raw scores into probabilities).
- Each finding got its own yes/no threshold, chosen to maximise F1 on validation.
- The cascade gate was chosen by a rule written before the test run. From 23 gate settings, it takes the one that sends the fewest validation images to RAD-DINO while keeping validation macro F1 within 0.01 of RAD-DINO alone. That rule picked a flat gate at t = 0.80 (unsure = probability between 0.2 and 0.8).

**Uncertainty.** 95% confidence intervals come from 1,000 bootstrap resamples of test *patients*, not images. This matters because many patients have several images. Paired differences use the same resamples for every system, with Holm correction across the six system pairs. A difference counts as significant only if its interval excludes zero *and* its corrected p-value is below 0.05.

**Metrics.** The primary measure is **macro AUROC**: how well a system ranks positive images above negative ones, averaged over the 14 findings. It does not depend on thresholds. Secondary measures are AUPRC (ranking quality that accounts for how rare a finding is), and F1, precision and recall at the chosen thresholds.

Full rules are in [evaluation_protocol.yaml](evaluation_protocol.yaml).

---

## 2. The test population differs from the data used to tune the systems

This is the key to reading everything that follows.

| | Train | Validation | Test (official) |
|---|---:|---:|---:|
| Images | 73,916 | 12,608 | 25,596 |
| Patients | 23,806 | 4,202 | 2,797 |
| Images per patient | 3.1 | 3.0 | **9.2** |
| Findings per image | 0.63 | 0.61 | **1.06** |
| Images with no finding | 58% | 59% | **39%** |
| PA (standing, posterior–anterior) views | 65% | 66% | **43%** |
| Male | 56% | 55% | 58% |
| Median age | 49 | 49 | 49 |

Several findings are 2–4× more common in the test set:

| Finding | Validation prevalence | Test prevalence | Ratio |
|---|---:|---:|---:|
| Pneumothorax | 2.8% | 10.4% | 3.8× |
| Emphysema | 1.2% | 4.3% | 3.6× |
| Edema | 1.4% | 3.6% | 2.6× |
| Consolidation | 2.9% | 7.1% | 2.4× |
| Pneumonia | 1.1% | 2.2% | 2.0× |
| Effusion | 9.9% | 18.2% | 1.8× |

The official test patients are sicker, are imaged more often, and have more bedside (non-PA) films. The validation set was drawn from the same pool as the training set, so it looks like the training data, not like the test data.

---

## 3. Accuracy on the test set

**Table 1. Test results (value [95% CI]).**

| System | Macro AUROC | Macro AUPRC | Macro F1 | Micro F1 | Macro recall | Macro precision |
|---|---|---|---|---|---|---|
| CNN alone | 0.8263 [0.8200, 0.8317] | 0.2988 [0.2849, 0.3154] | 0.3539 [0.3342, 0.3686] | 0.4172 [0.4090, 0.4247] | 0.4304 [0.4113, 0.4478] | 0.3188 [0.2965, 0.3350] |
| RAD-DINO alone | 0.8364 [0.8303, 0.8415] | 0.3223 [0.3103, 0.3366] | 0.3788 [0.3653, 0.3900] | 0.4317 [0.4238, 0.4391] | **0.4616** [0.4443, 0.4779] | 0.3362 [0.3235, 0.3491] |
| Average | **0.8428** [0.8373, 0.8478] | **0.3292** [0.3160, 0.3431] | **0.3818** [0.3674, 0.3928] | **0.4421** [0.4341, 0.4495] | 0.4561 [0.4389, 0.4724] | **0.3411** [0.3273, 0.3546] |
| Cascade (t = 0.80) | 0.8325 [0.8264, 0.8379] | 0.3176 [0.3048, 0.3315] | 0.3709 [0.3565, 0.3822] | 0.4273 [0.4192, 0.4347] | 0.4454 [0.4281, 0.4622] | 0.3401 [0.3272, 0.3536] |

Macro specificity was about 0.90 for every system (0.9022–0.9092).

**Table 2. Paired differences on the test set (A − B [95% CI]).** All differences are significant except those marked "n.s." (not significant).

| Comparison | Macro AUROC | Macro AUPRC | Macro F1 | Micro F1 |
|---|---|---|---|---|
| RAD-DINO − CNN | +0.0101 [+0.0063, +0.0140] | +0.0235 [+0.0131, +0.0339] | +0.0249 [+0.0108, +0.0390] | +0.0145 [+0.0104, +0.0187] |
| Average − CNN | +0.0165 [+0.0139, +0.0193] | +0.0303 [+0.0229, +0.0379] | +0.0279 [+0.0175, +0.0407] | +0.0249 [+0.0217, +0.0283] |
| Cascade − CNN | +0.0062 [+0.0034, +0.0089] | +0.0188 [+0.0099, +0.0290] | +0.0170 [+0.0028, +0.0301] | +0.0102 [+0.0061, +0.0145] |
| Average − RAD-DINO | +0.0064 [+0.0049, +0.0080] | +0.0069 [+0.0028, +0.0113] | +0.0030 [−0.0019, +0.0085] n.s. | +0.0104 [+0.0073, +0.0134] |
| **Cascade − RAD-DINO** | **−0.0039** [−0.0065, −0.0017] | **−0.0047** [−0.0085, −0.0012] | **−0.0079** [−0.0122, −0.0039] | **−0.0044** [−0.0063, −0.0025] |
| Cascade − Average | −0.0103 [−0.0129, −0.0084] | −0.0116 [−0.0151, −0.0077] | −0.0109 [−0.0174, −0.0053] | −0.0148 [−0.0182, −0.0113] |

What the tables show:

- **On AUROC, AUPRC and micro F1 the order is Average > RAD-DINO > Cascade > CNN, and every step is significant.** On macro F1 the order is the same, except that Average and RAD-DINO are tied.
- **RAD-DINO is clearly better than the CNN.** The AUROC gap (+0.010) is modest. The AUPRC gap (+0.023, about 8% relative) is larger, so RAD-DINO's advantage shows most on rare findings.
- **The cascade sits between its two members.** It keeps about 60% of RAD-DINO's AUROC gain over the CNN and about 70% of its F1 gain.
- **The cascade loses mainly on recall.** Its recall is 0.016 below RAD-DINO's (significant). Its precision is 0.004 higher (not significant). Section 5 explains where the lost recall goes.
- **Not significant:** every precision difference among RAD-DINO, Average and Cascade; recall of Average vs RAD-DINO; and recall of Cascade vs CNN.

**All systems score lower on test than on validation.** Macro AUROC dropped by 0.029–0.034 for every system (for example RAD-DINO: 0.8651 → 0.8364). The drop is similar for all four systems, so it reflects the harder test population rather than a weakness of any one system. Macro F1 rose slightly on test (RAD-DINO: 0.3722 → 0.3788) because F1 is easier when findings are more common. Calibration error roughly doubled (macro ECE 0.004–0.005 → 0.010–0.012), since the calibration was fitted on validation.

---

## 4. Cost: the saving seen on validation did not carry over

![Macro F1 and macro AUROC against the share of test images sent to RAD-DINO, for every gate setting](results_s42/cascade_tradeoff.png)

*Figure 1. Every gate setting on the test set. The x-axis is the share of test images sent to RAD-DINO. Horizontal lines show the single models and the average. The red circle is the gate selected on validation. The dashed red line is where that gate sat on validation (44%).*

**Table 3. Per-image forward-pass time on a Tesla T4 (fp16), and cost relative to RAD-DINO alone.**

| System | Share to RAD-DINO | Batch 32: ms/image | vs RAD-DINO | Batch 1: ms/image | vs RAD-DINO |
|---|---:|---:|---:|---:|---:|
| CNN alone | 0% | 4.0 | 0.18 | 11.4 | 0.51 |
| RAD-DINO alone | 100% | 22.8 | 1.00 | 22.2 | 1.00 |
| Average | 100% | 26.9 | 1.18 | 33.5 | 1.51 |
| Cascade, as seen on validation | 44.5% | 14.2 | **0.62** | 21.2 | 0.96 |
| Cascade, on test | **77.6%** | 21.7 | **0.95** | 28.6 | **1.29** |
| *Break-even share (cascade = RAD-DINO cost)* | | *82.5%* | | *48.7%* | |

- **On validation, the cascade looked worthwhile:** 38% less compute than RAD-DINO, for a macro F1 only 0.006 lower.
- **On test, 78% of images went to RAD-DINO.** At batch 32 that leaves a 5% saving. At batch 1 the cascade is 29% *slower* than RAD-DINO alone, because every image pays for the CNN and most still pay for RAD-DINO too.
- **The batch-size effect is large.** The CNN is 2.8× faster per image at batch 32 than at batch 1. RAD-DINO gets no speed-up from batching at 518 px. So the CNN is cheap in batch processing (18% of RAD-DINO's cost) but not in one-image-at-a-time use (51%).
- **No gate setting reached RAD-DINO's test macro F1 at lower cost** (Figure 1, left). The settings that matched it sent 93% or more of images to RAD-DINO and cost 1.11–1.17× RAD-DINO.

These cost figures cover the model forward pass only. A real deployment would also pay for loading and resizing each image at two resolutions (384 and 518 px), keeping two models in memory, and splitting batches after the gate. All of these make the cascade look worse, not better.

---

## 5. Why the cascade behaved differently on test

### 5.1 More images were "unsure", for two reasons

We split the images by how many findings they carry and checked how often each group is routed to RAD-DINO:

| Findings on the image | Validation: share of images | Validation: routed | Test: share of images | Test: routed |
|---|---:|---:|---:|---:|
| 0 | 58.9% | 28.0% | 38.5% | **61.7%** |
| 1 | 26.3% | 60.4% | 31.2% | 82.7% |
| 2 | 10.7% | 78.8% | 19.6% | 91.1% |
| 3 or more | 4.1% | 89.9% | 10.6% | 95.2% |
| **All** | | **44.5%** | | **77.6%** |

- **Case mix explains about a third of the jump.** If test images had kept the validation routing rate for their group, 54.7% would have been routed. That covers about 10 of the 33-point rise (31%).
- **The rest comes from the CNN being less sure about test images in every group.** The clearest case: test images with *no* findings were routed 62% of the time, against 28% on validation.
- **A few findings drive most of the uncertainty.** The share of images where the CNN was unsure about Infiltration rose from 24% to 50%. For Effusion it rose from 13% to 28%, and for Pneumothorax from 3% to 12%.

Two likely causes fit the data, but neither was tested directly: the lower share of PA views (66% → 43%), and the many follow-up films per test patient (9.2 images per patient). Bedside films often show tubes, lines and poorer positioning, which may make the CNN less certain even when no finding is labelled. Checking this needs routing rates by view position, which this analysis did not compute.

### 5.2 "Confident" CNN decisions miss many positive findings

On test, 5,742 images (22.4%) were decided by the CNN alone. These were not clean images: 34% of them carried at least one positive label.

| On the 5,742 CNN-decided test images | Count |
|---|---:|
| Positive labels present | 2,707 |
| Called positive by the cascade | 114 |
| Missed by the cascade | **2,593** (9.5% of all 27,206 positive labels in the test set) |
| Would have been found by RAD-DINO alone | 756 |

So on these images, the cascade finds about 640 fewer true positives than RAD-DINO alone would. This is the source of the recall gap in Table 1. Most of these findings are hard for every model: RAD-DINO also misses 72% of them. But RAD-DINO still finds 6–7 times more of them than the cascade does.

**Part of this comes from how the gate is defined.** The gate treats a probability of 0.2 or lower as a confident "No". Yet the F1-optimal threshold is below 0.2 for 13 of the 14 findings (range 0.06–0.20; only Effusion is higher at 0.32). The findings are rare, so the best cut-off is low. A finding with probability 0.15 would be called positive by the CNN alone, but the cascade treats it as a confident negative. On test, 285 positive labels were lost this way. They were called positive by the CNN's own thresholds and then turned into "No" by the gate. The other 2,308 would have been missed by the CNN either way.

### 5.3 Other gate designs did not solve the problem

- **Per-finding gates** set each finding's "confident No" limit from a missed-positive budget (1–10%). These were safe: only 1–65 positive labels were missed on test. But they sent 98–100% of test images to RAD-DINO, costing 1.15–1.17× RAD-DINO alone. At that point the cascade is just a more expensive RAD-DINO.
- **Letting RAD-DINO decide only the unsure findings of a routed image, instead of all 14, was worse at every gate.** At t = 0.80, test macro F1 was 0.333 against 0.371, and 13,249 labels were missed against 2,593. Both modes run RAD-DINO on the same images, so they cost the same. Ignoring RAD-DINO's answer for the "confident" findings just throws away information.
- **Using the average as the second stage** tied with the selected gate on validation routing and lost the tie-break on validation F1 by 0.001. On test, it did slightly better than the selected cascade: macro AUROC 0.8372, macro F1 0.3752, at the same 0.95 cost. This is an after-the-fact observation. It was not selected in advance and has no confidence interval, so it should not be read as a finding.

---

## 6. Results per finding

**Table 4. Test AUROC and F1 per finding.** Findings are sorted by the number of positive test images.

| Finding | Test positives | Prevalence | AUROC CNN | AUROC RAD-DINO | AUROC Average | AUROC Cascade | F1 RAD-DINO | F1 Cascade | Positives missed by the CNN-only path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Infiltration | 6,112 | 23.9% | 0.709 | 0.709 | 0.717 | 0.708 | 0.469 | 0.472 | 658 |
| Effusion | 4,658 | 18.2% | 0.839 | 0.843 | 0.848 | 0.842 | 0.555 | 0.554 | 242 |
| Atelectasis | 3,279 | 12.8% | 0.790 | 0.798 | 0.804 | 0.795 | 0.423 | 0.417 | 314 |
| Pneumothorax | 2,665 | 10.4% | 0.882 | 0.904 | 0.905 | 0.902 | 0.540 | 0.529 | 196 |
| Consolidation | 1,815 | 7.1% | 0.761 | 0.761 | 0.772 | 0.760 | 0.262 | 0.262 | 116 |
| Mass | 1,748 | 6.8% | 0.846 | 0.851 | 0.860 | 0.848 | 0.440 | 0.438 | 161 |
| Nodule | 1,623 | 6.3% | 0.798 | 0.803 | 0.817 | 0.794 | 0.318 | 0.304 | 250 |
| Pleural thickening | 1,143 | 4.5% | 0.794 | 0.814 | 0.817 | 0.811 | 0.264 | 0.251 | 150 |
| Emphysema | 1,093 | 4.3% | 0.936 | 0.949 | 0.948 | 0.945 | 0.541 | 0.535 | 107 |
| Cardiomegaly | 1,069 | 4.2% | 0.888 | 0.898 | 0.904 | 0.898 | 0.425 | 0.416 | 157 |
| Edema | 925 | 3.6% | 0.858 | 0.856 | 0.866 | 0.856 | 0.257 | 0.256 | 51 |
| Pneumonia | 555 | 2.2% | 0.727 | 0.729 | 0.742 | 0.731 | 0.089 | 0.086 | 51 |
| Fibrosis | 435 | 1.7% | 0.829 | 0.849 | 0.851 | 0.839 | 0.213 | 0.187 | 113 |
| Hernia ⚠ | 86 | 0.3% | 0.910 | 0.944 | 0.949 | 0.928 | 0.507 | 0.485 | 27 |

⚠ Fewer than 100 positive test images, so these values are very uncertain.

- **RAD-DINO gains most over the CNN on Pneumothorax (+0.022 AUROC), Pleural thickening (+0.020), Fibrosis (+0.019) and Emphysema (+0.013).** These are findings seen in fine detail or texture, where higher resolution and X-ray pretraining plausibly help. That explanation is our reading and was not tested.
- **On Infiltration, Consolidation, Pneumonia and Edema, the two models are equal (within 0.002).** The first three also have the lowest AUROC of all findings. They look alike on X-ray and are often described as inconsistently labelled in this dataset, so the ceiling may come from the labels rather than the models.
- **Pneumonia stays very hard for every system:** AUROC about 0.73, and F1 below 0.09.
- **The average of both models is the best or equal-best system on 12 of 14 findings.**
- Per-finding differences were not tested for significance, so small gaps in this table should not be read as real.

---

## 7. Why a better gate would not have helped (exploratory)

*This section was analysed after the test results were known. It explains the cascade's failure; it is not a confirmatory result. Scripts: [exploratory/](exploratory/).*

### 7.1 One unsure finding sends the whole image to RAD-DINO

An image stays with the CNN only if all 14 findings are "confident". Every finding is a separate chance to be unsure, so the more findings an image carries, the more likely it is routed. Even test images with no finding at all were routed 61.7% of the time (Section 5.1). Many test images do carry several findings: 30% have two or more, most often Effusion + Infiltration.

Routing only the unsure findings does not help (Section 5.3). Almost all of RAD-DINO's cost is reading the image; its 14 outputs are nearly free. So asking it about one finding costs the same as asking about all 14. And keeping the CNN's "confident" answers on routed images throws away RAD-DINO's corrections.

### 7.2 The CNN cannot flag its own misses

We compared the CNN alone with the average of both models, image by image. **89%** of the improvement (the drop in log loss) came from **positive** labels: findings that are present but that the CNN scored low. To the CNN, these images look like confident negatives.

A gate can only use what the CNN outputs. We measured how well different gate scores pick out the images where the second stage helps most (the top 30% by improvement; 0.5 = random, 1.0 = perfect):

| Gate score, computed from the CNN's probabilities | How well it finds the images where stage 2 helps (AUROC) |
|---|---:|
| Current gate: largest uncertainty over the 14 findings | 0.587 |
| Total uncertainty (sum of entropies) | 0.626 |
| Closeness to each finding's own decision threshold | 0.613 |
| Learned gate (gradient boosting on the CNN's outputs, fitted on validation) | 0.587 |

With the share of images sent to stage 2 held fixed, the best of these gates was about as good as choosing images at random (stage 2 = average of both models):

| Share sent to stage 2 | Cost vs RAD-DINO (batch 32) | Random choice: AUROC / F1 | Entropy gate: AUROC / F1 |
|---:|---:|---|---|
| 30% | 0.48 | 0.8310 / 0.3619 | 0.8300 / 0.3660 |
| 50% | 0.68 | 0.8347 / 0.3690 | 0.8327 / 0.3728 |
| 80% | 0.98 | 0.8394 / 0.3757 | 0.8386 / 0.3804 |

The gate is slightly better than random on F1 and slightly worse on AUROC. A gate that knew the true labels would have reached AUROC 0.855 with only 30% of images routed. So the room is there, but the CNN's outputs do not contain the information needed to use it. A Grad-CAM heatmap would likely not change this, because it is built from the same CNN features (not tested).

### 7.3 Simple gate changes

**Table 5. Other gates on the test set** (stage-2 thresholds not re-tuned, so values can differ slightly from T4).

| Gate | Test: share to RAD-DINO | Cost vs RAD-DINO (batch 32) | Macro AUROC | Macro F1 |
|---|---:|---:|---:|---:|
| Selected gate (unsure = probability 0.2–0.8) | 77.6% | 0.95 | 0.8325 | 0.371 |
| Unsure = any probability below 0.8 | 100% | 1.18 | 0.8364 | 0.379 |
| Lower edge at each finding's own threshold, upper edge 0.8 | 83.2% | 1.01 | 0.8330 | 0.376 |
| "Expected mistakes" gate, average as stage 2 * | 83.8% | 1.01 | 0.8382 | 0.382 |
| *RAD-DINO alone* | *100%* | *1.00* | *0.8364* | *0.379* |

\* This gate adds up, over the 14 findings, the chance that the CNN's yes/no answer is wrong (possible because the probabilities are calibrated), and routes an image when the total is high. Its cut-off was chosen on validation, where it sent 53% of images; on test it sent 84%. It was significantly better than the selected cascade (AUROC +0.0057, 95% CI +0.0045 to +0.0069) but only tied RAD-DINO alone (+0.0017, CI −0.0012 to +0.0043) at the same cost. It used a 0.4 / 0.6 CNN / RAD-DINO mix that looked best on test, so its numbers are optimistic.

**Conclusion.** Lowering the "confident No" limit catches more misses but removes the saving. The only place a gate can save compute, the CNN's "confident No" zone, is exactly where its misses are. No gate we tried beat RAD-DINO alone on cost at equal accuracy.

---

## 8. Distillation: RAD-DINO as a teacher instead of a second stage

*Rules fixed before the run: [evaluation_protocol.yaml](evaluation_protocol.yaml) v2.2.0, section `distillation`. Code: [4-distillation.ipynb](4-distillation.ipynb). Results: [results_distill/](results_distill/).*

### 8.1 Design

| Part | Choice |
|---|---|
| **Idea** | Move RAD-DINO's knowledge into the CNN during training, instead of routing images at test time |
| **Teacher** | Mean of the calibrated probabilities of the trained CNN and RAD-DINO (the "Average" system of Section 3), run once over the 73,916 training images without augmentation |
| **Student (M10)** | ConvNeXt-Tiny at 384 px, from ImageNet weights, with exactly the CNN's training recipe. Only the training target changes. |
| **Training target** | 0.5 × real label + 0.5 × teacher probability, with the same ASL loss. Early stopping on validation macro AUROC against the real labels. |
| **Go / no-go** | Train only if the teacher's macro AUROC on the training images is below 0.95. Above that, its soft labels would just copy the real labels. |
| **Outcomes** | *Helps*: significantly better than the CNN. *Matches RAD-DINO*: lower 95% CI bound of (Distilled − RAD-DINO) macro AUROC above −0.005 (half the CNN-vs-RAD-DINO gap). *Beats RAD-DINO*: significantly better than RAD-DINO. |
| **Evaluation** | As in Section 1: Platt calibration and F1 thresholds fitted on validation, the same 1,000 patient bootstrap resamples, Holm correction over the 3 comparisons per metric |

The equal-weight teacher is the pre-planned Average system. A 0.4 / 0.6 mix looked slightly better on test in the exploratory analysis and was deliberately not used.

### 8.2 Go / no-go and training

| | Macro AUROC, training images | Macro AUROC, validation |
|---|---:|---:|
| CNN | 0.9018 | 0.8598 |
| RAD-DINO | 0.9124 | 0.8651 |
| **Teacher (average)** | **0.9165** | **0.8735** |

- **Decision: go.** The teacher is somewhat overfit (0.917 on training images vs 0.874 on validation), but well below the 0.95 stop line. On training images it gave true positives an average probability of 0.28 and negatives 0.03, so its soft labels are far from 0 and 1 and carry information the real labels do not.
- **The teachers were rebuilt exactly.** Both reproduced their saved validation outputs (correlation 1.00000).
- **Training.** The student trained for 2.6 hours on a Tesla T4, plus 50 minutes for the teacher pass. Its best validation macro AUROC was **0.8640** (epoch 6), against 0.8598 for the CNN and 0.8651 for RAD-DINO.
- **Less over-fitting.** After its best epoch, the CNN's validation loss rose (0.305 → 0.316). The student's stayed flat (0.300 → 0.301). This fits the idea that soft labels soften the effect of noisy labels.

### 8.3 Test results

**Table 6. Distilled CNN and reference systems on the test set (value [95% CI]).**

| System | Macro AUROC | Macro AUPRC | Macro F1 | Micro F1 | Macro recall | Macro precision | Cost vs RAD-DINO, batch 32 / 1 |
|---|---|---|---|---|---|---|---|
| CNN alone | 0.8263 [0.8200, 0.8317] | 0.2988 [0.2849, 0.3154] | 0.3539 [0.3342, 0.3686] | 0.4172 [0.4090, 0.4247] | 0.4304 [0.4113, 0.4478] | 0.3188 [0.2965, 0.3350] | 0.18 / 0.51 |
| **Distilled CNN** | **0.8339** [0.8283, 0.8387] | **0.3146** [0.3010, 0.3296] | **0.3702** [0.3524, 0.3842] | 0.4192 [0.4109, 0.4268] | **0.4622** [0.4428, 0.4789] | 0.3326 [0.3134, 0.3492] | **0.18 / 0.51** |
| RAD-DINO alone | 0.8364 [0.8303, 0.8415] | 0.3223 [0.3103, 0.3366] | 0.3788 [0.3653, 0.3900] | 0.4317 [0.4238, 0.4391] | 0.4616 [0.4443, 0.4779] | 0.3362 [0.3235, 0.3491] | 1.00 / 1.00 |
| Average | 0.8428 [0.8373, 0.8478] | 0.3292 [0.3160, 0.3431] | 0.3818 [0.3674, 0.3928] | 0.4421 [0.4341, 0.4495] | 0.4561 [0.4389, 0.4724] | 0.3411 [0.3273, 0.3546] | 1.18 / 1.51 |
| *Cascade (t = 0.80)* | *0.8325 [0.8264, 0.8379]* | *0.3176 [0.3048, 0.3315]* | *0.3709 [0.3565, 0.3822]* | *0.4273 [0.4192, 0.4347]* | *0.4454 [0.4281, 0.4622]* | *0.3401 [0.3272, 0.3536]* | *0.95 / 1.29* |

The CNN, RAD-DINO and Average rows were recomputed in the distillation evaluation and are identical to Table 1. The cascade row is copied from Table 1 for reference; it was not part of the distillation comparison.

**Table 7. Paired differences, distilled CNN minus each reference (A − B [95% CI]).** All differences are significant except those marked "n.s.".

| Metric | Distilled − CNN | Distilled − RAD-DINO | Distilled − Average |
|---|---|---|---|
| Macro AUROC | **+0.0076** [+0.0058, +0.0096] | −0.0025 [−0.0056, +0.0004] n.s. | −0.0089 [−0.0112, −0.0071] |
| Macro AUPRC | +0.0158 [+0.0109, +0.0212] | −0.0077 [−0.0180, +0.0017] n.s. | −0.0146 [−0.0223, −0.0078] |
| Macro F1 | +0.0163 [+0.0103, +0.0230] | −0.0086 [−0.0204, +0.0036] n.s. | −0.0116 [−0.0218, −0.0026] |
| Micro F1 | +0.0020 [−0.0012, +0.0052] n.s. | −0.0126 [−0.0169, −0.0082] | −0.0230 [−0.0261, −0.0195] |
| Macro recall | +0.0318 [+0.0247, +0.0389] | +0.0006 [−0.0138, +0.0165] n.s. | +0.0061 [−0.0060, +0.0175] n.s. |
| Macro precision | +0.0137 [+0.0050, +0.0278] | −0.0036 [−0.0189, +0.0096] n.s. | −0.0086 [−0.0202, +0.0002] n.s. |

What the tables show:

- **Outcome: "helps".** The student is significantly better than the CNN on AUROC, AUPRC, macro F1, recall and precision.
- **"Matches RAD-DINO" was missed by 0.0006.** The lower CI bound was −0.0056 against the −0.005 margin. The student is not significantly worse than RAD-DINO on any macro metric (AUROC Holm p = 0.078), but by the rule fixed in advance it does not match it.
- **The average of both models is still clearly better** than the student on every ranking metric.
- **Validation and test agree.** The gain over the CNN was +0.0042 on validation and +0.0076 on test, so it is not a validation artefact.
- **There is a trade-off.** Macro specificity fell slightly (0.9040 → 0.8993) as recall rose, and micro F1 did not change significantly. Section 8.4 explains why.

### 8.4 Where the gain comes from (exploratory)

**Share of the CNN → RAD-DINO gap closed (test set):**

| Metric | CNN | RAD-DINO | Distilled | Share of gap closed |
|---|---:|---:|---:|---:|
| Macro AUROC | 0.8263 | 0.8364 | 0.8339 | 75% |
| Macro AUPRC | 0.2988 | 0.3223 | 0.3146 | 67% |
| Macro F1 | 0.3539 | 0.3788 | 0.3702 | 65% |
| Micro F1 | 0.4172 | 0.4317 | 0.4192 | 14% |
| Macro precision | 0.3188 | 0.3362 | 0.3326 | 79% |
| Macro recall | 0.4304 | 0.4616 | 0.4622 | 102% |

**Table 8. Test AUROC per finding.** "Share closed" is shown only where RAD-DINO leads the CNN by at least 0.005.

| Finding | Test positives | CNN | RAD-DINO | Distilled | Share of gap closed |
|---|---:|---:|---:|---:|---:|
| Infiltration | 6,112 | 0.709 | 0.709 | 0.711 | – |
| Effusion | 4,658 | 0.839 | 0.843 | 0.841 | – |
| Atelectasis | 3,279 | 0.790 | 0.798 | 0.796 | 80% |
| Pneumothorax | 2,665 | 0.882 | 0.904 | 0.889 | 31% |
| Consolidation | 1,815 | 0.761 | 0.761 | 0.767 | – |
| Mass | 1,748 | 0.846 | 0.851 | 0.851 | 94% |
| Nodule | 1,623 | 0.798 | 0.803 | 0.803 | – |
| Pleural thickening | 1,143 | 0.794 | 0.814 | 0.801 | 37% |
| Emphysema | 1,093 | 0.936 | 0.949 | 0.939 | 19% |
| Cardiomegaly | 1,069 | 0.888 | 0.898 | 0.895 | 70% |
| Edema | 925 | 0.858 | 0.856 | 0.864 | – |
| Pneumonia | 555 | 0.727 | 0.729 | 0.735 | – |
| Fibrosis | 435 | 0.829 | 0.849 | 0.843 | 68% |
| Hernia ⚠ | 86 | 0.910 | 0.944 | 0.940 | 87% |

⚠ Fewer than 100 positive test images.

- **Fine-detail findings close least:** Emphysema 19%, Pneumothorax 31%, Pleural thickening 37%. These are exactly where RAD-DINO had its biggest lead (Section 6). A plausible reading is that RAD-DINO's advantage there comes from its 518 px input and X-ray pretraining, which a student looking at 384 px cannot fully copy. This was not tested.
- **Where RAD-DINO had no lead, the student is above both single models** (Infiltration, Consolidation, Edema, Pneumonia). This fits soft labels smoothing noisy labels.
- Per-finding differences were not tested for significance.

**Did distillation fix the silent misses?** Counts of positive labels on the test set, each system at its own F1 thresholds:

| System | True positives | False positives | False negatives |
|---|---:|---:|---:|
| CNN alone | 14,884 | 29,267 | 12,322 |
| Distilled CNN | 15,466 | 31,124 | 11,740 |
| RAD-DINO alone | 15,715 | 29,883 | 11,491 |
| Average | 15,525 | 27,500 | 11,681 |

- **Partly.** Of the 12,322 positives the CNN missed, RAD-DINO finds 2,831 and the student 1,517, about half as many. Of the 2,831 positives that RAD-DINO found and the CNN missed, the student catches 936 (33%).
- **The price is more false alarms.** The student finds 582 more true positives than the CNN (RAD-DINO: +831) but also raises 1,857 more false positives, because its per-finding thresholds (tuned for F1 on validation) sit lower. That is why macro F1 rises while micro F1 hardly moves.
- **The student is still a CNN, pulled toward RAD-DINO.** Its outputs correlate 0.97 with the CNN's and 0.84 with RAD-DINO's (the CNN and RAD-DINO correlate 0.81 with each other).

---

## 9. All systems side by side

**Table 9. Accuracy against test-time cost (test set).**

| System | Cost vs RAD-DINO, batch 32 | Cost vs RAD-DINO, batch 1 | Macro AUROC | Macro F1 | Comment |
|---|---:|---:|---:|---:|---|
| CNN alone | 0.18 | 0.51 | 0.8263 | 0.3539 | Cheapest, least accurate |
| **Distilled CNN** | **0.18** | **0.51** | **0.8339** | **0.3702** | CNN cost, most of RAD-DINO's accuracy |
| Cascade t = 0.80 | 0.95 | 1.29 | 0.8325 | 0.3709 | Same accuracy as the distilled CNN at about 5× the cost |
| RAD-DINO alone | 1.00 | 1.00 | 0.8364 | 0.3788 | Best single model |
| Average of both | 1.18 | 1.51 | 0.8428 | 0.3818 | Most accurate; runs both models |

- **The cascade is dominated.** The distilled CNN reaches its accuracy at about one fifth of the cost, and needs one model in memory instead of two. Distilled vs cascade was not tested for significance.
- **Three systems are worth considering:** the distilled CNN (cheap), RAD-DINO (best single model) and the average (most accurate).
- **The one-off training cost is higher for the distilled CNN.** It needs both trained models first (2.8 h + 6.6 h), then 3.4 hours for the teacher pass and student training, all on one T4. Test-time cost is what the table compares.
- **All five systems lie within 0.017 AUROC of each other.** The labels were mined from report text and are noisy, so part of the ceiling may come from the labels rather than the models.

---

## 10. What this means

**For the cascade question, the answer is no.** On the official NIH test set, the ConvNeXt-Tiny → RAD-DINO cascade gives neither a useful compute saving nor RAD-DINO-level accuracy. A cascade's saving depends on how many images the first model finds "easy". That share is a property of the data, not of the model: here it fell from 56% of images on validation to 22% on test, because the test population was sicker and more varied. And with this CNN, no gate can find the images it gets wrong, because it does not know when it has missed a finding.

**Moving RAD-DINO to training time works better.** Distillation gives a CNN that keeps about three quarters of RAD-DINO's AUROC advantage at the CNN's cost. It does not formally reach RAD-DINO, and it pays for its higher recall with more false alarms.

**What we would recommend for these models:**

1. **If compute is limited, use the distilled CNN.** It is significantly better than the original CNN at the same cost (0.18× RAD-DINO in batch use, 0.51× one image at a time), and it matches the cascade's accuracy for one fifth of the cascade's cost.
2. **If you want the best single model and can afford it, use RAD-DINO alone.** It is 0.0025 AUROC above the distilled CNN (not significant) and has better micro F1.
3. **If accuracy matters most, use the average of both models.** It is the most accurate system: +0.006 AUROC and +0.010 micro F1 over RAD-DINO alone, at 1.18× the compute in batch use (1.51× one image at a time).
4. **Do not use the cascade with these models.** If a cascade is ever needed, tune the gate on data that looks like the target population, keep the "confident No" limit at or below each finding's own threshold, and monitor the routing rate on incoming data.

---

## 11. Limitations

- **One training seed.** The confidence intervals cover only the randomness of which test patients were sampled. They say nothing about how much results would change if the models were retrained. Small differences, such as the cascade's −0.004 AUROC or the distilled CNN's −0.0025 against RAD-DINO, could change size with another seed.
- **Noisy labels.** NIH labels were mined from radiology report text by software, not read from the images by radiologists. The metrics measure agreement with those text-mined labels, not with clinical truth. This is not a clinical validation.
- **One dataset.** No external test set was used, so we cannot say how any system behaves on X-rays from another hospital.
- **Compute is estimated from forward-pass times only** (see Section 4). Real end-to-end costs would favour the cascade even less.
- **The distillation teacher saw its own training images.** Its soft labels are sharper on those images than on new ones (training AUROC 0.917 vs validation 0.874), which may limit what the student learns.
- **The "matches RAD-DINO" margin (0.005) is a judgement call**, set at half the CNN-vs-RAD-DINO gap, as in the layer-depth protocol.
- **Some analyses were done after seeing the test results.** These are the routing breakdown (5.1), the missed-label comparison with RAD-DINO (5.2), the gate-threshold overlap (5.2), the average-as-second-stage observation (5.3), all of Section 7, and Section 8.4. They explain the main results, but they were not planned in advance and should be treated as exploratory.
- **These numbers must not be compared with the team models in [DM_Workbook.xlsx](DM_Workbook.xlsx) sheet 3.** Those models were tested on a different, random patient split that is easier than the official test list.

---

## 12. Checks and source files

All seven blocking quality checks of the cascade study passed:

- official split sizes and list-file hashes match;
- no patient appears in more than one partition;
- the split hash matches the hash stored in both models' outputs;
- both models were scored on identical validation and test images and labels;
- the label order matches the protocol;
- calibration, thresholds and gate were fitted on validation only;
- every macro value equals the mean of its per-finding values.

One warning stands: a single training seed (QC-09). Timing for both models came from the same benchmark run on the same GPU type (QC-08 passed).

All four blocking checks of the distillation study passed: both teacher checkpoint hashes match (QC-D1); both rebuilt teachers reproduce their saved validation logits exactly (QC-D2); soft labels exist for training images only (QC-D3); the student was scored on the same validation and test images, labels and split as the CNN (QC-D4).

| File | Contents |
|---|---|
| [results_s42/T1_dataset_characteristics.csv](results_s42/T1_dataset_characteristics.csv) | Population of each partition (Section 2) |
| [results_s42/T2_results.csv](results_s42/T2_results.csv) | All metrics, validation and test, with CIs (Section 3) |
| [results_s42/T3_paired_comparisons.csv](results_s42/T3_paired_comparisons.csv) | Paired differences and Holm-corrected p-values (Table 2) |
| [results_s42/T4_cascade_sweep.csv](results_s42/T4_cascade_sweep.csv) | All 23 gate settings (Figure 1, Section 5.3) |
| [results_s42/T5_per_class.csv](results_s42/T5_per_class.csv) | Per-finding results (Table 4) |
| [results_s42/T6_compute.csv](results_s42/T6_compute.csv) | Latency, parameters, training time (Table 3) |
| [results_s42/qc_report.csv](results_s42/qc_report.csv) | Quality-control results |
| [results_s42/calibration.json](results_s42/calibration.json), [results_s42/thresholds.json](results_s42/thresholds.json) | Fitted calibration, thresholds and the selected gate |
| [2-staged_official_split.ipynb](2-staged_official_split.ipynb) | Code for the cascade study |
| [results_distill/T10_distill_results.csv](results_distill/T10_distill_results.csv) | Distillation: all metrics, validation and test, with CIs (Table 6) |
| [results_distill/T11_distill_paired.csv](results_distill/T11_distill_paired.csv) | Distillation: paired differences and Holm-corrected p-values (Table 7) |
| [results_distill/distill_outcome.json](results_distill/distill_outcome.json), [results_distill/distill_go_no_go.json](results_distill/distill_go_no_go.json), [results_distill/qc_session_D.json](results_distill/qc_session_D.json) | Outcome against the rules, go/no-go decision, quality checks |
| [4-distillation.ipynb](4-distillation.ipynb) | Code for the distillation study (teacher labels, student training, evaluation) |
| [exploratory/](exploratory/) | Scripts for the post hoc analyses (Sections 7 and 8.4) |
| [DM_Workbook.xlsx](DM_Workbook.xlsx), sheet 7 | Distillation results in the workbook |

Training took 2.8 hours for ConvNeXt-Tiny, 6.6 hours for RAD-DINO and 2.6 hours for the distilled CNN, each on one Tesla T4. Model selection used the best validation macro AUROC: fine-tuning epoch 5 of 7 for the CNN, epoch 2 of 4 for RAD-DINO, and fine-tuning epoch 5 of 7 for the distilled CNN.
