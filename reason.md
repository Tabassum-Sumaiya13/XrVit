 I'll show you exactly how the CNN processes **one real image from your test set**.

---

## Example: One Chest X-ray Image

Let me imagine a test patient's X-ray. I'll show what happens at each step.

### **Step 0: The Image Arrives**

```
Input: Chest X-ray (frontal view, 384×384 pixels)
       ConvNeXt-Tiny processes it
```

---

### **Step 1: CNN Forward Pass (Raw Outputs)**

The CNN has learned to output 14 numbers (one for each finding). These are called **logits** (before calibration):

```
The CNN reads the image and outputs:

Atelectasis        →  raw logit = 0.32
Cardiomegaly       →  raw logit = -0.18
Effusion           →  raw logit = 1.45
Infiltration       →  raw logit = 0.88
Mass               →  raw logit = -0.55
Nodule             →  raw logit = 0.02
Pneumonia          →  raw logit = -1.20
Pneumothorax       →  raw logit = 2.10
Consolidation      →  raw logit = 0.65
Edema              →  raw logit = -0.40
Emphysema          →  raw logit = 1.92
Fibrosis           →  raw logit = -0.70
Pleural Thickening →  raw logit = 0.15
Hernia             →  raw logit = -2.30
```

These are **not probabilities yet**. They're scores. Positive = "leans toward YES", negative = "leans toward NO".

---

### **Step 2: Calibration (Converting to Probability)**

Your calibration file has learned a formula for each finding:

```
For Atelectasis:
  a = 2.600  (learned weight)
  b = -1.723 (learned offset)
  
Probability = 1 / (1 + exp(-(a × logit + b)))
```

**Applying calibration:**

```
Atelectasis:
  P = 1 / (1 + exp(-(2.600 × 0.32 - 1.723)))
  P = 1 / (1 + exp(-0.832 + 1.723))
  P = 1 / (1 + exp(0.891))
  P = 1 / (1 + 2.44)
  P ≈ 0.29 ✓ probability
```

**After calibration, the 14 probabilities are:**

```
Finding                Probability
─────────────────────────────────
Atelectasis            0.29
Cardiomegaly           0.09
Effusion               0.82
Infiltration           0.70
Mass                   0.12
Nodule                 0.51
Pneumonia              0.03
Pneumothorax           0.89
Consolidation          0.65
Edema                  0.08
Emphysema              0.87
Fibrosis               0.05
Pleural Thickening     0.54
Hernia                 0.01
```

These are **calibrated probabilities**: 0.0 = "definitely NO", 1.0 = "definitely YES", 0.5 = "no idea".

---

### **Step 3: The Gate Decision**

Now the **gate checks all 14 probabilities**.

**Gate rule from your thresholds.json:**
```
For EACH finding:
  - If probability is between 0.2 and 0.8  → "unsure"
  - Otherwise                             → "confident"

If ANY finding is "unsure":
  → Send to RAD-DINO
Otherwise:
  → CNN decides alone
```

**Checking this image:**

```
Atelectasis            0.29  ✓ between 0.2–0.8 → UNSURE
Cardiomegaly           0.09  ✗ below 0.2       → confident NO
Effusion               0.82  ✗ above 0.8       → confident YES
Infiltration           0.70  ✓ between 0.2–0.8 → UNSURE ⚠️
Mass                   0.12  ✗ below 0.2       → confident NO
Nodule                 0.51  ✓ between 0.2–0.8 → UNSURE
Pneumonia              0.03  ✗ below 0.2       → confident NO
Pneumothorax           0.89  ✗ above 0.8       → confident YES
Consolidation          0.65  ✓ between 0.2–0.8 → UNSURE
Edema                  0.08  ✗ below 0.2       → confident NO
Emphysema              0.87  ✗ above 0.8       → confident YES
Fibrosis               0.05  ✗ below 0.2       → confident NO
Pleural Thickening     0.54  ✓ between 0.2–0.8 → UNSURE
Hernia                 0.01  ✗ below 0.2       → confident NO

COUNT: 5 findings are "unsure"
```

**Gate decision:**
```
ANY finding unsure? → YES (Atelectasis, Infiltration, Nodule, Consolidation, Pleural Thickening)
                   ↓
ROUTE TO RAD-DINO
```

---

### **Step 4: What The CNN Would Have Decided (If It Went Alone)**

But the gate says "you're unsure about too many things", so the CNN's answer is **thrown away**. Let me show what it would have decided:

```
Using CNN's thresholds from thresholds.json:

Finding                CNN P    Threshold   Decision
──────────────────────────────────────────────────
Atelectasis            0.29  ≥  0.1950   → YES ✓
Cardiomegaly           0.09  <  0.1567   → NO
Effusion               0.82  ≥  0.3250   → YES ✓
Infiltration           0.70  ≥  0.1643   → YES ✓
Mass                   0.12  <  0.1370   → NO
Nodule                 0.51  ≥  0.1935   → YES ✓
Pneumonia              0.03  <  0.0633   → NO
Pneumothorax           0.89  ≥  0.1975   → YES ✓
Consolidation          0.65  ≥  0.1181   → YES ✓
Edema                  0.08  <  0.1482   → NO
Emphysema              0.87  ≥  0.0986   → YES ✓
Fibrosis               0.05  <  0.1041   → NO
Pleural Thickening     0.54  ≥  0.1325   → YES ✓
Hernia                 0.01  <  0.1273   → NO
```

**CNN alone would predict:** Atelectasis, Effusion, Infiltration, Nodule, Pneumothorax, Consolidation, Emphysema, Pleural Thickening (8 findings).

---

### **Step 5: RAD-DINO Decides Instead**

Since the gate said "unsure", the image is **sent to RAD-DINO** (the expensive transformer).

RAD-DINO reads the same image and outputs 14 different probabilities (higher resolution, better model):

```
Finding                RAD-DINO P
──────────────────────────────
Atelectasis            0.31
Cardiomegaly           0.08
Effusion               0.71
Infiltration           0.68
Mass                   0.14
Nodule                 0.49
Pneumonia              0.05
Pneumothorax           0.91
Consolidation          0.61
Edema                  0.09
Emphysema             0.89
Fibrosis               0.04
Pleural Thickening     0.52
Hernia                 0.02
```

RAD-DINO applies its own thresholds and decides the same 14 findings.

---

### **Step 6: Final Answer (What Gets Reported)**

```
The cascade's final answer = RAD-DINO's answer:

Atelectasis        ✓ YES   (P=0.31)
Cardiomegaly       ✗ NO    (P=0.08)
Effusion           ✓ YES   (P=0.71)
Infiltration       ✓ YES   (P=0.68)
Mass               ✗ NO    (P=0.14)
Nodule             ✓ YES   (P=0.49)
Pneumonia          ✗ NO    (P=0.05)
Pneumotharax       ✓ YES   (P=0.91)
Consolidation      ✓ YES   (P=0.61)
Edema              ✗ NO    (P=0.09)
Emphysema          ✓ YES   (P=0.89)
Fibrosis           ✗ NO    (P=0.04)
Pleural Thickening ✓ YES   (P=0.52)
Hernia             ✗ NO    (P=0.02)

Cost: CNN (4.0 ms) + RAD-DINO (22.8 ms) = 26.8 ms per image
```

---

## **Now Here's The Problem**

Suppose the **true labels** for this image were:

```
Atelectasis        ✓ YES    (ground truth)
Effusion           ✓ YES
Infiltration       ✓ YES
Pneumothorax       ✗ NO     ← CNN and RAD-DINO both wrong
Emphysema          ✓ YES
Nodule             ✗ NO     ← RAD-DINO wrong
... rest correct
```

In this case:
- **RAD-DINO made 2 errors** (Pneumothorax, Nodule)
- **CNN made 1 error** (Pneumothorax)
- **But the cascade cost 26.8 ms** and didn't help

---

## **The Silent Failure Case** (Why Your F1 Dropped)

Now imagine a different image where:

```
CNN probabilities:

Infiltration       0.12    ← below 0.2, CNN says "confident NO"
Effusion           0.30    ← between 0.2–0.8, CNN says "unsure"
(other findings: clearly confident)

Gate decision: "Infiltration is confident, but Effusion is unsure"
             → ROUTE TO RAD-DINO
```

**What happens:**

```
True labels:
Infiltration       ✓ YES    ← CNN missed it
Effusion           ✓ YES

CNN's answer (if it went alone):
Infiltration       ✗ NO     ← wrong, but gate doesn't know
Effusion           ✓ YES    ← correct (routed anyway)

RAD-DINO's answer:
Infiltration       ✓ YES    ← caught the miss
Effusion           ✓ YES
```

**The issue:** The CNN output probability 0.12 because it **looked at the infiltration and didn't see it**. So it's not uncertain—it's wrong. But the gate only sees "probability = 0.12" and thinks "confident NO".

---

## Summary: What The CNN Does on One Image

| Step | What happens | Time |
|---|---|---|
| 1 | Image → CNN neural network | 4.0 ms |
| 2 | CNN outputs 14 raw scores | instant |
| 3 | Calibration converts to probabilities | instant |
| 4 | Gate checks: any probability between 0.2–0.8? | instant |
| 5a | If yes: throw away CNN, run RAD-DINO instead | +22.8 ms |
| 5b | If no: keep CNN decision, skip RAD-DINO | 0 ms |
| 6 | Return final predictions | instant |

**The problem: When CNN is wrong (confidently), the gate still thinks it's confident and keeps it.**



## 1. The Gating Mechanism (How It Works)

Your cascade uses this rule:

```
For each of 14 findings on the image:
  - Calculate CNN's calibrated probability
  
If ANY finding has probability between 0.2 and 0.8:
  → Send entire image to RAD-DINO
  → RAD-DINO decides ALL 14 findings
  
Else (all findings outside 0.2–0.8):
  → CNN decides, skip RAD-DINO
```

**In plain terms:** If the CNN is unsure about *even one* finding, the whole image gets the expensive model.

---

## 2. Why Results Are Lower (Accuracy Loss)

On your test set:

| Metric | RAD-DINO alone | Your cascade | Loss |
|---|---|---|---|
| Macro AUROC | 0.8364 | 0.8325 | −0.0039 |
| Macro F1 | 0.3788 | 0.3709 | −0.0079 |
| Macro recall | 0.4616 | 0.4454 | −0.0162 |

Where does this loss come from? **Your own data shows:**

```
5,742 test images (22.4% of test)
    ↓
CNN says "confident" (routed to stay with CNN only)
    ↓
These images contain 2,707 positive findings
    ↓
Cascade finds:        114 of them
RAD-DINO alone finds: 756 of them
    ↓
Missed: 2,593 − 114 = 2,593 positives lost
```

**This is your recall gap.** The CNN is confidently wrong.

### Example of what's happening:

Imagine an X-ray with **Infiltration** (lung haziness). 

- CNN outputs: "probability = 0.05" (confident: NO)
- RAD-DINO outputs: "probability = 0.42" (uncertain)
- Ground truth: YES (infiltration is there)

**Your cascade decision:**
- All CNN probabilities are ≤ 0.2, so gate says: "CNN is confident about everything"
- Image stays with CNN → **Infiltration marked as NO** ❌
- RAD-DINO never sees it

**Why this happens:** The CNN looked at the lung and genuinely missed the haziness. It's not uncertain—it's confidently wrong.

---

## 3. Why Compute Didn't Save (Cost Problem)

You designed the cascade expecting:

```
Validation:  "44% of images go to RAD-DINO"
             → save 38% of compute ✓

Deploy on test:  "Should be about the same"
                 → expect 38% compute saving
```

**What actually happened:**

```
Test reality:  77.6% of images went to RAD-DINO
               → only 5% compute saving ❌
               → at batch size 1: 29% SLOWER than RAD-DINO alone ❌
```

### Why did 44% → 77.6%?

Your test data is **much sicker** than validation:

| | Validation | Test | Change |
|---|---|---|---|
| Findings per image | 0.61 | 1.06 | +74% |
| Images with no findings | 59% | 39% | −20% |
| PA view (easier images) | 66% | 43% | −23% |

**How this breaks the gate:**

```
VALIDATION (sicker patients rare):
- Most images: 0–1 finding
- CNN: "I'm confident about all 14 findings"
- Gate: "All probabilities outside 0.2–0.8, skip RAD-DINO"
- Result: 44% sent to stage 2

TEST (sicker patient population):
- Most images: 1–2 findings
- CNN: "Hmm, some findings are hard to judge"
- Gate: "Some probabilities between 0.2–0.8, use RAD-DINO"
- Result: 77.6% sent to stage 2
```

**The gate was tuned on validation, then deployed on different data.** This is called **distribution shift**.

---

## 4. The Actual Root Problem (Why It Can't Be Fixed With a Better Gate)

Here's what I tested on your saved logits:

| Gate type | At 30% budget | At 50% budget | At 80% budget |
|---|---|---|---|
| Your current gate | F1: 0.3576 | F1: 0.3624 | F1: 0.3597 |
| Entropy gate (different rule) | F1: 0.3660 | F1: 0.3728 | F1: 0.3766 |
| "Learned" gate (AI-trained) | F1: 0.3701 | F1: 0.3745 | F1: 0.3770 |
| **Random routing** (flip a coin) | F1: 0.3619 | F1: 0.3690 | F1: 0.3757 |
| Oracle gate (uses test labels) | F1: 0.4018 | F1: 0.4062 | F1: 0.4056 |

**The gates barely beat random.** And here's the critical part:

```
Can any real gate identify which images RAD-DINO helps?
(Measured by AUROC: 0.5 = random, 1.0 = perfect)

Current gate:  0.587
Entropy gate:  0.626
Learned gate:  0.587
Random:        0.500
```

**The problem:** When the CNN misses a positive finding, it doesn't **feel** uncertain. The CNN outputs a confident probability (say, 0.08) because it looked at the finding and didn't see it.

So the gate can't tell the difference:

```
CNN says "probability = 0.08":

Case A: Image really has no finding (CNN is RIGHT)
        Gate: "Confident negative" ✓

Case B: Image has the finding, CNN missed it (CNN is WRONG)
        Gate: "Confident negative" ✗
        
Gate can't tell A and B apart from probability alone.
```

**This is why better gates don't help.** The CNN's confidence score doesn't measure "am I right?" It measures "how sure am I about what I see?" Missing something means you're not uncertain about it—you're wrong.

---

## 5. Summary: What's Actually Broken

| Problem | Evidence |
|---|---|
| **CNN misses findings silently** | 89% of RAD-DINO's added value comes from positives the CNN scored low |
| **No gate can fix it** | Best real gate (entropy) ≈ random routing |
| **The gate isn't the real problem** | Even if you tuned it perfectly on test data, you'd still lose accuracy |
| **Cost shift is secondary** | Even at a *fixed* 30% budget, the cascade still underperforms |

---

## 6. Why My Solution (Distillation) Works

Instead of trying to route at test time (which fails), teach the CNN **during training**:

```
TRAINING:
  All 73,916 train images
       ↓
  Run through both models
       ↓
  Create soft targets: 0.4×CNN_output + 0.6×RAD-DINO_output
       ↓
  Train new ConvNeXt with these soft targets
       ↓
  
TEST TIME:
  New CNN (taught by RAD-DINO)
       ↓
  4.0 ms, 0.18× cost, no gating
```

**Why this works:** The CNN learns from RAD-DINO's errors during training, so it doesn't miss things silently anymore.

