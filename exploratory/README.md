# Exploratory analyses (post hoc, not part of the protocol)

These scripts produced the numbers in RESULTS.md sections 7 and 8.7. They were written **after** the test results were known, so their numbers explain the main results but are not confirmatory.

Run from `D:/XrVit/exploratory`; the scripts read `D:/XrVit/results_s42` and `D:/XrVit/results_distill` by absolute path.

| Script | What it does | RESULTS.md |
|---|---|---|
| `load.py` | Loads saved val/test logits, rebuilds the v2.0.0 calibrated probabilities (checked: reproduces T2 exactly) | – |
| `budget.py` | Gate types at fixed RAD-DINO budgets (current, entropy, near-threshold, learned, oracle) | 7.2 |
| `ext.py` | Fusion weights, how well each gate finds the images where stage 2 helps, share of benefit on positive labels | 7.2 |
| `rnd.py` | Gates vs random routing at the same budget | 7.2 |
| `gate80.py` | Simple gate variants (only the 0.8 cut-off, own-threshold lower edge) | 7.3 |
| `mygate.py`, `boot2.py` | Hand-coded "expected mistakes" gate and its patient bootstrap | 7.3 |
| `distill_analysis.py` | Distilled CNN: error counts, silent misses, agreement with the teachers | 8.7 |

Caveats: `mygate.py` uses a 0.4/0.6 CNN/RAD-DINO mix that looked best on **test**, so its result is optimistic. Stage-2 thresholds in `gate80.py`/`mygate.py` are each model's own thresholds, not re-tuned for the cascade as the notebook does, so numbers differ slightly from T4.
