# Sudoku Masked Diffusion

Part of a project comparing autoregressive and diffusion-based approaches to constraint satisfaction problems.

---

## What This Is

A discrete masked diffusion model trained to solve Sudoku puzzles. Given cells are fixed; unknown cells are randomly masked during training and iteratively unmasked at inference in order of model confidence.

---

## Architecture

| Component | Value |
|---|---|
| Sequence length | 81 tokens (one per cell, row-major) |
| Input vocabulary | 11 tokens (digits 0–9 + MASK token 10) |
| Output vocabulary | 10 tokens (digits 0–9) |
| Embedding dim | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Feedforward dim | 512 |
| Attention | Bidirectional — no causal mask |
| Total parameters | 806,154 |

The model uses a **TransformerEncoder** — every cell attends to every other cell simultaneously. This is the architectural property that eliminates the position gradient observed in the autoregressive baseline.

---

## Training

```
Dataset:      Kaggle 1M Sudoku dataset (bryanpark/sudoku)
              500,000 puzzles used for training
Noise:        Per-puzzle random masking density, unknown cells only
              Given cells never masked
Epochs:       20
Batch:        64
Optimizer:    Adam, lr=1e-3
Loss:         CrossEntropyLoss on masked positions only
Device:       Apple MPS (M-series Mac)
Time:         ~9 hours
```

The dataset is uniformly easy. All puzzles have 31–36 givens (avg 33.8). No hard puzzles in the training distribution.
---

## Results

**Easy puzzles (31–36 givens, n=500,000) — one-shot inference:**

```
Cell accuracy:   98.76%
Puzzle accuracy: 68.28%
```

**Position accuracy grid (unknown cells only) — flat, no gradient:**

```
98.7 | 98.8 | 98.8 | 98.9 | 98.7 | 98.7 | 98.8 | 98.8 | 98.7
98.7 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.8 | 98.8
98.7 | 98.7 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8
98.7 | 98.8 | 98.7 | 98.8 | 98.7 | 98.8 | 98.8 | 98.9 | 98.8
98.7 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.8 | 98.8
98.8 | 98.8 | 98.8 | 98.9 | 98.8 | 98.8 | 98.8 | 98.8 | 98.8
98.8 | 98.8 | 98.8 | 98.8 | 98.7 | 98.7 | 98.8 | 98.7 | 98.7
98.7 | 98.7 | 98.8 | 98.8 | 98.7 | 98.8 | 98.8 | 98.7 | 98.8
98.7 | 98.7 | 98.8 | 98.7 | 98.7 | 98.7 | 98.8 | 98.7 | 98.7
```

Compare to AR position accuracy baseline: 85.6% (A1) → 99.6% (I9), a 14 percentage point gradient.

**Iterative confidence-based decoding (n=1000 sample):**

| k | Puzzle accuracy | Avg passes | Time |
|---|---|---|---|
| 1 | 100.00% | 47.2 | 253s |
| 5 | 100.00% | 10.0 | 93s |
| 10 | 99.90% | 5.0 | 66s |
| 20 | 99.90% | 3.0 | 57s |
| 81 (one-shot) | 67.40% | 1.0 | 49s |

**Constraint violation analysis (n=1000):**

| Model | Solved | Puzzles w/ violations | Avg violations |
|---|---|---|---|
| AR one-shot | 119/1000 | 881/1000 | 6.16 |
| Diffusion one-shot | 685/1000 | 315/1000 | 1.26 |
| Diffusion k=5 | 1000/1000 | 0/1000 | 0.00 |

**Hard puzzle (hard1, 17 givens — out of distribution):**

Both one-shot and AR produce 27/27 constraint violations. Iterative decoding (k=5) reduces to 8/27 violations. Single data point — hard puzzle dataset evaluation pending.

---

## Key Findings

**Position gradient eliminated.** Bidirectional attention gives every cell equal access to global context. The 14-point gradient in the AR position accuracy baseline disappears entirely.

**Constraint structure learned.** The diffusion model produces 5x fewer violations than AR even in one-shot inference. Iterative decoding reaches zero violations on 1000 easy puzzles.

**Iterative decoding is the mechanism.** Locking in high-confidence predictions first propagates constraints to uncertain cells — the learned analog of Norvig's minimum remaining values heuristic. k=5 is the practical optimum: 100% accuracy in 10 passes.

**Cell accuracy is a misleading metric for CSPs.** The AR model achieves 96.3% cell accuracy with 88% of puzzles containing constraint violations. Puzzle accuracy and violation rate are more meaningful measures of whether constraint structure has been learned, compared to numerical probability distribution.