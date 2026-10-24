# Assignment 2

Full task spec: [`assignment_description.md`](assignment_description.md)

## Files

| File | Content |
|---|---|
| `hw2_1_315511079.ipynb` | Problem 1: basic / input-gated diagonal SSM for selective key–value memory, comparison, Q4 improvement and Kaggle submission |
| `hw2_2_315511079.ipynb` | Problem 2: Masked Diffusion Language Model (MDLM) trained from scratch on TinyStories |
| `submission_hw2_1.csv` | Final Kaggle submission for Problem 1 (2,000 rows, `id,answer`) |
| `hw2_1_outputs/` | Generated `hw2_1.py` / `hw2_1_q4.py`, `final_model_config.json`, per-run histories, curves, failure cases and checkpoints (`runs/`), Kaggle screenshot |
| `hw2_2_outputs/` | `results.json` (summary, MCE, generations, loss histories), `generations.txt`, loss / MCE figures, tokenizer |

The MDLM checkpoint (`tiny_mdlm_v2.pt`, 165 MB) exceeds GitHub's 100 MB file limit and is not included; it is submitted on the course platform.

## Results

**Problem 1** (validation set, seed 42, A100)

| Model | Params | Val accuracy | Best epoch |
|---|---|---|---|
| Basic SSM (Q1) | 31,712 | 0.1913 | 10 |
| Gated SSM (Q2) | 40,032 | 0.4060 | 8 |
| A: gated + augmentation | 40,032 | 0.4067 | 11 |
| **D: d256 / N256 / 3 layers + query-key head** | **955,680** | **0.9913** | 198 |
| E: d192 / N256 / 4 layers + query-key head | 867,552 | 0.4053 | 13 |

Kaggle (exact-match accuracy, public): **0.99500** with model D.

Q3 cost comparison (basic vs gated): 2.24 vs 2.47 s/epoch, 30,696 vs 27,129 samples/s, 92.5 vs 104.8 MiB peak memory.

Main changes in Q4 (SSM update equations unchanged, token-only inputs, < 1M parameters):

- Task-consistent augmentation of `train.jsonl` only: consistent key / value renaming, record shuffling, re-sampled valid query
- Wider state and residual stream (d = N = 256), 3 gated SSM blocks
- Prediction head combining the ANSWER representation with the query-key embedding (bilinear term)
- 200 epochs, batch 128, AdamW lr 1e-3, warm-up + cosine, EMA of weights, early stopping on validation accuracy

**Problem 2** (Colab Tesla T4, 20 epochs = 84,200 steps, 112.7 min)

| Item | Value |
|---|---|
| Parameters | 13.77M |
| Final validation MDLM loss | 2.332 |
| MCE @ 25 / 50 / 75 % masking | 1.160 / 1.849 / 3.156 |
| Masked-token accuracy @ 25 / 50 / 75 % | 72.2 / 59.2 / 37.8 % |
| Decoding time, 64 tokens, 4 / 8 / 16 steps | 14.8 / 30.3 / 61.2 ms |

More decoding steps give more fluent, less repetitive text at proportionally higher cost.

## Compliance

- Problem 1: only the five model TODOs were filled for Q1–Q3, with the prescribed backbone and matched settings. Q4 keeps the basic/gated SSM family (no attention / RNN / pretrained models), uses only token inputs, trains only on `train.jsonl`, selects on `val.jsonl`, and uses a single validation-selected model (no ensemble).
- Problem 2: the provided model, tokenizer and data split, random initialisation, one Colab T4 within the 4-hour budget. Adjustments (120k stories, batch 64, epoch-based stopping with a 200-min cap, validation-loss probe) are stated at the top of the notebook.

## How to run

**Problem 1:** put `train.jsonl`, `val.jsonl`, `test.jsonl` in `data/` (or upload the competition zip), then run the notebook top to bottom on a GPU (about 1 hour on an A100). It writes `submission.csv` and `hw2_1_artifacts.zip`.

**Problem 2:** run on a Colab T4. TinyStories is downloaded automatically; the last cell writes `hw2_2_artifacts.zip`.

Datasets are provided by the course and are not included in this repo.
