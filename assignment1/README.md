# Assignment 1

Full task spec: [`assignment_description.md`](assignment_description.md)

## Files

| File | Content |
|---|---|
| `hw1_1_315511079.ipynb` | Problem 1: character-level bigram / trigram / RNN / LSTM language models for language identification (EN, FR, ES, PT, DE) |
| `hw1_2_315511079.ipynb` | Problem 2: Transformer encoder (self-attention from scratch) for patent phrase similarity, Kaggle submission |
| `hw1_315511079.pdf` | Written report covering both problems |
| `latex_file/` | LaTeX source and figures of the report (`hw1_315511079.tex`) |
| `submission_part2.csv` | Final Kaggle submission for Problem 2 (7,295 rows, original test ID order) |

## Results

**Problem 1** (test set, corpus-level perplexity)

| Model | Test PPL | Accuracy | Macro-F1 |
|---|---|---|---|
| Bigram | 11.33 | 0.9729 | 0.9727 |
| Trigram | 9.40 | **0.9871** | **0.9871** |
| RNN | **5.71** | 0.9857 | 0.9857 |
| LSTM | 7.98 | 0.9800 | 0.9799 |

Lower perplexity does not always mean higher accuracy; most errors are short Spanish ↔ Portuguese texts.

**Problem 2** (validation set)

| Model | Val Pearson | Val RMSE |
|---|---|---|
| Baseline (4 heads, mean pooling, 10 epochs) | 0.3173 | 0.2487 |
| Main model (4 heads, final setting) | 0.6210 | 0.2177 |
| Best single configuration in tuning (D12) | **0.6400** | **0.2091** |

Final model: an 8-model ensemble of the final setting with three pooling feature sets, averaged.
Kaggle leaderboard (Pearson): **0.68679**.

Heads comparison (10 epochs, baseline setting): 1 head 0.3282, 4 heads 0.3173, 8 heads 0.2968.

Main changes over the baseline (all inside the `TODO` / `TODO (Optional)` blocks):

- Self-attention and encoder layer implemented from scratch (no `torch.nn.MultiheadAttention`), with an input LayerNorm in each encoder layer
- Pooling: anchor/target positions recovered from `input_ids`, and `[g, u, v, |u-v|, u*v]` (optionally plus max-pooled features) fed through a small MLP created inside the pooling block
- AdamW with a 20× learning rate for the token embedding, weight decay 0.01, dropout 0.1
- 2-epoch warm-up + cosine decay, 25 epochs

## Compliance

Following the course announcements, only the code inside the `TODO` / `TODO (Optional)` blocks of the provided `atten.ipynb` is changed. The (c)/(d) experiments and the (e) ensemble are in extra code cells that only call the provided functions; the ensemble in a separate cell is explicitly permitted by the TA.

## How to run

Put the course data next to the notebooks, or set `DATA_DIR` in Problem 2 to the Kaggle input folder:

```
problem1: train.csv, val.csv, test.csv            (columns: Text, Language)
problem2: train.csv, val.csv, test.csv, sample_submission.csv
```

Then run each notebook top to bottom on a GPU. Seed is fixed to 42. The full Problem 2 notebook (main model, (c), (d) and the 8-model ensemble) takes about 2–2.5 hours on a Kaggle GPU.

To build the report, compile `latex_file/hw1_315511079.tex` with pdfLaTeX (all figures are in the same folder).
