# Assignment 1

Full task spec: [`assignment_description.md`](assignment_description.md)

## Files

| File | Content |
|---|---|
| `hw1_1_315511079.ipynb` | Problem 1: character-level bigram / trigram / RNN / LSTM language models for language identification (EN, FR, ES, PT, DE) |
| `hw1_2_315511079.ipynb` | Problem 2: Transformer encoder (self-attention from scratch) for patent phrase similarity, Kaggle submission |

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
| Baseline (4 heads, mean pooling, 10 epochs) | 0.3079 | 0.2496 |
| Best single model (cosine LR, scaled embedding init, anchor/target interaction pooling) | 0.6206 | 0.2342 |
| Final: 5-model ensemble | **0.6409** | **0.2233** |

Kaggle leaderboard (Pearson): **0.66983**

## How to run

Put the course data next to the notebooks (or in a Kaggle input folder; Problem 2 finds `train.csv` automatically):

```
problem1: train.csv, val.csv, test.csv            (columns: Text, Language)
problem2: train.csv, val.csv, test.csv, sample_submission.csv
```

Then run each notebook top to bottom on a GPU. Seed is fixed to 42.
