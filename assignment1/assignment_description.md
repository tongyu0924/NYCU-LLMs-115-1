# Homework 1 — Assignment Description

Large Language Models, Homework 1 · Due 2026/10/16 23:55 (hard deadline)

## Problem 1: N-Gram, RNN, and LSTM Models for Language Identification (50%)

Build character-level language models for five languages (English, French, Spanish, Portuguese, German; TED Talks dataset) and classify each text as the language whose model gives the **lowest perplexity**.

**Data:** 1,400 texts per language, pre-split into 1,120 train / 140 val / 140 test (`Text`, `Language` columns). Do not resample or repartition. Build one shared character vocabulary from `train.csv` with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.

1. **Bigram & trigram** with Add-α smoothing (α = 1), implemented by hand.
   - (a) Corpus-level perplexity on `val.csv`. (5%)
   - (b) Accuracy, Macro-F1, perplexity and confusion matrix on `test.csv`. (5%)
   - (c) Build time, inference time, peak memory; analyse errors, confusable languages, and short texts. (5%)
2. **RNN** (PyTorch, one model per language). Suggested: embedding 128, hidden 128, 2 layers, Adam lr 0.001, 10 epochs, batch 32, cross-entropy.
   - (a) Training-loss and validation-perplexity curves. (5%)
   - (b) Select checkpoint on val; report test Accuracy, Macro-F1, perplexity. (5%)
   - (c) Confusion matrix, training/inference time, peak memory, example predictions. (5%)
3. **LSTM**: same settings as the RNN, same (a)–(c). (15%)
4. **Comparison** of all four models: hardware, time, Accuracy, Macro-F1, perplexity. Does lower perplexity always mean higher accuracy? Which pairs are confused? Are short texts harder? (5%)

Notes: exclude `<BOS>`/`<PAD>` from targets and include `<EOS>`. Always use corpus-level perplexity (total NLL / total targets); never average per-text perplexities.

## Problem 2: Self-Attention Layer for Semantic Similarity Scoring (50%)

Kaggle competition based on U.S. Patent Phrase-to-Phrase Matching: predict a similarity score in [0, 1] for each (anchor, target, context).

**Data:** train 25,531 / val 3,647 / test 7,295. Scores take values in {0, 0.25, 0.5, 0.75, 1}.

**Model:** input `[CONTEXT] context [SEP] anchor [SEP] target` → token + positional embedding → Transformer encoder → masked mean pooling → `sigmoid(W·h + b)`. Complete the TODOs in the provided notebook (Q/K/V projections, scaled dot-product attention, head concatenation, residuals, layer norm, FFN). `torch.nn.MultiheadAttention` and pretrained models are **not** allowed. Train with MSE; select models by Pearson. Keep the given vocab, max length, embedding size, layers, batch size and seed unless a setting is listed as tunable.

- (a) 4 heads: training/validation loss curves and validation Pearson curve. (10%)
- (b) Validation Pearson and RMSE. (5%)
- (c) Compare 1, 4 and 8 heads under the same budget. (5%)
- (d) Improve validation Pearson by tuning permitted settings; document every change. (5%)
- (e) Predict `test.csv`, submit `submission.csv` to Kaggle (keep IDs and order), include a leaderboard screenshot. (25%)

## Rules

- Submit two notebooks named `hw1_<ProblemNumber>_<StudentID>.ipynb` with code, results, figures and discussion.
- Python only. N-gram part: basic libraries only (NumPy, pandas, collections, math, random). RNN/LSTM/attention: PyTorch allowed.
- GPU required for neural experiments. At least one valid Kaggle submission. No external labeled data. No plagiarism.
