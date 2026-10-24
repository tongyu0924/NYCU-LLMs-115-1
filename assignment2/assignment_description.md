# Assignment 2: Large Language Models

Due: 2026/11/13 23:55 (hard deadline)

## Problem 1: Selective memory with state-space models (50%)

Train SSMs from scratch to retrieve values from a token sequence:

- `STORE K_i V_j`: a valid key–value pair to remember
- `NOISE K_i V_j`: a distracting pair to ignore (even if the key matches a stored one)
- `QUERY K_i ANSWER`: predict the stored value V0–V31

Data: `train.jsonl` (12,000), `val.jsonl` (1,500), `test.jsonl` (2,000, Kaggle). Each prompt has 6–30 records, 2–4 of them STORE with distinct keys. Fixed 71-token vocabulary.

1. Basic diagonal SSM: `h_t = a ⊙ h_{t-1} + B x_t`, `y_t = C h_t + D x_t`, `a = σ(θ_a)` (10%)
2. Input-gated SSM: `g_t = σ(W_g x_t + b_g)`, `h_t = (1 - g_t) ⊙ h_{t-1} + g_t ⊙ (a ⊙ h_{t-1} + B x_t)` (10%)
3. Compare parameters, accuracy, training time, throughput and GPU memory (5%)
4. Kaggle: one improvement (state size, depth, LR, schedule, head; task-consistent augmentation of train only), < 10⁶ parameters, single validation-selected model (25%, 20% of it by leaderboard)

Shared backbone for Q1–3: embedding, two pre-LN SSM blocks (`x + Dropout(GELU(SSM(LN(x))))`), final LayerNorm, 32-way head, d = N = 64, dropout 0.1, AdamW lr 1e-3, wd 0.01, batch 64, 10 epochs, clip 1.0, seed 42.

Not allowed: self-attention, RNN/LSTM/GRU, pretrained models, external data, rule-based retrieval, training on val/test, tuning to the public leaderboard.

## Problem 2: Masked diffusion language models (50%)

Train the provided ~14M-parameter bidirectional Transformer (6 layers, hidden 384, 6 heads, 8K BPE, length 128) from random initialisation on 120,000 TinyStories, with one Colab T4 and at most 4 GPU hours.

1. Implement `forward_masking` (7%), `compute_mdlm_loss` with 1/(tL) weighting (10%), and confidence-based `iterative_decode` (8%)
2. Train; report loss curve, validation loss, steps, GPU time, parameter count; save the checkpoint (10%)
3. Masked cross-entropy at 25/50/75% masking; generate 64 tokens from "Once upon a time" with 4/8/16 steps and compare (10%)
4. Short answers: MDLM vs autoregressive generation, why multiple masking ratios, why low loss can still give repetitive text (5%)

## Submission

`hw2_1_<StudentID>.ipynb` and `hw2_2_<StudentID>.ipynb` with code, outputs, figures and written discussion; Kaggle CSV, submission ID, checkpoint and configuration for Problem 1; MDLM checkpoint for Problem 2.
