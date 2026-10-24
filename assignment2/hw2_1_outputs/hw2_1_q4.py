"""Q4 improvement: wider/deeper gated SSM + query-key head + task-consistent augmentation.

Everything here re-uses the provided utilities and the *same* GatedDiagonalSSM class
from hw2_1.py (the SSM update equations are unchanged).  Inputs are token IDs only.
Training uses data/train.jsonl only; checkpoints are selected on data/val.jsonl only.
"""
import argparse
import copy
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hw2_1 import (GatedDiagonalSSM, SSMBlock, MemoryDataset, collate, loader, fixed_vocab,
                   seed_all, evaluate, benchmark, sync, on_device, resolve_device,
                   save_json, plot_history)

# ----------------------------------------------------------------------------------
# Task-consistent augmentation of TRAINING samples only
# ----------------------------------------------------------------------------------
def augment_tokens(tokens, rng):
    """Return (new_tokens, new_answer_index) for one training prompt.

    1. Random permutation of key names K0..K31 (applied consistently to every record and query).
    2. Random permutation of value names V0..V31 (applied consistently; the label is mapped too).
    3. Random shuffle of record order (the task states record order is random).
    4. Re-sample the query key uniformly from the valid STORE keys and recompute the label
       ("valid query change").  STORE keys are distinct, so the label is unambiguous.
    The number of records, the STORE/NOISE split and every key/value co-occurrence pattern are
    preserved, so the sample stays inside the task distribution.
    """
    n_rec = (len(tokens) - 4) // 4
    records = [tokens[1 + 4 * i: 5 + 4 * i] for i in range(n_rec)]
    kperm = list(range(32)); rng.shuffle(kperm)
    vperm = list(range(32)); rng.shuffle(vperm)
    rng.shuffle(records)
    new_records = [[typ, f'K{kperm[int(k[1:])]}', f'V{vperm[int(v[1:])]}', sep]
                   for typ, k, v, sep in records]
    stores = [r for r in new_records if r[0] == 'STORE']
    q = rng.choice(stores)
    new_tokens = ['BOS'] + [t for r in new_records for t in r] + ['QUERY', q[1], 'ANSWER']
    return new_tokens, int(q[2][1:])

def check_sample(tokens, answer):
    """Assert the prompt format and that `answer` is the STORE value of the query key."""
    assert tokens[0] == 'BOS' and tokens[-3] == 'QUERY' and tokens[-1] == 'ANSWER'
    n_rec = (len(tokens) - 4) // 4
    assert len(tokens) == 4 * n_rec + 4
    store = {}
    for i in range(n_rec):
        typ, k, v, sep = tokens[1 + 4 * i: 5 + 4 * i]
        assert typ in ('STORE', 'NOISE') and sep == 'SEP'
        if typ == 'STORE':
            assert k not in store
            store[k] = v
    assert 2 <= len(store) <= 4
    assert store[tokens[-2]] == f'V{answer}'

class AugmentedTrainDataset(Dataset):
    def __init__(self, path, vocab, seed, augment=True):
        self.base = MemoryDataset(path, vocab)
        self.vocab, self.augment = vocab, augment
        self.rng = random.Random(seed)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        row = self.base.rows[i]
        if not self.augment:
            return self.base[i]
        toks, ans = augment_tokens(row['tokens'], self.rng)
        return [self.vocab[t] for t in toks], ans, {'id': row['id'] + '_aug', 'tokens': toks, 'answer': f'V{ans}'}

# ----------------------------------------------------------------------------------
# Improved model: same SSMBlock / GatedDiagonalSSM, configurable width/depth/state + head
# ----------------------------------------------------------------------------------
class ImprovedMemoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, n = config['d'], config['n']
        self.config = config
        self.embedding = nn.Embedding(71, d, padding_idx=0)
        self.blocks = nn.ModuleList([
            SSMBlock(GatedDiagonalSSM(d, n, a_min=config['a_min'], a_max=config['a_max']), d, config['dropout'])
            for _ in range(config['layers'])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 32)
        if config['head'] == 'query':
            h = config['head_hidden']
            # bilinear interaction between the ANSWER representation and the query-key embedding
            self.r_proj = nn.Linear(d, h, bias=False)
            self.q_proj = nn.Linear(d, h, bias=False)
            self.q_out = nn.Linear(h, 32, bias=False)
    def forward(self, tokens, lengths, return_gates=False):
        x = self.embedding(tokens)
        gates = []
        for block in self.blocks:
            x, g = block(x, return_gates)
            gates.append(g)
        rows = torch.arange(tokens.size(0), device=tokens.device)
        last = self.norm(x)[rows, lengths - 1]               # final valid ANSWER position
        logits = self.head(last)
        if self.config['head'] == 'query':
            q = self.embedding(tokens[rows, lengths - 2])    # observed query key token (QUERY K? ANSWER)
            logits = logits + self.q_out(self.r_proj(last) * self.q_proj(q))
        return logits, gates

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------------------------------------------------------------
# Training with warm-up + cosine LR schedule; checkpoint selection on validation only
# ----------------------------------------------------------------------------------
def train_improved(cfg):
    cfg = dict(cfg)
    seed_all(cfg['seed'])
    device = resolve_device(cfg.get('device', 'auto'))
    data, out = Path(cfg['data']), Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=True)
    model_cfg = {k: cfg[k] for k in ['d', 'n', 'layers', 'dropout', 'a_min', 'a_max', 'head', 'head_hidden']}
    model = ImprovedMemoryModel(model_cfg).to(device)
    n_params = count_params(model)
    if n_params >= 1_000_000:
        raise ValueError(f'{n_params} trainable parameters; must be below 1,000,000')
    print(f"[{cfg['name']}] trainable parameters: {n_params:,}", flush=True)
    vocab = fixed_vocab()
    train_set = AugmentedTrainDataset(data / 'train.jsonl', vocab, cfg['seed'], cfg['augment'])
    g = torch.Generator(); g.manual_seed(cfg['seed'])
    tr = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate,
                    num_workers=0, generator=g, drop_last=True)
    va = loader(data / 'val.jsonl', vocab, 256, False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = cfg['epochs'] * len(tr)
    warm = max(1, int(cfg['warmup_frac'] * total_steps))
    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return cfg['min_lr_frac'] + (1 - cfg['min_lr_frac']) * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # EMA (exponential moving average) of the weights is evaluated/saved; it smooths the noisy
    # augmented-data updates.  It is part of the training schedule, not a different model.
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    ema_decay = cfg.get('ema_decay', 0.0)
    history, best, peak, start_epoch, bad_epochs = [], (-1.0, float('-inf')), 0, 1, 0
    env = {'torch': torch.__version__, 'python': platform.python_version(), 'device': str(device),
           'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
           'precision': 'float32'}
    last_path = out / 'last.pt'
    if last_path.exists():   # resume after a Colab disconnect (our own file -> weights_only=False is safe)
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st['model']); ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        history, best, peak, bad_epochs = st['history'], tuple(st['best']), st['peak'], st['bad_epochs']
        start_epoch = st['epoch'] + 1
        train_set.rng.setstate(st['data_rng']); g.set_state(st['loader_gen'])
        torch.set_rng_state(st['torch_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state(st['cuda_rng'])
        print(f"[{cfg['name']}] resumed from epoch {st['epoch']}", flush=True)
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        if bad_epochs >= cfg.get('patience', 10**9) or best[0] >= 1.0:
            print(f"[{cfg['name']}] early stop before epoch {epoch} (best val acc {best[0]:.4f})", flush=True)
            break
        model.train(); seen = loss_sum = 0
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        sync(device); t0 = time.perf_counter()
        for batch in tr:
            tokens, lengths, labels, _ = on_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(tokens, lengths)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                d_ = ema_decay if epoch > 1 else 0.0          # plain copy during the first epoch
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1.0 - d_)
            seen += len(labels); loss_sum += loss.item() * len(labels)
        sync(device); train_sec = time.perf_counter() - t0
        if device.type == 'cuda': peak = max(peak, torch.cuda.max_memory_allocated(device))
        metrics, _ = evaluate(ema, va, device)
        rec = dict(epoch=epoch, train_loss=loss_sum / seen, val_loss=metrics['loss'],
                   val_accuracy=metrics['accuracy'], train_seconds=train_sec, lr=sched.get_last_lr()[0])
        history.append(rec)
        print(json.dumps(rec), flush=True)
        rank = (metrics['accuracy'], -metrics['loss'])
        if rank > best:
            best, bad_epochs = rank, 0
            torch.save({'state_dict': ema.state_dict(), 'config': model_cfg, 'vocab': vocab,
                        'epoch': epoch, 'validation': metrics, 'seed': cfg['seed'],
                        'train_config': {k: str(v) for k, v in cfg.items()}}, out / 'best.pt')
        else:
            bad_epochs += 1
        save_json(out / 'history.json', history)
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'history': history, 'best': list(best), 'peak': peak,
                    'bad_epochs': bad_epochs, 'epoch': epoch, 'data_rng': train_set.rng.getstate(),
                    'loader_gen': g.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state() if device.type == 'cuda' else None},
                   str(last_path) + '.tmp')
        Path(str(last_path) + '.tmp').replace(last_path)     # atomic: a disconnect never leaves a broken file
    ckpt = torch.load(out / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    inference = benchmark(model, loader(data / 'val.jsonl', vocab, 64, False), device)
    validation, preds = evaluate(model, va, device, True)
    last_path.unlink(missing_ok=True)   # run finished; training_summary.json marks completion
    save_json(out / 'failure_examples.json', [r for r in preds if r['prediction'] != r['answer']][:5])
    plot_history(history, out / 'learning_curves.png')
    summary = {'name': cfg['name'], 'effective_trainable_parameters': n_params,
               'best_epoch': ckpt['epoch'], 'best_validation_accuracy': validation['accuracy'],
               'best_validation_loss': validation['loss'],
               'mean_train_seconds_per_epoch': sum(h['train_seconds'] for h in history) / len(history),
               'inference': inference, 'peak_training_allocated_MiB': peak / 2**20 if device.type == 'cuda' else None,
               'environment': env, 'config': {k: str(v) for k, v in cfg.items()}}
    save_json(out / 'training_summary.json', summary)
    return summary

def restore_improved(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ImprovedMemoryModel(ckpt['config']).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    return model, ckpt

def write_submission(checkpoint, test_path, output, device='auto'):
    device = resolve_device(device)
    model, ckpt = restore_improved(checkpoint, device)
    vocab = fixed_vocab()
    _, rows = evaluate(model, loader(test_path, vocab, 256, False), device, True)
    with open(output, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'answer'])
        w.writerows((r['id'], r['prediction']) for r in rows)
    # format checks: every test id exactly once, answers in V0..V31
    test_ids = [json.loads(l)['id'] for l in Path(test_path).read_text().splitlines() if l.strip()]
    with open(output) as f:
        got = list(csv.DictReader(f))
    assert [r['id'] for r in got] == test_ids and len(set(test_ids)) == len(test_ids)
    assert all(r['answer'] in {f'V{i}' for i in range(32)} for r in got)
    print(f'Wrote {len(got)} predictions to {output} (checkpoint epoch {ckpt["epoch"]}, '
          f'val acc {ckpt["validation"]["accuracy"]:.4f})')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help='JSON string with the training config')
    args = p.parse_args()
    print(json.dumps(train_improved(json.loads(args.config)), indent=2))
