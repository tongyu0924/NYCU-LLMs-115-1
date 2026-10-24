"""Provided training/evaluation utilities. Model inputs are tokens only."""
import argparse
import csv
import json
import math
import platform
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

def fixed_vocab():
    tokens = ['PAD','BOS','SEP','STORE','NOISE','QUERY','ANSWER'] + [f'K{i}' for i in range(32)] + [f'V{i}' for i in range(32)]
    return {token: i for i, token in enumerate(tokens)}

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class MemoryDataset(Dataset):
    def __init__(self, path, vocab, limit=0):
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f'Empty dataset: {path}')
        self.ids = [[vocab[t] for t in row['tokens']] for row in self.rows]
        self.labels = [int(row['answer'][1:]) if 'answer' in row else -100 for row in self.rows]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.ids[i], self.labels[i], self.rows[i]

def collate(samples):
    lengths = torch.tensor([len(x[0]) for x in samples], dtype=torch.long)
    # PAD is ID 0 in the fixed vocabulary.
    tokens = torch.zeros(len(samples), int(lengths.max()), dtype=torch.long)
    for i,(ids,_,_) in enumerate(samples):
        tokens[i,:len(ids)] = torch.tensor(ids)
    return tokens, lengths, torch.tensor([x[1] for x in samples]), [x[2] for x in samples]

def loader(path, vocab, batch_size, shuffle=False, limit=0):
    return DataLoader(MemoryDataset(path,vocab,limit),batch_size=batch_size,shuffle=shuffle,collate_fn=collate,num_workers=0)

class SSMBlock(nn.Module):
    def __init__(self, ssm, d, dropout):
        super().__init__()
        self.ssm=ssm
        self.norm=nn.LayerNorm(d)
        self.act=nn.GELU()
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,return_gates=False):
        y,g=self.ssm(self.norm(x),return_gates=return_gates)
        return x+self.dropout(self.act(y)),g

class MemoryModel(nn.Module):
    def __init__(self, config, basic_class, gated_class):
        super().__init__()
        d,n=config['d'],config['n']
        self.embedding=nn.Embedding(71,d,padding_idx=0)
        layers=[]
        for _ in range(config['layers']):
            ssm=basic_class(d,n) if config['model']=='basic' else gated_class(d,n)
            layers.append(SSMBlock(ssm,d,config['dropout']))
        self.blocks=nn.ModuleList(layers)
        self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,32)
    def forward(self,tokens,lengths,return_gates=False):
        x=self.embedding(tokens)
        gates=[]
        for block in self.blocks:
            x,g=block(x,return_gates)
            gates.append(g)
        # Only valid ANSWER representations reach the prediction head.
        last=self.norm(x)[torch.arange(tokens.size(0),device=tokens.device),lengths-1]
        return self.head(last),gates

def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def on_device(batch,device):
    tokens,lengths,labels,rows=batch
    return tokens.to(device),lengths.to(device),labels.to(device),rows

@torch.no_grad()
def evaluate(model,batches,device,keep=False):
    model.eval()
    total_loss=correct=total=0
    predictions=[]
    for batch in batches:
        tokens,lengths,labels,rows=on_device(batch,device)
        logits,_=model(tokens,lengths)
        pred=logits.argmax(-1)
        valid=labels.ne(-100)
        if valid.any():
            total_loss+=nn.functional.cross_entropy(logits[valid],labels[valid],reduction='sum').item()
            correct+=(pred[valid]==labels[valid]).sum().item()
            total+=valid.sum().item()
        if keep:
            for row,p in zip(rows,pred.cpu().tolist()):
                predictions.append(dict(id=row['id'],prediction=f'V{p}',answer=row.get('answer'),group=row.get('group','all'),tokens=row['tokens']))
    return {'loss':total_loss/total if total else None,'accuracy':correct/total if total else None,'count':total},predictions

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')

def plot_history(history,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Training')
    ax.plot([r['epoch'] for r in history],[r['val_loss'] for r in history],label='Validation')
    ax.set(xlabel='Epoch',ylabel='Cross-entropy loss'); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

def train(args,basic,gated):
    seed_all(args.seed)
    device=resolve_device(args.device)
    config={k:getattr(args,k) for k in ['model','d','n','layers','dropout']}
    model=MemoryModel(config,basic,gated).to(device)
    effective=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if effective>=1000000: raise ValueError('Effective trainable parameter count must be below 1,000,000.')
    vocab=fixed_vocab()
    tr=loader(args.data/'train.jsonl',vocab,args.batch_size,True,args.limit_train)
    va=loader(args.data/'val.jsonl',vocab,args.batch_size,False,args.limit_val)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    best=(-1.,float('-inf')); history=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    peak_training=0
    environment={'torch':torch.__version__,'python':platform.python_version(),'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),'precision':'float32','threads':torch.get_num_threads()}
    for epoch in range(1,args.epochs+1):
        model.train(); total=loss_sum=0
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        sync(device); start=time.perf_counter()
        for batch in tr:
            tokens,lengths,labels,_=on_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            logits,_=model(tokens,lengths)
            loss=nn.functional.cross_entropy(logits,labels)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total+=len(labels); loss_sum+=loss.item()*len(labels)
        sync(device); elapsed=time.perf_counter()-start
        if device.type=='cuda': peak_training=max(peak_training,torch.cuda.max_memory_allocated(device))
        sync(device); start=time.perf_counter(); metrics,_=evaluate(model,va,device); sync(device)
        record=dict(epoch=epoch,train_loss=loss_sum/total,val_loss=metrics['loss'],val_accuracy=metrics['accuracy'],train_seconds=elapsed,val_seconds=time.perf_counter()-start)
        history.append(record); print(json.dumps(record),flush=True)
        rank=(metrics['accuracy'],-metrics['loss'])
        if rank>best:
            best=rank
            # No optimizer or executable Python objects in the checkpoint.
            torch.save({'state_dict':model.state_dict(),'config':config,'vocab':vocab,'epoch':epoch,'validation':metrics,'seed':args.seed},out/'best.pt')
        save_json(out/'history.json',history)
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    inference=benchmark(model,va,device)
    validation, predictions=evaluate(model,va,device,True)
    save_json(out/'failure_examples.json',[row for row in predictions if row['prediction']!=row['answer']][:3])
    plot_history(history,out/'learning_curves.png')
    save_json(out/'run_config.json',{**vars_serializable(args),'environment':environment})
    save_json(out/'training_summary.json',{'effective_trainable_parameters':effective,'total_parameters':sum(p.numel() for p in model.parameters()),'mean_train_seconds_per_epoch':sum(h['train_seconds'] for h in history)/len(history),'inference':inference,'best_validation_accuracy':best[0],'peak_training_allocated_bytes':peak_training if device.type=='cuda' else None,'note':'CPU memory is not GPU allocated memory; no GPU estimate is supplied. Validation excluded from epoch training time.'})

def vars_serializable(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}

def resolve_device(name):
    if name=='auto': name='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    return device

def restore(args,basic,gated):
    device=resolve_device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
    model=MemoryModel(checkpoint['config'],basic,gated).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model,checkpoint['vocab'],device

def predict(args,basic,gated):
    model,vocab,device=restore(args,basic,gated)
    batches=loader(args.input,vocab,args.batch_size)
    _,rows=evaluate(model,batches,device,True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','answer']); w.writerows((r['id'],r['prediction']) for r in rows)
    print(f'Wrote {len(rows)} predictions to {args.output}')

@torch.no_grad()
def benchmark(model,batches,device):
    # Time forward only on a fixed padded batch, with copies excluded.
    tokens,lengths,_,_=on_device(next(iter(batches)),device)
    for _ in range(3): model(tokens,lengths)
    sync(device); start=time.perf_counter()
    for _ in range(10): model(tokens,lengths)
    sync(device); seconds=(time.perf_counter()-start)/10
    return dict(forward_batch_seconds=seconds,forward_samples_per_second=tokens.size(0)/seconds,batch_size=tokens.size(0),padded_length=tokens.size(1),timing_scope='10 repeated forwards after 3 warmups; data transfer excluded')


def diag_scan(lam, u):
    """Causal linear recurrence h_t = lam_t * h_{t-1} + u_t with h_0 = 0.

    lam, u: [batch, length, N].  Returns h: [batch, length, N].
    Hillis-Steele parallel prefix scan over the time axis (log2(L) steps).
    It only multiplies decays in (0, 1], so it is numerically stable, and each
    output position t depends on positions <= t only (causal).  A fresh scan is
    run on every forward call, so no state is carried between samples/batches.
    """
    A, h = lam, u
    length = u.size(1)
    offset = 1
    while offset < length:
        # combine element t with element t-offset: (A_t, h_t) o (A_s, h_s)
        h = torch.cat([h[:, :offset], A[:, offset:] * h[:, :-offset] + h[:, offset:]], dim=1)
        A = torch.cat([A[:, :offset], A[:, offset:] * A[:, :-offset]], dim=1)
        offset *= 2
    return h

def reference_loop(lam, u):
    """Slow step-by-step reference used only by the sanity-check cell."""
    h = torch.zeros_like(u[:, 0])
    out = []
    for t in range(u.size(1)):
        h = lam[:, t] * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1)

class DiagonalSSM(nn.Module):
    def __init__(self, d, n, a_min=0.5, a_max=0.99):
        super().__init__()
        self.d, self.n = d, n
        # TODO: basic_init
        # a = sigmoid(theta_a) is initialised evenly over [a_min, a_max] (default 0.5-0.99,
        # the suggested spread). B, C, D: Gaussian with std 1/sqrt(fan_in); no biases.
        a = torch.linspace(a_min, a_max, n)
        self.theta_a = nn.Parameter(torch.log(a) - torch.log1p(-a))   # logit(a)
        self.B = nn.Parameter(torch.randn(n, d) / math.sqrt(d))        # [N, d]
        self.C = nn.Parameter(torch.randn(d, n) / math.sqrt(n))        # [d, N]
        self.D = nn.Parameter(torch.randn(d, d) / math.sqrt(d))        # [d, d]

    def forward(self, x, return_gates=False):
        """x: [batch,length,d]; return (y [batch,length,d], gates or None)."""
        # TODO: basic_forward
        a = torch.sigmoid(self.theta_a)                 # [N], in (0, 1)
        u = x @ self.B.t()                              # B x_t        -> [batch, length, N]
        h = diag_scan(a.expand_as(u), u)                # h_t = a*h_{t-1} + B x_t, h_0 = 0
        y = h @ self.C.t() + x @ self.D.t()             # C h_t + D x_t -> [batch, length, d]
        return y, None

class GatedDiagonalSSM(DiagonalSSM):
    def __init__(self, d, n, a_min=0.5, a_max=0.99):
        super().__init__(d, n, a_min=a_min, a_max=a_max)
        # TODO: gate_init
        # Same rule as B (std 1/sqrt(d)); gate bias zero -> g = 0.5 at initialisation.
        self.W_g = nn.Parameter(torch.randn(n, d) / math.sqrt(d))      # [N, d]
        self.b_g = nn.Parameter(torch.zeros(n))                        # [N]

    def forward(self, x, return_gates=False):
        # TODO: gate_compute
        g = torch.sigmoid(x @ self.W_g.t() + self.b_g)  # g_t = sigma(W_g x_t + b_g) -> [batch, length, N]
        # TODO: gate_forward
        a = torch.sigmoid(self.theta_a)                 # [N]
        bx = x @ self.B.t()                             # B x_t
        # h_t = (1-g_t)*h_{t-1} + g_t*(a*h_{t-1} + B x_t)
        #     = (1 - g_t*(1-a)) * h_{t-1} + g_t * B x_t
        lam = 1 - g * (1 - a)
        h = diag_scan(lam, g * bx)
        y = h @ self.C.t() + x @ self.D.t()
        return y, (g if return_gates else None)


def run_cli(basic,gated):
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads',type=int,default=2)
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--model',choices=['basic','gated'],required=True)
    t.add_argument('--data',type=Path,default=Path('data'));t.add_argument('--out',type=Path,required=True)
    for name,value in [('d',64),('n',64),('layers',2),('epochs',10),('batch-size',64),('seed',42),('limit-train',0),('limit-val',0)]:t.add_argument('--'+name,type=int,default=value)
    t.add_argument('--dropout',type=float,default=.1);t.add_argument('--lr',type=float,default=.001);t.add_argument('--weight-decay',type=float,default=.01);t.add_argument('--device',default='auto')
    q=sub.add_parser('predict');q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--input',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--batch-size',type=int,default=64);q.add_argument('--device',default='auto')
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if args.command=='train':train(args,basic,gated)
    else:predict(args,basic,gated)

if __name__=='__main__':run_cli(DiagonalSSM,GatedDiagonalSSM)
