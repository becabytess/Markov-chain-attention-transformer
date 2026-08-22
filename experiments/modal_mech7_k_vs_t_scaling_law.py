"""
Mechanistic Study 7: The (K, T) Compensation Frontier (Parallelized Sweep)
Tests whether increasing thought depth T can compensate for ultra-sparse neighborhood K:
- Sweeps K in [1, 2, 4, 8, 16] and T in [1, 2, 4, 8]
- Runs concurrently on Modal GPU workers for maximal speed and stability.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests", "tiktoken")
)

app = modal.App("gravimem-mech7-k-vs-t-parallel", image=image)


@app.function(gpu="T4", timeout=600)
def train_and_eval_config(item):
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import tiktoken

    K_val = item["K"]
    T_val = item["T"]
    desc = item["desc"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_text = urllib.request.urlopen(url).read().decode('utf-8')

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(raw_text)
    data = torch.tensor(tokens, dtype=torch.long)
    vocab_size = enc.n_vocab

    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    block_size = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    class DynamicKSurfer(nn.Module):
        def __init__(self, d_model, n_heads, K):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.K = K

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.gru_cell = nn.GRUCell(d_model, d_model)

        def forward(self, x, T=4):
            B, L, D = x.shape
            device = x.device

            q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            pos = torch.arange(L, device=device)
            cand_indices = []
            cand_mask = []
            for i in range(L):
                if i == 0:
                    cand_indices.append(torch.zeros(self.K, dtype=torch.long, device=device))
                    cand_mask.append(torch.zeros(self.K, dtype=torch.bool, device=device))
                    continue
                valid_prev = torch.arange(i, device=device)
                d = i - valid_prev
                log_d = torch.log2(d.float() + 1.0)
                max_log = log_d.max()
                
                if self.K == 1:
                    chosen = valid_prev[-1:] # immediate predecessor
                else:
                    stride_buckets = torch.linspace(0, max_log, steps=self.K, device=device)
                    chosen = valid_prev[torch.abs(log_d.unsqueeze(1) - stride_buckets.unsqueeze(0)).argmin(dim=0)]
                
                cand_indices.append(chosen)
                cand_mask.append(torch.ones(self.K, dtype=torch.bool, device=device))

            cand_indices = torch.stack(cand_indices, dim=0)
            cand_mask = torch.stack(cand_mask, dim=0)

            k_cand = k[:, :, cand_indices, :]
            v_cand = v[:, :, cand_indices, :]

            if self.K == 1:
                ctx_h = v_cand.squeeze(3)
            else:
                q_exp = q.unsqueeze(3)
                scores = (q_exp * k_cand).sum(dim=-1) / math.sqrt(self.head_dim)
                scores = scores.masked_fill(~cand_mask.unsqueeze(0).unsqueeze(0), -1e9)
                attn_weights = F.softmax(scores, dim=-1)
                ctx_h = (attn_weights.unsqueeze(-1) * v_cand).sum(dim=3)

            ctx = ctx_h.transpose(1, 2).contiguous().view(B, L, D)
            ctx_flat = ctx.view(B * L, D)

            s_t = torch.zeros(B * L, D, device=device)
            for step in range(1, T + 1):
                s_t = self.gru_cell(ctx_flat, s_t)

            out = self.out_proj(s_t.view(B, L, D))
            return out

    class DynamicKGravimemLM(nn.Module):
        def __init__(self, vocab_size, d_model, n_heads, d_mlp, block_size, K):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)
            self.ln1 = nn.LayerNorm(d_model)
            self.surfer = DynamicKSurfer(d_model, n_heads, K=K)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T=4):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            s_out = self.surfer(self.ln1(x), T=T)
            x = x + s_out
            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x))
            return logits

    model = DynamicKGravimemLM(vocab_size, d_model, n_heads, d_mlp, block_size, K=K_val).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    t0 = time.time()
    for step in range(1, 501):
        model.train()
        xb, yb = get_batch('train')
        logits = model(xb, T=T_val)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.time() - t0

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(25):
            xb, yb = get_batch('val')
            logits = model(xb, T=T_val)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            val_losses.append(loss.item())

    mean_loss = np.mean(val_losses)
    ppl = math.exp(min(mean_loss, 20.0))
    
    return {
        "K": K_val,
        "T": T_val,
        "desc": desc,
        "val_loss": mean_loss,
        "ppl": ppl,
        "time": elapsed
    }


@app.local_entrypoint()
def main():
    grid = [
        # Ultra-Sparse K=1 (Single Pointer)
        {"K": 1, "T": 1, "desc": "K=1, T=1 (Single Pointer, 1 Hop)"},
        {"K": 1, "T": 4, "desc": "K=1, T=4 (Single Pointer, 4 Hops)"},

        # Binary Branch K=2 (Binary Tree Graph)
        {"K": 2, "T": 1, "desc": "K=2, T=1 (Binary Graph, 1 Hop)"},
        {"K": 2, "T": 4, "desc": "K=2, T=4 (Binary Graph, 4 Hops)"},

        # Quad Branch K=4
        {"K": 4, "T": 1, "desc": "K=4, T=1 (4-Branch, 1 Hop)"},
        {"K": 4, "T": 4, "desc": "K=4, T=4 (4-Branch, 4 Hops)"},

        # Octa Branch K=8
        {"K": 8, "T": 1, "desc": "K=8, T=1 (8-Branch, 1 Hop)"},
        {"K": 8, "T": 2, "desc": "K=8, T=2 (8-Branch, 2 Hops)"},

        # Golden Reference K=16
        {"K": 16, "T": 1, "desc": "K=16, T=1 (16-Branch, 1 Hop)"},
        {"K": 16, "T": 2, "desc": "K=16, T=2 (16-Branch, 2 Hops)"},
    ]

    print("=" * 80)
    print("  LAUNCHING CONCURRENT (K, T) SCALING SWEEP ON MODAL GPU WORKERS...")
    print("=" * 80)

    # Parallel map across Modal GPUs
    results = list(train_and_eval_config.map(grid))

    print("\n" + "=" * 80)
    print("  THE (K, T) SCALING LAW & COMPENSATION FRONTIER")
    print("=" * 80)
    print(f"{'Configuration':<42} | {'Width K':<8} | {'Depth T':<8} | {'Val Loss':<10} | {'Perplexity':<12}")
    print("-" * 80)

    for r in results:
        print(f"  {r['desc']:<40} | K={r['K']:<5} | T={r['T']:<5} | {r['val_loss']:>8.4f} | {r['ppl']:>10.2f}")

    print("=" * 80)
    print("\n---> SUMMARY OF PATTERNS & RELATIONSHIP:")
    for k_group in [1, 2, 4, 8, 16]:
        sub = [r for r in results if r["K"] == k_group]
        if sub:
            t1 = sub[0]
            t2 = sub[-1]
            print(f"  * K={k_group:<2}: T={t1['T']} (PPL={t1['ppl']:.2f}) -> T={t2['T']} (PPL={t2['ppl']:.2f}) | Delta = {t2['ppl'] - t1['ppl']:+.2f}")

    print("=" * 80)
