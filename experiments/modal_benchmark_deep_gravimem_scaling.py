"""
Benchmarking Deep Multi-Layer Gravimem vs Deep Standard Transformer Scaling on Modal GPU:
- Standard 2-Layer Transformer (415k params)
- Standard 4-Layer Transformer (810k params)
- Gravimem 2-Layer (T=3 per layer, 416k params) -> effective depth = 6 hops!
- Gravimem 4-Layer (T=3 per layer, 812k params) -> effective depth = 12 hops!
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-deep-scaling", image=image)


@app.function(gpu="T4", timeout=3600)
def run_deep_scaling_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: DEEP MULTI-LAYER STACKING VS STANDARD TRANSFORMER SCALING")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Multi-Layer Architectures
    # -------------------------------------------------------------
    
    class GravimemLayer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.T = T
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))

        def forward(self, x, causal_mask=None):
            B, L, _ = x.shape
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if causal_mask is not None:
                scores = scores + causal_mask[:L, :L]
            P = F.softmax(scores, dim=-1)

            I = torch.eye(L, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            res_scale = 1.0 / math.sqrt(self.T)

            # Settle Markov Mass
            for _ in range(self.T):
                M = (1.0 - alpha) * (P @ M) + alpha * I

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return x

    class DeepGravimem(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, n_layers=2, T=3):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.layers = nn.ModuleList([
                GravimemLayer(d_model=d_model, n_heads=n_heads, d_mlp=d_mlp, T=T)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for layer in self.layers:
                x = layer(x, causal_mask=causal_mask)
            return self.head(self.ln_f(x))

    class StandardTransformerLayer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

        def forward(self, x, causal_mask=None):
            B, L, _ = x.shape
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if causal_mask is not None:
                scores = scores + causal_mask[:L, :L]
            att = F.softmax(scores, dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return x

    class DeepStandardTransformer(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, n_layers=2):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.layers = nn.ModuleList([
                StandardTransformerLayer(d_model=d_model, n_heads=n_heads, d_mlp=d_mlp)
                for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for layer in self.layers:
                x = layer(x, causal_mask=causal_mask)
            return self.head(self.ln_f(x))

    # =========================================================================
    # PART 1: AUTOREGRESSIVE LANGUAGE MODELING (TinyShakespeare, 4,000 Steps)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 1: AUTOREGRESSIVE LM SCALING (4,000 STEPS)")
    print("=" * 70)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    block_size = 128
    batch_size = 64

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)

    models = {
        "1. Standard 2-Layer (420k)": DeepStandardTransformer(vocab_size=vocab_size, max_seq_len=block_size, n_layers=2).to(device),
        "2. Gravimem 2-Layer (T=3, 420k)": DeepGravimem(vocab_size=vocab_size, max_seq_len=block_size, n_layers=2, T=3).to(device),
        "3. Standard 4-Layer (815k)": DeepStandardTransformer(vocab_size=vocab_size, max_seq_len=block_size, n_layers=4).to(device),
        "4. Gravimem 4-Layer (T=3, 815k)": DeepGravimem(vocab_size=vocab_size, max_seq_len=block_size, n_layers=4, T=3).to(device),
    }

    lm_results = {}
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n>>> Training {name} ({param_count:,} parameters)...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        for step in range(1, 4001):
            model.train()
            bx, by = get_batch('train')
            opt.zero_grad()
            logits = model(bx, causal_mask=causal_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1000 == 0 or step == 4000:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch('val')
                    val_logits = model(vx, causal_mask=causal_mask)
                    val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), vy.view(-1)).item()
                print(f"Step {step:4d}/4000 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        lm_results[name] = {"params": param_count, "val_loss": val_loss}

    print("\n" + "=" * 80)
    print("  SUMMARY: DEEP GRAVIMEM VS STANDARD TRANSFORMER SCALING")
    print("=" * 80)
    for k, v in lm_results.items():
        print(f"{k:40s} | Params: {v['params']:,} | Val Loss: {v['val_loss']:.4f}")

    return lm_results


@app.local_entrypoint()
def main():
    print("Launching Deep Scaling Suite on Modal GPU...")
    res = run_deep_scaling_suite.remote()
    print("\nDeep Scaling Suite Finished!")
    print("Results:", res)
