"""
Benchmarking MLP Variants in Gravimem on Modal GPU:
1. Gravimem-SharedMLP: 1 Shared Reusable MLP looped across all T steps (223k params)
2. Gravimem-UniqueMLPs: Distinct Non-Reusable MLPs per step (MLP_1, ..., MLP_T) (485k params)
3. Gravimem-InnerSettling: Pure inner Markov settling M^(T) -> 1 Single Post-MLP (223k params)
4. Standard 3-Layer Baseline (618k params)
5. Standard 1-Layer Baseline (222k params)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-mlp-variants", image=image)


@app.function(gpu="T4", timeout=3600)
def run_mlp_variants():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: REUSABLE VS NON-REUSABLE MLP ARCHITECTURE BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Model Definitions
    # -------------------------------------------------------------
    
    # VARIANT A: Shared Reusable MLP looped across T steps
    class GravimemSharedMLP(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.T = T
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.step_emb = nn.Embedding(T, d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            res_scale = 1.0 / math.sqrt(self.T)

            for step in range(self.T):
                step_vec = self.step_emb(torch.tensor(step, device=idx.device)).unsqueeze(0).unsqueeze(0)
                x_norm = self.ln1(x + step_vec)
                Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
                if causal_mask is not None:
                    scores = scores + causal_mask[:L, :L]
                P = F.softmax(scores, dim=-1)
                M = (1.0 - alpha) * (P @ M) + alpha * I
                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + res_scale * self.out(H)
                x = x + res_scale * self.mlp(self.ln2(x))

            return self.head(self.ln_f(x))

    # VARIANT B: Non-Reusable / Unique MLPs per step (MLP_1, ..., MLP_T)
    class GravimemUniqueMLPs(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.T = T
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            
            # Non-reusable MLPs: Each settling step gets its own dedicated MLP & LayerNorm!
            self.mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
                for _ in range(T)
            ])
            self.ln_mlps = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(T)])
            
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            res_scale = 1.0 / math.sqrt(self.T)

            for step in range(self.T):
                x_norm = self.ln1(x)
                Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
                if causal_mask is not None:
                    scores = scores + causal_mask[:L, :L]
                P = F.softmax(scores, dim=-1)
                M = (1.0 - alpha) * (P @ M) + alpha * I
                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + res_scale * self.out(H)
                # Apply step-specific non-reusable MLP
                x = x + res_scale * self.mlps[step](self.ln_mlps[step](x))

            return self.head(self.ln_f(x))

    # VARIANT C: Inner Markov Mass Settling -> Single Post-Settling MLP (No recurrent MLP)
    class GravimemInnerSettling(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.T = T
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            
            # 1. Project Q, K, V once
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if causal_mask is not None:
                scores = scores + causal_mask[:L, :L]
            P = F.softmax(scores, dim=-1)

            # 2. Pure Markov Mass Settling Loop for T steps
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            for _ in range(self.T):
                M = (1.0 - alpha) * (P @ M) + alpha * I

            # 3. Read out mixed representation and apply single post-MLP
            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))

            return self.head(self.ln_f(x))

    # Standard Baselines
    class Standard1Layer(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
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
            return self.head(self.ln_f(x))

    class Standard3Layer(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'q': nn.Linear(d_model, d_model, bias=False),
                    'k': nn.Linear(d_model, d_model, bias=False),
                    'v': nn.Linear(d_model, d_model, bias=False),
                    'out': nn.Linear(d_model, d_model, bias=False),
                    'ln1': nn.LayerNorm(d_model),
                    'ln2': nn.LayerNorm(d_model),
                    'mlp': nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
                }) for _ in range(3)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for l in self.layers:
                x_norm = l['ln1'](x)
                Q = l['q'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = l['k'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = l['v'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
                if causal_mask is not None:
                    scores = scores + causal_mask[:L, :L]
                att = F.softmax(scores, dim=-1)
                H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + l['out'](H)
                x = x + l['mlp'](l['ln2'](x))
            return self.head(self.ln_f(x))

    # =========================================================================
    # PART 1: LANGUAGE MODELING BENCHMARK (TinyShakespeare, 3,000 Steps)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 1: AUTOREGRESSIVE LANGUAGE MODELING (3,000 STEPS)")
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
        "1. Standard 1-Layer (222k)": Standard1Layer(vocab_size=vocab_size, max_seq_len=block_size).to(device),
        "2. Gravimem-SharedMLP (223k)": GravimemSharedMLP(vocab_size=vocab_size, max_seq_len=block_size, T=3).to(device),
        "3. Gravimem-InnerSettling (223k)": GravimemInnerSettling(vocab_size=vocab_size, max_seq_len=block_size, T=3).to(device),
        "4. Gravimem-UniqueMLPs (485k)": GravimemUniqueMLPs(vocab_size=vocab_size, max_seq_len=block_size, T=3).to(device),
        "5. Standard 3-Layer (618k)": Standard3Layer(vocab_size=vocab_size, max_seq_len=block_size).to(device),
    }

    lm_results = {}
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n>>> Training {name} ({param_count:,} parameters)...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        for step in range(1, 3001):
            model.train()
            bx, by = get_batch('train')
            opt.zero_grad()
            logits = model(bx, causal_mask=causal_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1000 == 0 or step == 3000:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch('val')
                    val_logits = model(vx, causal_mask=causal_mask)
                    val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), vy.view(-1)).item()
                print(f"Step {step:4d}/3000 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        lm_results[name] = {"params": param_count, "val_loss": val_loss}

    # =========================================================================
    # PART 2: MULTI-HOP RELATIONAL REASONING BENCHMARK (3-HOP GRAPHS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  PART 2: MULTI-HOP REASONING BENCHMARK (3-HOP GRAPH NAVIGATION)")
    print("=" * 70)

    num_nodes = 32
    ARROW = num_nodes
    QUERY = num_nodes + 1
    g_vocab = num_nodes + 2

    def make_hops_data(n_samples, hops=3):
        inputs, targets = [], []
        for _ in range(n_samples):
            nodes = random.sample(range(num_nodes), hops + 1)
            edges = [(nodes[i], nodes[i+1]) for i in range(hops)]
            random.shuffle(edges)
            seq = []
            for u, v in edges:
                seq.extend([u, ARROW, v])
            seq.extend([nodes[0], QUERY])
            inputs.append(seq)
            targets.append(nodes[-1])
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    gx_tr, gy_tr = make_hops_data(50000, hops=3)
    gx_te, gy_te = make_hops_data(5000, hops=3)

    g_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(gx_tr, gy_tr), batch_size=128, shuffle=True
    )

    g_models = {
        "1. Standard 1-Layer (208k)": Standard1Layer(vocab_size=g_vocab, max_seq_len=16).to(device),
        "2. Gravimem-SharedMLP (208k)": GravimemSharedMLP(vocab_size=g_vocab, max_seq_len=16, T=3).to(device),
        "3. Gravimem-InnerSettling (208k)": GravimemInnerSettling(vocab_size=g_vocab, max_seq_len=16, T=3).to(device),
        "4. Gravimem-UniqueMLPs (470k)": GravimemUniqueMLPs(vocab_size=g_vocab, max_seq_len=16, T=3).to(device),
        "5. Standard 3-Layer (603k)": Standard3Layer(vocab_size=g_vocab, max_seq_len=16).to(device),
    }

    g_results = {}
    for name, model in g_models.items():
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        print(f"\n>>> Training {name} on 3-Hop Graphs...")
        step = 0
        for epoch in range(6):
            for bx, by in g_loader:
                step += 1
                if step > 2000:
                    break
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                logits = model(bx)[:, -1, :]
                loss = F.cross_entropy(logits, by)
                loss.backward()
                opt.step()
            if step > 2000:
                break

        # Eval
        model.eval()
        correct, total = 0, 0
        l = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(gx_te, gy_te), batch_size=256)
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)[:, -1, :].argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = 100.0 * correct / total
        print(f"Final 3-Hop Accuracy: {acc:.2f}%")
        g_results[name] = acc

    print("\n" + "=" * 80)
    print("  SUMMARY: REUSABLE VS NON-REUSABLE MLP COMPARISON")
    print("=" * 80)
    print("\n--- Language Modeling (TinyShakespeare Val Loss) ---")
    for k, v in lm_results.items():
        print(f"{k:35s} | Params: {v['params']:,} | Val Loss: {v['val_loss']:.4f}")

    print("\n--- 3-Hop Relational Reasoning Accuracy ---")
    for k, v in g_results.items():
        print(f"{k:35s} | Accuracy: {v:.2f}%")

    return {
        "lm_results": lm_results,
        "multihop_results": g_results
    }


@app.local_entrypoint()
def main():
    print("Launching MLP Variants Benchmark on Modal GPU...")
    res = run_mlp_variants.remote()
    print("\nBenchmark Finished!")
    print("Results:", res)
