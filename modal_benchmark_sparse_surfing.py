"""
Gravimem Sparse & Discrete Path Surfing Benchmark on Modal GPU:
Compares:
1. Standard Dense Transformer (Full Softmax Attention)
2. Soft Gravimem (Continuous Fluid Markov Mass Diffusion)
3. Top-k Sparse Gravimem (k=2 sparse routing per hop)
4. Hard Top-1 Discrete Surfer (Straight-Through Estimator argmax path hopping)
5. Gumbel-Softmax Stochastic Surfer (Temperature-annealed discrete sampling)

Evaluated on:
- Multi-Hop Relational Graph Navigation (A -> B -> C -> D)
- TinyShakespeare Autoregressive Language Modeling
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

app = modal.App("gravimem-sparse-surfing", image=image)


@app.function(gpu="T4", timeout=3600)
def run_sparse_surfing_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: SPARSE & DISCRETE PATH SURFING BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # PART 1: Multi-Hop Relational Graph Navigation (3-Hop)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: 3-HOP GRAPH NAVIGATION (A -> B -> C -> D)")
    print("=" * 80)

    NUM_NODES = 26
    VOCAB_SIZE = NUM_NODES + 4
    PAD_TOKEN = 0
    START_TOKEN = NUM_NODES + 1
    TARGET_TOKEN = NUM_NODES + 2
    QUERY_TOKEN = NUM_NODES + 3

    def generate_graph_dataset(num_samples=15000, num_edges=12, num_hops=3):
        samples = []
        for _ in range(num_samples):
            nodes = list(range(1, NUM_NODES + 1))
            random.shuffle(nodes)
            chain = nodes[:num_hops + 1]
            chain_edges = [(chain[i], chain[i+1]) for i in range(num_hops)]

            other_edges = set()
            while len(other_edges) < (num_edges - num_hops):
                u, v = random.sample(nodes, 2)
                if (u, v) not in chain_edges and u != v:
                    other_edges.add((u, v))

            all_edges = chain_edges + list(other_edges)
            random.shuffle(all_edges)

            seq = []
            for u, v in all_edges:
                seq.extend([u, v])

            start_node = chain[0]
            target_node = chain[-1]
            seq.extend([QUERY_TOKEN, start_node])
            samples.append((seq, target_node - 1))

        return samples

    train_data = generate_graph_dataset(num_samples=16000, num_edges=12, num_hops=3)
    val_data = generate_graph_dataset(num_samples=2000, num_edges=12, num_hops=3)
    seq_len = len(train_data[0][0])

    def get_graph_batch(data, batch_size=128):
        batch = random.sample(data, batch_size)
        x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        return x, y

    # Models for Graph Task
    class SparseGraphSurfer(nn.Module):
        def __init__(self, mode="soft", k=2, T=3, d_model=128, n_heads=4):
            super().__init__()
            self.mode = mode  # 'soft', 'topk', 'hard_ste', 'gumbel'
            self.k = k
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
            self.pos_emb = nn.Embedding(seq_len + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, NUM_NODES, bias=False)
            self.alpha = nn.Parameter(torch.tensor(0.2))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x_emb)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if self.mode == "soft":
                P = F.softmax(scores, dim=-1)
            elif self.mode == "topk":
                # Keep only top-k transitions per row, renormalize
                topk_vals, topk_indices = torch.topk(scores, k=self.k, dim=-1)
                mask = torch.full_like(scores, float('-inf'))
                mask.scatter_(-1, topk_indices, topk_vals)
                P = F.softmax(mask, dim=-1)
            elif self.mode == "hard_ste":
                # Hard argmax with Straight-Through Estimator
                soft_P = F.softmax(scores, dim=-1)
                hard_idx = torch.argmax(soft_P, dim=-1, keepdim=True)
                hard_P = torch.zeros_like(soft_P).scatter_(-1, hard_idx, 1.0)
                P = hard_P - soft_P.detach() + soft_P
            elif self.mode == "gumbel":
                # Gumbel-Softmax with temperature tau=0.5
                P = F.gumbel_softmax(scores, tau=0.5, hard=True, dim=-1)

            I = torch.eye(L, device=idx.device, dtype=x_emb.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.alpha)

            for _ in range(self.T):
                M = (1.0 - alpha) * (P @ M) + alpha * I

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x_emb + self.out(H)
            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x[:, -1, :]))
            return logits

    modes_to_test = [
        ("1. Soft Gravimem (Dense Markov T=3)", "soft", 2),
        ("2. Top-2 Sparse Surfer (k=2 per hop)", "topk", 2),
        ("3. Top-4 Sparse Surfer (k=4 per hop)", "topk", 4),
        ("4. Hard Top-1 Discrete Surfer (STE argmax)", "hard_ste", 1),
        ("5. Gumbel-Softmax Stochastic Surfer (tau=0.5)", "gumbel", 1),
    ]

    graph_results = {}
    for name, mode, k in modes_to_test:
        print(f"\n>>> Training {name} on 3-Hop Graph...")
        model = SparseGraphSurfer(mode=mode, k=k, T=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        for step in range(1, 1501):
            model.train()
            bx, by = get_graph_batch(train_data, batch_size=128)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vx, vy = get_graph_batch(val_data, batch_size=1000)
            v_logits = model(vx)
            v_preds = torch.argmax(v_logits, dim=-1)
            acc = (v_preds == vy).float().mean().item() * 100

        print(f"--> {name} Final Val Accuracy: {acc:.2f}%")
        graph_results[name] = acc

    # =========================================================================
    # PART 2: Autoregressive Language Modeling (TinyShakespeare)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: AUTOREGRESSIVE LANGUAGE MODELING (TINYSHAKESPEARE)")
    print("=" * 80)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_lm = data[:n_train]
    val_lm = data[n_train:]

    block_size = 128
    batch_size = 64

    def get_lm_batch(split):
        d = train_lm if split == 'train' else val_lm
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    class SparseLMSurfer(nn.Module):
        def __init__(self, mode="soft", k=4, T=3, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.mode = mode
            self.k = k
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
            self.alpha = nn.Parameter(torch.tensor(-1.0))

        def forward(self, idx, causal_mask):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x_emb)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + causal_mask[:L, :L]

            if self.mode == "soft":
                P = F.softmax(scores, dim=-1)
            elif self.mode == "topk":
                topk_vals, topk_indices = torch.topk(scores, k=min(self.k, L), dim=-1)
                mask = torch.full_like(scores, float('-inf'))
                mask.scatter_(-1, topk_indices, topk_vals)
                P = F.softmax(mask, dim=-1)
            elif self.mode == "hard_ste":
                soft_P = F.softmax(scores, dim=-1)
                hard_idx = torch.argmax(soft_P, dim=-1, keepdim=True)
                hard_P = torch.zeros_like(soft_P).scatter_(-1, hard_idx, 1.0)
                P = hard_P - soft_P.detach() + soft_P
            elif self.mode == "gumbel":
                P = F.gumbel_softmax(scores, tau=0.5, hard=True, dim=-1)

            I = torch.eye(L, device=idx.device, dtype=x_emb.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.alpha)

            for _ in range(self.T):
                M = (1.0 - alpha) * (P @ M) + alpha * I

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x_emb + self.out(H)
            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x))
            return logits

    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)

    lm_modes = [
        ("1. Soft Dense Gravimem (T=3)", "soft", 4),
        ("2. Top-4 Sparse Surfer (k=4)", "topk", 4),
        ("3. Top-2 Sparse Surfer (k=2)", "topk", 2),
        ("4. Hard Top-1 Discrete Surfer (STE)", "hard_ste", 1),
    ]

    lm_results = {}
    for name, mode, k in lm_modes:
        print(f"\n>>> Training LM {name} (3,000 steps)...")
        model = SparseLMSurfer(mode=mode, k=k, T=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)

        for step in range(1, 3001):
            model.train()
            bx, by = get_lm_batch('train')
            opt.zero_grad()
            logits = model(bx, causal_mask=causal_mask)
            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            vx, vy = get_lm_batch('val')
            v_logits = model(vx, causal_mask=causal_mask)
            v_loss = F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1)).item()

        print(f"--> {name} Final Val Loss: {v_loss:.4f}")
        lm_results[name] = v_loss

    print("\n" + "=" * 80)
    print("  SUMMARY: SPARSE & DISCRETE SURFING EXPERIMENT RESULTS")
    print("=" * 80)
    print("\n[Part 1: 3-Hop Relational Graph Accuracy]")
    for k, v in graph_results.items():
        print(f"  {k:50s} : {v:.2f}%")

    print("\n[Part 2: TinyShakespeare LM Validation Loss]")
    for k, v in lm_results.items():
        print(f"  {k:50s} : {v:.4f}")

    return {
        "graph_results": graph_results,
        "lm_results": lm_results
    }


@app.local_entrypoint()
def main():
    print("Launching Sparse & Discrete Surfing Suite on Modal GPU...")
    res = run_sparse_surfing_suite.remote()
    print("\nFinished Successfully!")
