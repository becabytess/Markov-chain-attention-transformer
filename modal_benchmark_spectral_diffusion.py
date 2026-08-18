"""
Benchmarking Exponential Repeated-Squaring vs Linear Markov Iteration vs Neumann Series in Gravimem:
- Linear Markov Surfer: M^(t+1) = (1-a) P M^(t) + a I (T matrix multiplies for T hops)
- Repeated-Squaring Fast Surfer: P^(2^k) computes 2^k hops with k matrix multiplies! (e.g., 16 hops in 4 matrix multiplies)
- Adaptive Pondering (PonderNet ACT): Dynamic per-token halting distribution.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-spectral-ponder", image=image)


@app.function(gpu="T4", timeout=3600)
def run_spectral_ponder_suite():
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: SPECTRAL DIFFUSION & REPEATED SQUARING (DEEP HOP REASONING)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Model Definitions
    # -------------------------------------------------------------

    # Model 1: Standard Linear Markov Unroll (T steps = T matmuls)
    class GravimemLinear(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, T=4):
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

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            P = F.softmax(scores, dim=-1)

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)

            for _ in range(self.T):
                M = (1.0 - alpha) * (P @ M) + alpha * I

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # Model 2: Repeated-Squaring Exponential Surfer (Computes 2^K hops in K steps)
    class GravimemRepeatedSquaring(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, K=3):
            """
            K=2 -> 4 hops in 2 steps
            K=3 -> 8 hops in 3 steps
            K=4 -> 16 hops in 4 steps
            """
            super().__init__()
            self.K = K
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

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            P = F.softmax(scores, dim=-1)

            # Lazy Random Walk Transition: P_lazy = (1-alpha) * P + alpha * I
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            alpha = torch.sigmoid(self.raw_alpha)
            P_lazy = (1.0 - alpha) * P + alpha * I

            # Repeated Squaring: P_lazy^(2^K)
            M = P_lazy
            for _ in range(self.K):
                M = M @ M

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # Model 3: PonderNet-Gravimem (Dynamic ACT Halting per Token)
    class GravimemPonderNet(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, max_T=6):
            super().__init__()
            self.max_T = max_T
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
            
            # Halting unit: predicts lambda_t in (0, 1)
            self.halting_head = nn.Linear(d_model, 1)
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            P = F.softmax(scores, dim=-1)
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)

            # PonderNet unroll
            halting_probs = [] # p_t
            remainders = torch.ones(B, L, 1, device=idx.device)
            accum_logits = torch.zeros(B, L, self.tok_emb.num_embeddings, device=idx.device)
            steps_taken = torch.zeros(B, L, device=idx.device)

            for step in range(self.max_T):
                M = (1.0 - alpha) * (P @ M) + alpha * I
                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x_step = x + self.out(H)
                x_step = x_step + self.mlp(self.ln2(x_step))
                step_logits = self.head(self.ln_f(x_step))

                if step < self.max_T - 1:
                    lambda_t = torch.sigmoid(self.halting_head(x_step)) # (B, L, 1)
                    p_t = remainders * lambda_t
                    remainders = remainders * (1.0 - lambda_t)
                else:
                    p_t = remainders # last step absorbs remainder

                accum_logits = accum_logits + p_t * step_logits
                halting_probs.append(p_t)
                steps_taken = steps_taken + (step + 1) * p_t.squeeze(-1)

            # Stack halting distribution: (B, L, max_T)
            p_dist = torch.cat(halting_probs, dim=-1)
            return accum_logits, p_dist, steps_taken

    # Standard Baselines
    class Standard1Layer(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512):
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

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            att = F.softmax(scores, dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------
    # 2. Deep Multi-Hop Dataset (4-Hop and 8-Hop Paths)
    # -------------------------------------------------------------
    print("\n--- Generating Deep Multi-Hop Reasoning Data (4-Hop & 8-Hop) ---")
    num_nodes = 48
    ARROW = num_nodes
    QUERY = num_nodes + 1
    vocab_size = num_nodes + 2

    def make_deep_hops(n_samples, hops=4):
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

    # 4-Hop dataset
    x4_tr, y4_tr = make_deep_hops(40000, hops=4)
    x4_te, y4_te = make_deep_hops(4000, hops=4)

    # 8-Hop dataset
    x8_tr, y8_tr = make_deep_hops(40000, hops=8)
    x8_te, y8_te = make_deep_hops(4000, hops=8)

    loader_4hop = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x4_tr, y4_tr), batch_size=128, shuffle=True
    )
    loader_8hop = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x8_tr, y8_tr), batch_size=128, shuffle=True
    )

    # =========================================================================
    # PART 1: 4-HOP BENCHMARK
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: 4-HOP GRAPH REASONING BENCHMARK")
    print("=" * 70)

    models_4hop = {
        "1. Standard 1-Layer (208k)": Standard1Layer(vocab_size=vocab_size, max_seq_len=32).to(device),
        "2. Gravimem-Linear (T=4 hops)": GravimemLinear(vocab_size=vocab_size, max_seq_len=32, T=4).to(device),
        "3. Gravimem-Squaring (K=2, 4 hops in 2 mults)": GravimemRepeatedSquaring(vocab_size=vocab_size, max_seq_len=32, K=2).to(device),
        "4. Gravimem-PonderNet (Adaptive ACT)": GravimemPonderNet(vocab_size=vocab_size, max_seq_len=32, max_T=6).to(device),
    }

    results_4hop = {}
    for name, model in models_4hop.items():
        print(f"\n>>> Training {name} on 4-Hop Graphs...")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        step = 0
        for epoch in range(5):
            for bx, by in loader_4hop:
                step += 1
                if step > 2000:
                    break
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                
                if "PonderNet" in name:
                    logits, p_dist, steps_taken = model(bx)
                    query_logits = logits[:, -1, :]
                    task_loss = F.cross_entropy(query_logits, by)
                    # PonderNet KL regularization towards geometric prior p_prior(t) ~ Geom(0.5)
                    # encourage efficiency while penalizing unnecessary pondering
                    geom_prior = torch.tensor([0.5 * (0.5**t) for t in range(6)], device=device)
                    geom_prior = geom_prior / geom_prior.sum()
                    query_p = p_dist[:, -1, :] # (B, max_T)
                    kl_loss = F.kl_div(query_p.log().clamp(min=-100), geom_prior.unsqueeze(0).expand_as(query_p), reduction='batchmean')
                    loss = task_loss + 0.01 * kl_loss
                else:
                    logits = model(bx)[:, -1, :]
                    loss = F.cross_entropy(logits, by)

                loss.backward()
                opt.step()
            if step > 2000:
                break

        # Eval
        model.eval()
        correct, total = 0, 0
        avg_steps = []
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x4_te, y4_te), batch_size=256)
        with torch.no_grad():
            for bx, by in eval_loader:
                bx, by = bx.to(device), by.to(device)
                if "PonderNet" in name:
                    logits, p_dist, steps_taken = model(bx)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    avg_steps.append(steps_taken[:, -1].mean().item())
                else:
                    preds = model(bx)[:, -1, :].argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = 100.0 * correct / total
        extra = f" | Avg Steps Taken: {sum(avg_steps)/len(avg_steps):.2f}" if avg_steps else ""
        print(f"4-Hop Accuracy: {acc:.2f}%{extra}")
        results_4hop[name] = acc

    # =========================================================================
    # PART 2: 8-HOP DEEP DIFFUSION BENCHMARK
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: 8-HOP DEEP GRAPH REASONING BENCHMARK")
    print("=" * 70)

    models_8hop = {
        "1. Gravimem-Linear (T=8 hops, 8 mults)": GravimemLinear(vocab_size=vocab_size, max_seq_len=64, T=8).to(device),
        "2. Gravimem-Squaring (K=3, 8 hops in 3 mults!)": GravimemRepeatedSquaring(vocab_size=vocab_size, max_seq_len=64, K=3).to(device),
        "3. Gravimem-PonderNet (Adaptive max_T=10)": GravimemPonderNet(vocab_size=vocab_size, max_seq_len=64, max_T=10).to(device),
    }

    results_8hop = {}
    for name, model in models_8hop.items():
        print(f"\n>>> Training {name} on 8-Hop Graphs...")
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        step = 0
        for epoch in range(6):
            for bx, by in loader_8hop:
                step += 1
                if step > 2500:
                    break
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                
                if "PonderNet" in name:
                    logits, p_dist, steps_taken = model(bx)
                    query_logits = logits[:, -1, :]
                    task_loss = F.cross_entropy(query_logits, by)
                    geom_prior = torch.tensor([0.3 * (0.7**t) for t in range(10)], device=device)
                    geom_prior = geom_prior / geom_prior.sum()
                    query_p = p_dist[:, -1, :]
                    kl_loss = F.kl_div(query_p.log().clamp(min=-100), geom_prior.unsqueeze(0).expand_as(query_p), reduction='batchmean')
                    loss = task_loss + 0.01 * kl_loss
                else:
                    logits = model(bx)[:, -1, :]
                    loss = F.cross_entropy(logits, by)

                loss.backward()
                opt.step()
            if step > 2500:
                break

        # Eval
        model.eval()
        correct, total = 0, 0
        avg_steps = []
        eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x8_te, y8_te), batch_size=256)
        with torch.no_grad():
            for bx, by in eval_loader:
                bx, by = bx.to(device), by.to(device)
                if "PonderNet" in name:
                    logits, p_dist, steps_taken = model(bx)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    avg_steps.append(steps_taken[:, -1].mean().item())
                else:
                    preds = model(bx)[:, -1, :].argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = 100.0 * correct / total
        extra = f" | Avg Steps Taken: {sum(avg_steps)/len(avg_steps):.2f}" if avg_steps else ""
        print(f"8-Hop Accuracy: {acc:.2f}%{extra}")
        results_8hop[name] = acc

    print("\n" + "=" * 80)
    print("  SUMMARY: SPECTRAL DIFFUSION & PONDERNET BENCHMARK RESULTS")
    print("=" * 80)
    print("\n--- 4-Hop Reasoning Results ---")
    for k, v in results_4hop.items():
        print(f"{k:45s} | Accuracy: {v:.2f}%")

    print("\n--- 8-Hop Deep Reasoning Results ---")
    for k, v in results_8hop.items():
        print(f"{k:45s} | Accuracy: {v:.2f}%")

    return {
        "results_4hop": results_4hop,
        "results_8hop": results_8hop
    }


@app.local_entrypoint()
def main():
    print("Launching Spectral Diffusion & PonderNet Suite on Modal GPU...")
    res = run_spectral_ponder_suite.remote()
    print("\nSuite Completed!")
    print("Results:", res)
