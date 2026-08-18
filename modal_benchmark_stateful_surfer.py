"""
Gravimem Stateful Surfer Backpack & Memory Cell Benchmark on Modal GPU:
Compares how the surfer accumulates knowledge along its trajectory:
1. Pure Markov Fluid Baseline (Matrix Diffusion M @ V)
2. Residual Backpack Surfer: s^(t+1) = LayerNorm(s^(t) + gathered_V)
3. Gated GRU Backpack Surfer: s^(t+1) = GRUCell(gathered_V, s^(t))
4. Gated MLP Backpack Surfer: s^(t+1) = s^(t) + gate * MLP(gathered_V)

Evaluates on:
- Multi-Hop Variable Dependency Tracking
- TinyShakespeare Autoregressive Language Modeling
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

app = modal.App("gravimem-stateful-surfer", image=image)


@app.function(gpu="T4", timeout=3600)
def run_stateful_surfer_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: STATEFUL SURFER BACKPACK & TRAJECTORY ACCUMULATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # PART 1: Variable Tracking Chains (x = 5, y = x + 3, z = y - 2 -> z?)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  EXPERIMENT 1: VARIABLE DEPENDENCY TRACKING (4-STEP CHAINS)")
    print("=" * 80)

    NUM_VARS = 10
    MAX_VAL = 20
    VOCAB_SIZE = NUM_VARS + MAX_VAL + 10

    def generate_var_dataset(num_samples=15000, chain_len=4):
        samples = []
        for _ in range(num_samples):
            var_indices = random.sample(range(NUM_VARS), chain_len)
            init_val = random.randint(1, 9)
            cur_val = init_val
            tokens = [var_indices[0], cur_val]
            for step in range(1, chain_len):
                prev_var = var_indices[step - 1]
                cur_var = var_indices[step]
                delta = random.randint(1, 3)
                cur_val = (cur_val + delta) % MAX_VAL
                tokens.extend([cur_var, prev_var, delta])
            
            query_var = var_indices[-1]
            tokens.append(query_var)
            target = cur_val
            samples.append((tokens, target))
        return samples

    train_vars = generate_var_dataset(num_samples=16000, chain_len=4)
    val_vars = generate_var_dataset(num_samples=2000, chain_len=4)
    max_len = max(len(s[0]) for s in train_vars)

    def pad_and_batch_vars(data, batch_size=128):
        batch = random.sample(data, batch_size)
        x = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        for i, b in enumerate(batch):
            x[i, :len(b[0])] = torch.tensor(b[0], dtype=torch.long, device=device)
        return x, y

    class StatefulSurferVarModel(nn.Module):
        def __init__(self, backpack_type="fluid", T=4, d_model=128, n_heads=4):
            super().__init__()
            self.backpack_type = backpack_type
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

            if backpack_type == "gru":
                self.gru = nn.GRUCell(d_model, d_model)
            elif backpack_type == "gated_mlp":
                self.gate = nn.Linear(d_model * 2, d_model)
                self.update_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model)
                )

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, MAX_VAL, bias=False)
            self.alpha = nn.Parameter(torch.tensor(0.2))

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x_emb)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            P = F.softmax(scores, dim=-1)

            if self.backpack_type == "fluid":
                I = torch.eye(L, device=idx.device, dtype=x_emb.dtype).unsqueeze(0).unsqueeze(0)
                M = I.expand(B, self.n_heads, L, L)
                alpha = torch.sigmoid(self.alpha)
                for _ in range(self.T):
                    M = (1.0 - alpha) * (P @ M) + alpha * I
                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x_emb + self.out(H)

            elif self.backpack_type == "residual":
                s = x_emb
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                    s = s + self.out(gathered)
                x = s

            elif self.backpack_type == "gru":
                s = x_emb.view(-1, self.d_model)
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B*L, self.d_model)
                    s = self.gru(self.out(gathered), s)
                x = s.view(B, L, self.d_model)

            elif self.backpack_type == "gated_mlp":
                s = x_emb
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                    gathered_proj = self.out(gathered)
                    g = torch.sigmoid(self.gate(torch.cat([s, gathered_proj], dim=-1)))
                    s = s + g * self.update_mlp(gathered_proj)
                x = s

            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x[:, -1, :]))
            return logits

    backpack_configs = [
        ("1. Pure Markov Fluid (M @ V Baseline)", "fluid"),
        ("2. Residual Backpack Surfer (s + V_t)", "residual"),
        ("3. Gated GRU Backpack Surfer (GRUCell)", "gru"),
        ("4. Gated MLP Backpack Surfer (Gate * MLP)", "gated_mlp"),
    ]

    var_results = {}
    for name, b_type in backpack_configs:
        print(f"\n>>> Training {name} on Variable Tracking...")
        model = StatefulSurferVarModel(backpack_type=b_type, T=4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        for step in range(1, 1501):
            model.train()
            bx, by = pad_and_batch_vars(train_vars, batch_size=128)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vx, vy = pad_and_batch_vars(val_vars, batch_size=1000)
            v_logits = model(vx)
            v_preds = torch.argmax(v_logits, dim=-1)
            acc = (v_preds == vy).float().mean().item() * 100

        print(f"--> {name} Final Val Accuracy: {acc:.2f}%")
        var_results[name] = acc

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

    class StatefulLMSurfer(nn.Module):
        def __init__(self, backpack_type="fluid", T=3, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.backpack_type = backpack_type
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

            if backpack_type == "gru":
                self.gru = nn.GRUCell(d_model, d_model)
            elif backpack_type == "gated_mlp":
                self.gate = nn.Linear(d_model * 2, d_model)
                self.update_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model)
                )

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
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + causal_mask[:L, :L]
            P = F.softmax(scores, dim=-1)

            if self.backpack_type == "fluid":
                I = torch.eye(L, device=idx.device, dtype=x_emb.dtype).unsqueeze(0).unsqueeze(0)
                M = I.expand(B, self.n_heads, L, L)
                alpha = torch.sigmoid(self.alpha)
                for _ in range(self.T):
                    M = (1.0 - alpha) * (P @ M) + alpha * I
                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x_emb + self.out(H)

            elif self.backpack_type == "residual":
                s = x_emb
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                    s = s + (1.0 / math.sqrt(self.T)) * self.out(gathered)
                x = s

            elif self.backpack_type == "gru":
                s = x_emb.view(-1, self.d_model)
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B*L, self.d_model)
                    s = self.gru(self.out(gathered), s)
                x = s.view(B, L, self.d_model)

            elif self.backpack_type == "gated_mlp":
                s = x_emb
                for _ in range(self.T):
                    gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                    gathered_proj = self.out(gathered)
                    g = torch.sigmoid(self.gate(torch.cat([s, gathered_proj], dim=-1)))
                    s = s + g * self.update_mlp(gathered_proj)
                x = s

            x = x + self.mlp(self.ln2(x))
            logits = self.head(self.ln_f(x))
            return logits

    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)

    lm_results = {}
    for name, b_type in backpack_configs:
        print(f"\n>>> Training LM {name} (3,000 steps)...")
        model = StatefulLMSurfer(backpack_type=b_type, T=3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)

        for step in range(1, 3001):
            model.train()
            bx, by = get_lm_batch('train')
            opt.zero_grad()
            logits = model(bx, causal_mask=causal_mask)
            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    print("  SUMMARY: STATEFUL SURFER BACKPACK BENCHMARK RESULTS")
    print("=" * 80)
    print("\n[Part 1: Variable Tracking Accuracy]")
    for k, v in var_results.items():
        print(f"  {k:50s} : {v:.2f}%")

    print("\n[Part 2: TinyShakespeare LM Validation Loss]")
    for k, v in lm_results.items():
        print(f"  {k:50s} : {v:.4f}")

    return {
        "var_results": var_results,
        "lm_results": lm_results
    }


@app.local_entrypoint()
def main():
    print("Launching Stateful Surfer Benchmark on Modal GPU...")
    res = run_stateful_surfer_suite.remote()
    print("\nFinished Successfully!")
