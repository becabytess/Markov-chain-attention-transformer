"""
Mechanistic Experiment 1: Does Gravimem Iteratively Approximate Dense Attention?
Directly measures representation alignment, cosine similarity, KL divergence, and Top-1 agreement
between progressive Gravimem thought hops (T=1..8) and a trained Dense Transformer.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-mech1-dense-approx", image=image)


@app.function(gpu="T4", timeout=3600)
def run_dense_approximation():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  MECHANISTIC EXP 1: DENSE ATTENTION APPROXIMATION & TRAJECTORY ALIGNMENT")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "input.txt")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    seq_len = 256
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1500

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    # --- 1. Train Dense Transformer Baseline ---
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_mlp):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.d_k = d_model // n_heads
            self.n_heads = n_heads

        def forward(self, x, causal_mask):
            B, L, D = x.shape
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + causal_mask[:L, :L]
            attn = F.softmax(scores, dim=-1)
            H = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return x

    class DenseTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len + 16, d_model)
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, causal_mask, return_hidden=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x, causal_mask)
            h = self.ln_f(x)
            logits = self.head(h)
            if return_hidden:
                return logits, h
            return logits

    # --- 2. Train Gravimem Model ---
    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 255]

    class GravimemModel(nn.Module):
        def __init__(self, vocab_size, jump_offsets, default_T=4, d_model=128, d_mlp=512, max_len=256):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.K = len(jump_offsets)
            self.default_T = default_T
            self.d_model = d_model

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.jump_policy = nn.Linear(d_model, self.K)
            self.gru = nn.GRUCell(d_model, d_model)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            target_indices = torch.zeros((max_len, self.K), dtype=torch.long)
            valid_jump_mask = torch.zeros((max_len, self.K), dtype=torch.bool)
            for i in range(max_len):
                for k, offset in enumerate(self.jump_offsets):
                    target_pos = i - offset
                    if target_pos >= 0:
                        target_indices[i, k] = target_pos
                        valid_jump_mask[i, k] = True

            self.register_buffer("target_indices", target_indices)
            self.register_buffer("valid_jump_mask", valid_jump_mask)

        def forward(self, idx, T=None, return_all_hops=False):
            if T is None:
                T = self.default_T
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            V = self.v_proj(self.ln1(x_emb))
            target_idx = self.target_indices[:L]
            valid_mask = self.valid_jump_mask[:L]

            flat_targets = target_idx.unsqueeze(0).expand(B, -1, -1)
            V_jumps = torch.gather(
                V.unsqueeze(2).expand(-1, -1, self.K, -1),
                1,
                flat_targets.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            )
            V_jumps = V_jumps.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(-1), 0.0)

            s = x_emb
            step_hiddens = []
            step_logits = []

            for step in range(1, T + 1):
                s_norm = self.ln1(s)
                policy_logits = self.jump_policy(s_norm)
                policy_logits = policy_logits.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                pi = F.softmax(policy_logits, dim=-1)

                surfed_v = torch.sum(pi.unsqueeze(-1) * V_jumps, dim=2)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

                if return_all_hops:
                    h_step = s + self.mlp(self.ln2(s))
                    h_step_norm = self.ln_f(h_step)
                    step_hiddens.append(h_step_norm)
                    step_logits.append(self.head(h_step_norm))

            if return_all_hops:
                return step_logits, step_hiddens

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    print("\n---> Phase 1: Training Reference 4-Layer Dense Transformer...")
    dense_model = DenseTransformer(vocab_size).to(device)
    opt_dense = torch.optim.AdamW(dense_model.parameters(), lr=1e-3, weight_decay=1e-2)
    for step in range(1, num_steps + 1):
        dense_model.train()
        x, y = get_batch("train")
        opt_dense.zero_grad()
        loss = F.cross_entropy(dense_model(x, causal_mask).view(-1, vocab_size), y.view(-1))
        loss.backward()
        opt_dense.step()
    print("     Dense Transformer trained successfully!")

    print("\n---> Phase 2: Training Gravimem (Mixed T in [1, 6])...")
    grav_model = GravimemModel(vocab_size, jump_offsets).to(device)
    opt_grav = torch.optim.AdamW(grav_model.parameters(), lr=1e-3, weight_decay=1e-2)
    for step in range(1, num_steps + 1):
        grav_model.train()
        x, y = get_batch("train")
        T_curr = torch.randint(1, 7, (1,)).item()
        opt_grav.zero_grad()
        loss = F.cross_entropy(grav_model(x, T=T_curr).view(-1, vocab_size), y.view(-1))
        loss.backward()
        opt_grav.step()
    print("     Gravimem trained successfully!")

    # --- Phase 3: Alignment Evaluation across T = 1, 2, 3, 4, 6, 8 ---
    print("\n---> Phase 3: Measuring Hop-by-Hop Alignment to Dense Attention...")
    dense_model.eval()
    grav_model.eval()

    hop_levels = [1, 2, 3, 4, 6, 8]
    metrics = {T: {"cosine": 0.0, "kl_div": 0.0, "top1_agree": 0.0, "samples": 0} for T in hop_levels}

    with torch.no_grad():
        for _ in range(50):
            x_val, _ = get_batch("val")
            dense_logits, dense_hidden = dense_model(x_val, causal_mask, return_hidden=True)
            dense_probs = F.softmax(dense_logits, dim=-1)
            dense_top1 = dense_logits.argmax(dim=-1)

            step_logits, step_hiddens = grav_model(x_val, T=max(hop_levels), return_all_hops=True)

            for T in hop_levels:
                hop_idx = T - 1
                g_hidden = step_hiddens[hop_idx]
                g_logits = step_logits[hop_idx]
                g_probs = F.softmax(g_logits, dim=-1)
                g_log_probs = F.log_softmax(g_logits, dim=-1)
                g_top1 = g_logits.argmax(dim=-1)

                # 1. Cosine similarity of representations
                cos_sim = F.cosine_similarity(g_hidden, dense_hidden, dim=-1).mean().item()

                # 2. KL Divergence D_KL(P_dense || P_grav)
                kl = F.kl_div(g_log_probs, dense_probs, reduction="batchmean").item()

                # 3. Top-1 Prediction Agreement
                agree = (g_top1 == dense_top1).float().mean().item() * 100.0

                metrics[T]["cosine"] += cos_sim
                metrics[T]["kl_div"] += kl
                metrics[T]["top1_agree"] += agree
                metrics[T]["samples"] += 1

    print("\n" + "=" * 80)
    print("  DENSE ATTENTION CONVERGENCE & ALIGNMENT ACROSS THOUGHT HOPS (T)")
    print("=" * 80)
    print("Hop Step (T)  | Cosine Alignment to Dense | KL Divergence (P_d||P_g) | Top-1 Agreement")
    print("-" * 80)
    results = {}
    for T in hop_levels:
        c = metrics[T]["cosine"] / metrics[T]["samples"]
        k = metrics[T]["kl_div"] / metrics[T]["samples"]
        a = metrics[T]["top1_agree"] / metrics[T]["samples"]
        results[T] = {"cosine": c, "kl_div": k, "top1_agree": a}
        print(f"  T = {T:2d}       |         {c:7.4f}          |        {k:8.4f}          |     {a:5.2f} %")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main():
    print("Launching Mechanistic Experiment 1 on dedicated Modal GPU...")
    res = run_dense_approximation.remote()
    print("Experiment 1 complete!")
