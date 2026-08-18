"""
Benchmarking Temperature-Annealed Surfing and Dynamic Decay Schedules in Gravimem on Modal GPU:
1. Standard Gravimem (Fixed temp tau=1.0)
2. Annealed Gravimem: Linear Decay (tau: 2.0 -> 0.5) [Diffuse exploration -> Focused retrieval]
3. Annealed Gravimem: Learned Per-Step Temperature Parameter tau_t
4. Exponential Teleport Prior Decay: alpha_t decaying across steps
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-annealed-surfing", image=image)


@app.function(gpu="T4", timeout=3600)
def run_annealing_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: TEMPERATURE-ANNEALED SURFING & DYNAMIC TRANSITION DYNAMICS")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # Model Architectures
    # -------------------------------------------------------------
    class AnnealedGravimemLM(nn.Module):
        def __init__(self, vocab_size, max_seq_len=256, d_model=128, n_heads=4, d_mlp=512, T=3, mode="fixed"):
            super().__init__()
            self.T = T
            self.mode = mode
            self.d_k = d_model // n_heads
            self.n_heads = n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)

            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            
            if mode == "learned_temp":
                # Learned log temperatures for each step t
                self.log_temps = nn.Parameter(torch.zeros(T))
            elif mode == "learned_alpha_schedule":
                self.raw_alphas = nn.Parameter(torch.full((T, n_heads, 1, 1), -1.73))

        def forward(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            raw_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

            I = torch.eye(L, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)

            for t in range(self.T):
                if self.mode == "fixed":
                    tau = 1.0
                    alpha = torch.sigmoid(self.raw_alpha)
                elif self.mode == "linear_anneal":
                    # Cool from tau=2.0 down to tau=0.5
                    tau = 2.0 - (1.5 * t / max(1, self.T - 1))
                    alpha = torch.sigmoid(self.raw_alpha)
                elif self.mode == "learned_temp":
                    tau = torch.exp(self.log_temps[t]).clamp(0.2, 5.0)
                    alpha = torch.sigmoid(self.raw_alpha)
                elif self.mode == "learned_alpha_schedule":
                    tau = 1.0
                    alpha = torch.sigmoid(self.raw_alphas[t])

                scaled_scores = raw_scores / tau
                if causal_mask is not None:
                    scaled_scores = scaled_scores + causal_mask[:L, :L]
                P_t = F.softmax(scaled_scores, dim=-1)

                M = (1.0 - alpha) * (P_t @ M) + alpha * I

            H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # =========================================================================
    # BENCHMARK ON TINYSHAKESPEARE LM (3,000 STEPS)
    # =========================================================================
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

    variants = {
        "1. Fixed Temp (tau=1.0)": AnnealedGravimemLM(vocab_size=vocab_size, max_seq_len=block_size, T=3, mode="fixed").to(device),
        "2. Linear Annealing (tau: 2.0 -> 0.5)": AnnealedGravimemLM(vocab_size=vocab_size, max_seq_len=block_size, T=3, mode="linear_anneal").to(device),
        "3. Learned Temperature Schedule": AnnealedGravimemLM(vocab_size=vocab_size, max_seq_len=block_size, T=3, mode="learned_temp").to(device),
        "4. Learned Alpha Schedule (Teleport)": AnnealedGravimemLM(vocab_size=vocab_size, max_seq_len=block_size, T=3, mode="learned_alpha_schedule").to(device),
    }

    results = {}
    for name, model in variants.items():
        print(f"\n>>> Training {name} (3,000 steps)...")
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

        results[name] = val_loss
        if hasattr(model, "log_temps"):
            learned_t = [f"{t:.3f}" for t in torch.exp(model.log_temps).detach().cpu().tolist()]
            print(f"   -> Learned Temperatures: {learned_t}")
        if hasattr(model, "raw_alphas"):
            learned_a = [f"{a:.3f}" for a in torch.sigmoid(model.raw_alphas).squeeze().mean(dim=-1).detach().cpu().tolist()]
            print(f"   -> Learned Alpha Schedule: {learned_a}")

    print("\n" + "=" * 80)
    print("  SUMMARY: TEMPERATURE ANNEALING & DYNAMICS RESULTS")
    print("=" * 80)
    for k, v in results.items():
        print(f"{k:45s} | Val Loss: {v:.4f}")

    return results


@app.local_entrypoint()
def main():
    print("Launching Temperature-Annealed Surfing Suite on Modal GPU...")
    res = run_annealing_suite.remote()
    print("\nAnnealing Suite Finished!")
    print("Results:", res)
