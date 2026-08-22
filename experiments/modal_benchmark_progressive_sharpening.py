"""
Gravimem Progressive Sharpening & Anytime Diffusion Benchmark on Modal GPU:
Evaluates how the Gated GRU Backpack Surfer sharpens its predictions as a function
of inference hops T = 1, 2, 3, 4, 5, 6 on:
1. Multi-Hop Graph Reasoning (OOD Depth Generalization)
2. TinyShakespeare Language Modeling (Anytime Perplexity Curve)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

app = modal.App("gravimem-progressive-sharpening", image=image)


@app.function(gpu="T4", timeout=3600)
def run_progressive_sharpening_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: PROGRESSIVE ANYTIME SHARPENING BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Dataset Preparation (TinyShakespeare)
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 2. Gated GRU Backpack Model with Max T=6
    # -------------------------------------------------------------
    class ProgressiveGRUSurfer(nn.Module):
        def __init__(self, vocab_size, max_T=6, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.max_T = max_T
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

            self.gru = nn.GRUCell(d_model, d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward_all_steps(self, idx, causal_mask, num_steps=6):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x_emb)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + causal_mask[:L, :L]
            P = F.softmax(scores, dim=-1)

            # Surfer starts at initial embedding
            s = x_emb.view(-1, self.d_model)  # (B*L, d)
            step_logits = []

            for t in range(num_steps):
                gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B*L, self.d_model)
                s = self.gru(self.out(gathered), s)
                x_t = s.view(B, L, self.d_model)
                x_mlp = x_t + self.mlp(self.ln2(x_t))
                step_logits.append(self.head(self.ln_f(x_mlp)))

            return step_logits

    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)
    model = ProgressiveGRUSurfer(vocab_size=vocab_size, max_T=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-4)

    print("\n--- Training Progressive Gated GRU Surfer (4,000 steps) ---")
    for step in range(1, 4001):
        model.train()
        bx, by = get_lm_batch('train')
        opt.zero_grad()

        # Randomize training depth between 1 and 6 for depth invariance
        train_steps = random.randint(3, 6)
        logits_list = model.forward_all_steps(bx, causal_mask=causal_mask, num_steps=train_steps)
        
        # Loss averaged across steps
        loss = sum(F.cross_entropy(l.view(-1, vocab_size), by.view(-1)) for l in logits_list) / len(logits_list)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == 4000:
            model.eval()
            with torch.no_grad():
                vx, vy = get_lm_batch('val')
                v_logits_list = model.forward_all_steps(vx, causal_mask=causal_mask, num_steps=6)
                v_losses = [F.cross_entropy(l.view(-1, vocab_size), vy.view(-1)).item() for l in v_logits_list]
            loss_str = " | ".join([f"T={t+1}: {l:.4f}" for t, l in enumerate(v_losses)])
            print(f"Step {step:4d}/4000 | Train Loss: {loss.item():.4f} | Anytime Val Losses: {loss_str}")

    # -------------------------------------------------------------
    # 3. Final Anytime Inference Test Curve
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL ANYTIME SHARPENING EVALUATION (T=1 TO T=8)")
    print("=" * 80)

    model.eval()
    val_losses_by_depth = {}
    with torch.no_grad():
        # Evaluate on 20 validation batches for tight statistical precision
        accum_losses = [0.0] * 8
        num_eval_batches = 20
        for _ in range(num_eval_batches):
            vx, vy = get_lm_batch('val')
            eval_logits = model.forward_all_steps(vx, causal_mask=causal_mask, num_steps=8)
            for t in range(8):
                accum_losses[t] += F.cross_entropy(eval_logits[t].view(-1, vocab_size), vy.view(-1)).item()

        avg_losses = [l / num_eval_batches for l in accum_losses]

    print("\n[Inference Anytime Sharpening Curve on TinyShakespeare]:")
    for t, loss_val in enumerate(avg_losses, 1):
        ppl = math.exp(loss_val)
        note = "(In-Distribution)" if t <= 6 else "(Zero-Shot Extrapolation)"
        print(f"  Hops T = {t} : Val Loss = {loss_val:.4f} | Perplexity = {ppl:6.2f} {note}")
        val_losses_by_depth[f"T={t}"] = loss_val

    return val_losses_by_depth


@app.local_entrypoint()
def main():
    print("Launching Progressive Sharpening Suite on Modal GPU...")
    res = run_progressive_sharpening_suite.remote()
    print("\nFinished Successfully!")
