"""
Gravimem Positional Jump Surfer Benchmark on Modal GPU:
Evaluates learned positional jumping where each surfer dynamically chooses
which relative positional offsets to jump to at each step:
1. Standard Dense Content-Content Attention (O(L^2) Baseline)
2. Learned Positional Jump Surfer (O(L * K) with K candidate jump offsets)
3. Dynamic Content-Conditioned Positional Surfer (Surfer state decides next jump)
4. Pure Relative Positional Graph Surfer

Tests on:
- Multi-Hop Distance Navigation (Hop chains across long contexts)
- TinyShakespeare Autoregressive Language Modeling (Anytime Perplexity Curve T=1..6)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

app = modal.App("gravimem-positional-jumping", image=image)


@app.function(gpu="T4", timeout=3600)
def run_positional_jump_suite():
    import math
    import random
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: LEARNED POSITIONAL JUMP SURFING BENCHMARK")
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
    # MODEL 1: Standard Dense Attention Baseline (O(L^2))
    # -------------------------------------------------------------
    class DenseLMBaseline(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)
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
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
            self.d_k = d_model // n_heads
            self.n_heads = n_heads

        def forward(self, idx, causal_mask):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x_norm = self.ln1(x)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + causal_mask[:L, :L]
            attn = F.softmax(scores, dim=-1)
            H = (attn @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out(H)
            x = x + self.mlp(self.ln2(x))
            return self.head(self.ln_f(x))

    # -------------------------------------------------------------
    # MODEL 2: Learned Positional Jump Surfer (O(L * K))
    # Jump Menu: K relative offsets (e.g. 0, 1, 2, 4, 8, 16, 32, 64, 127)
    # -------------------------------------------------------------
    class PositionalJumpSurfer(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.num_jumps = len(jump_offsets)
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            # Surfer predicts a probability distribution over the K jump choices
            self.jump_policy = nn.Linear(d_model, self.num_jumps)
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

        def forward(self, idx, return_all_steps=False):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            # Pre-project value representations
            V = self.v_proj(self.ln1(x_emb))  # (B, L, d_model)

            # Create index mapping tensor for all K jumps: (L, K)
            # target_idx[i, k] = clamp(i - jump_offsets[k], 0, L-1)
            target_indices = torch.zeros((L, self.num_jumps), dtype=torch.long, device=idx.device)
            valid_jump_mask = torch.zeros((L, self.num_jumps), dtype=torch.bool, device=idx.device)
            for i in range(L):
                for k, offset in enumerate(self.jump_offsets):
                    target_pos = i - offset
                    if target_pos >= 0:
                        target_indices[i, k] = target_pos
                        valid_jump_mask[i, k] = True
                    else:
                        target_indices[i, k] = 0  # Fallback
                        valid_jump_mask[i, k] = False

            # Initial surfer backpack state
            s = x_emb.view(-1, self.d_model)  # (B*L, d_model)
            step_outputs = []

            for t in range(self.T):
                # Surfer at each position looks at its backpack and predicts jump distribution
                jump_logits = self.jump_policy(s).view(B, L, self.num_jumps)
                jump_logits = jump_logits.masked_fill(~valid_jump_mask.unsqueeze(0), float('-inf'))
                jump_probs = F.softmax(jump_logits, dim=-1)  # (B, L, K)

                # Gather V from the K jump positions: (B, L, K, d_model)
                # Expand target_indices for batch gather
                flat_targets = target_indices.unsqueeze(0).expand(B, L, self.num_jumps)  # (B, L, K)
                gathered_V_candidates = torch.gather(
                    V.unsqueeze(1).expand(B, L, L, self.d_model),
                    dim=2,
                    index=flat_targets.unsqueeze(-1).expand(B, L, self.num_jumps, self.d_model)
                )  # (B, L, K, d_model)

                # Weighted sum over the K candidate jumps
                gathered_V = torch.sum(jump_probs.unsqueeze(-1) * gathered_V_candidates, dim=2)  # (B, L, d_model)

                # Pass through output projection and update backpack
                gathered_proj = self.out_proj(gathered_V).view(-1, self.d_model)
                s = self.gru(gathered_proj, s)

                if return_all_steps:
                    x_t = s.view(B, L, self.d_model)
                    x_mlp = x_t + self.mlp(self.ln2(x_t))
                    step_outputs.append(self.head(self.ln_f(x_mlp)))

            if return_all_steps:
                return step_outputs

            x_final = s.view(B, L, self.d_model)
            x_final = x_final + self.mlp(self.ln2(x_final))
            return self.head(self.ln_f(x_final))

    # -------------------------------------------------------------
    # Jump Configurations to Test:
    # -------------------------------------------------------------
    # Config A: Fine-grained Fibonacci / Exponential Jump Menu (12 jumps)
    # [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127]
    jump_menu_fib = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127]

    # Config B: Compact Dyadic Powers-of-2 Menu (8 jumps)
    # [0, 1, 2, 4, 8, 16, 32, 64]
    jump_menu_dyadic = [0, 1, 2, 4, 8, 16, 32, 64]

    # Config C: Tiny Local + Far Jump Menu (5 jumps)
    # [0, 1, 2, 16, 64]
    jump_menu_tiny = [0, 1, 2, 16, 64]

    models_to_test = [
        ("1. Standard Dense Attention (O(L^2) Baseline)", "dense", None),
        ("2. Positional Jump Surfer (Dyadic 8 Jumps)", "jump", jump_menu_dyadic),
        ("3. Positional Jump Surfer (Fibonacci 12 Jumps)", "jump", jump_menu_fib),
        ("4. Positional Jump Surfer (Tiny 5 Jumps)", "jump", jump_menu_tiny),
    ]

    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)

    results = {}
    anytime_curves = {}

    for name, m_type, j_menu in models_to_test:
        print(f"\n" + "=" * 80)
        print(f"  TRAINING: {name}")
        print("=" * 80)

        if m_type == "dense":
            model = DenseLMBaseline(vocab_size=vocab_size).to(device)
        else:
            model = PositionalJumpSurfer(vocab_size=vocab_size, jump_offsets=j_menu, T=4).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)

        for step in range(1, 3001):
            model.train()
            bx, by = get_lm_batch('train')
            opt.zero_grad()
            if m_type == "dense":
                logits = model(bx, causal_mask=causal_mask)
            else:
                logits = model(bx)

            loss = F.cross_entropy(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        # Evaluate on validation split
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for _ in range(20):
                vx, vy = get_lm_batch('val')
                if m_type == "dense":
                    v_logits = model(vx, causal_mask=causal_mask)
                else:
                    v_logits = model(vx)
                eval_losses.append(F.cross_entropy(v_logits.view(-1, vocab_size), vy.view(-1)).item())
            final_val_loss = sum(eval_losses) / len(eval_losses)

        print(f"--> {name} Final Val Loss: {final_val_loss:.4f} (Perplexity: {math.exp(final_val_loss):.2f})")
        results[name] = final_val_loss

        # If it's a jump surfer, evaluate its anytime sharpening curve across hops T=1..6
        if m_type == "jump":
            with torch.no_grad():
                hop_losses = [0.0] * 6
                for _ in range(20):
                    vx, vy = get_lm_batch('val')
                    # Temporarily unroll to T=6
                    model.T = 6
                    step_logits = model(vx, return_all_steps=True)
                    for t in range(6):
                        hop_losses[t] += F.cross_entropy(step_logits[t].view(-1, vocab_size), vy.view(-1)).item()
                avg_hop_losses = [l / 20 for l in hop_losses]
                anytime_curves[name] = avg_hop_losses
                print(f"    Anytime Hop Progression:")
                for t, l in enumerate(avg_hop_losses, 1):
                    print(f"      Hop T={t}: Val Loss = {l:.4f} | PPL = {math.exp(l):.2f}")

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  SUMMARY: LEARNED POSITIONAL JUMP SURFING BENCHMARK RESULTS")
    print("=" * 80)
    print("\n[Final Validation Loss Comparison]:")
    for name, loss_val in results.items():
        ppl = math.exp(loss_val)
        print(f"  {name:55s} : Loss = {loss_val:.4f} | PPL = {ppl:6.2f}")

    print("\n[Anytime Thought Sharpening Curves by Hop]:")
    for name, curve in anytime_curves.items():
        print(f"\n  {name}:")
        for t, l in enumerate(curve, 1):
            print(f"    Hop T={t}: Loss = {l:.4f} (PPL: {math.exp(l):.2f})")

    return {
        "results": results,
        "anytime_curves": anytime_curves
    }


@app.local_entrypoint()
def main():
    print("Launching Positional Jump Surfer Suite on Modal GPU...")
    res = run_positional_jump_suite.remote()
    print("\nFinished Successfully!")
