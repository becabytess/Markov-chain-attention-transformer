"""
Qualitative Interpretability & Thought-Step Analysis of Gravimem Language Model on Modal GPU:
1. Trains Gravimem with Step-by-Step Any-Time Prediction Heads on TinyShakespeare.
2. Evaluates the iterative "Reasoning / Thought-Step" refinement from t=0 to t=4:
   - What does the model predict at Step 0 vs Step 1 vs Step 2 vs Step 3 vs Step 4?
   - How does probability mass M^(t) flow across tokens (clearing local noise -> attending to long-range semantic anchors)?
   - Text generation sampling at each step t=0, t=1, t=2, t=4.
   - Case studies on specific Shakespearean prompts.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
)

app = modal.App("gravimem-qualitative-analysis", image=image)


@app.function(gpu="T4", timeout=3600)
def run_qualitative_analysis():
    import math
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: QUALITATIVE THOUGHT-STEP & DIFFUSION REFINEMENT ANALYSIS")
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
    train_data = data[:n_train]
    val_data = data[n_train:]

    block_size = 128
    max_seq_len = 512
    batch_size = 64

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # -------------------------------------------------------------
    # 2. Gravimem Model with Any-Time Step Readouts
    # -------------------------------------------------------------
    class GravimemAnyTimeLM(nn.Module):
        def __init__(self, vocab_size, max_seq_len=512, d_model=128, n_heads=4, d_mlp=512, T=4):
            super().__init__()
            self.T = T
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)

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

            # Step-dependent teleportation parameters
            self.raw_alphas = nn.Parameter(torch.full((T, n_heads, 1, 1), -1.73))

        def forward_steps(self, idx, causal_mask=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x_emb = self.tok_emb(idx) + self.pos_emb(pos)

            x_norm = self.ln1(x_emb)
            Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if causal_mask is not None:
                scores = scores + causal_mask[:L, :L]
            P = F.softmax(scores, dim=-1)

            I = torch.eye(L, device=idx.device, dtype=x_emb.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)

            step_logits = []
            mass_matrices = [M]

            # Step 0 (Instant token embedding readout)
            H0 = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x0 = x_emb + self.out(H0)
            x0_mlp = x0 + self.mlp(self.ln2(x0))
            step_logits.append(self.head(self.ln_f(x0_mlp)))

            # Steps 1 to T (Markov Mass Settling / Diffusion)
            for t in range(self.T):
                alpha = torch.sigmoid(self.raw_alphas[t])
                M = (1.0 - alpha) * (P @ M) + alpha * I
                mass_matrices.append(M)

                Ht = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                xt = x_emb + self.out(Ht)
                xt_mlp = xt + self.mlp(self.ln2(xt))
                step_logits.append(self.head(self.ln_f(xt_mlp)))

            return step_logits, mass_matrices, P

        def forward(self, idx, causal_mask=None):
            step_logits, _, _ = self.forward_steps(idx, causal_mask)
            return step_logits[-1]

    # -------------------------------------------------------------
    # 3. Train Model on TinyShakespeare (4,000 Steps)
    # -------------------------------------------------------------
    causal_mask = torch.triu(torch.full((block_size, block_size), float('-inf'), device=device), diagonal=1)
    model = GravimemAnyTimeLM(vocab_size=vocab_size, max_seq_len=max_seq_len, T=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized: GravimemAnyTimeLM ({param_count:,} parameters, T=4 steps)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000, eta_min=1e-4)

    print("\n--- Training Model with Any-Time Step Supervision ---")
    for step in range(1, 4001):
        model.train()
        bx, by = get_batch('train')
        opt.zero_grad()

        step_logits, _, _ = model.forward_steps(bx, causal_mask=causal_mask)
        # Deep supervision across all thought steps with increasing weight on final steps
        losses = [F.cross_entropy(l.view(-1, l.size(-1)), by.view(-1)) for l in step_logits]
        total_loss = 0.1 * losses[0] + 0.15 * losses[1] + 0.2 * losses[2] + 0.25 * losses[3] + 0.3 * losses[4]
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % 1000 == 0 or step == 4000:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch('val')
                v_step_logits, _, _ = model.forward_steps(vx, causal_mask=causal_mask)
                v_losses = [F.cross_entropy(l.view(-1, l.size(-1)), vy.view(-1)).item() for l in v_step_logits]
            print(f"Step {step:4d}/4000 | Train Loss: {total_loss.item():.4f} | Val Losses per Step: " +
                  " | ".join([f"t={t}: {loss:.4f}" for t, loss in enumerate(v_losses)]))

    # -------------------------------------------------------------
    # 4. In-Depth Qualitative Thought-Step Refinement Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  QUALITATIVE CASE STUDIES: STEP-BY-STEP REASONING & DIFFUSION DYNAMICS")
    print("=" * 80)

    test_prompts = [
        # Case 1: Dialogue Speaker Attribution & Stylistic Consistency
        "ROMEO:\nLady, by yonder blessed moon I vow,\nThat tips with silver all these fruit-tree tops,--\nJULIET:\nO, swear not by the ",
        
        # Case 2: Long-Range Syntactic & Semantic Subject-Verb Resolution
        "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to ",
        
        # Case 3: Archaic Rhyme & Poetic Structure
        "KING RICHARD III:\nGive me another horse: bind up my wounds.\nHave mercy, Jesu!--Soft! I did but dream.\nO cowardly conscience, how dost thou afflict "
    ]

    analysis_results = []

    model.eval()
    for case_idx, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*70}\nCASE STUDY {case_idx}:")
        print(f"Prompt Text:\n\"{prompt}\"")
        print(f"{'='*70}")

        tokens = torch.tensor([char2idx.get(c, 0) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
        prompt_len = tokens.size(1)
        c_mask = torch.triu(torch.full((prompt_len, prompt_len), float('-inf'), device=device), diagonal=1)

        with torch.no_grad():
            step_logits, mass_matrices, P = model.forward_steps(tokens, causal_mask=c_mask)

        # Inspect next-character predictions at the very last token position across steps t=0...4
        print("\n--- NEXT-TOKEN PREDICTION EVOLUTION ACROSS THOUGHT STEPS (t=0 -> t=4) ---")
        step_evolution = []
        for t in range(len(step_logits)):
            last_token_logits = step_logits[t][0, -1, :]  # Shape: (vocab_size,)
            probs = F.softmax(last_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=5)
            
            topk_chars = []
            for prob, idx in zip(topk_probs, topk_indices):
                char = idx2char[idx.item()]
                repr_char = repr(char) if char not in [' ', '\n'] else ('<SPACE>' if char == ' ' else '<NEWLINE>')
                topk_chars.append(f"{repr_char} ({prob.item()*100:.1f}%)")
            
            summary_str = f"Step t={t} | Top Candidates: " + ", ".join(topk_chars)
            print(summary_str)
            step_evolution.append(summary_str)

        # Inspect Markov Mass Matrix M^(t) at the last query position
        print("\n--- ATTENTION & MARKOV MASS FLOW TO PREVIOUS TOKENS ---")
        print("Examining top tokens receiving mass at query position at step t=0, t=1, t=2, t=4:")
        
        # Average across heads
        for t in [0, 1, 2, 4]:
            M_t = mass_matrices[t][0].mean(dim=0)  # Shape: (L, L)
            query_mass = M_t[-1, :]  # Mass from last token to all previous tokens
            top_mass_vals, top_mass_pos = torch.topk(query_mass, k=min(6, prompt_len))
            
            print(f"\nStep t={t} (Markov Diffusion Depth = {t}):")
            for val, pos in zip(top_mass_vals, top_mass_pos):
                token_char = prompt[pos.item()]
                token_disp = repr(token_char) if token_char not in [' ', '\n'] else ('<SPACE>' if token_char == ' ' else '<NEWLINE>')
                snippet_start = max(0, pos.item() - 10)
                snippet_end = min(len(prompt), pos.item() + 10)
                snippet = prompt[snippet_start:snippet_end].replace('\n', '\\n')
                print(f"  Pos {pos.item():3d} [{token_disp:9s}] Mass: {val.item()*100:5.1f}% | Context: \"...{snippet}...\"")

        # Measure Mass Entropy / Dispersion across steps
        entropies = []
        for t in range(len(mass_matrices)):
            M_t = mass_matrices[t][0].mean(dim=0)[-1, :]
            M_clamped = M_t.clamp(min=1e-9)
            entropy = -(M_clamped * torch.log(M_clamped)).sum().item()
            entropies.append(f"t={t}: {entropy:.3f}")
        print(f"\nMass Entropy (Information Spread) at Query Token: {' -> '.join(entropies)}")

        # ---------------------------------------------------------
        # Text Generation from each step t (Autoregressive Rollout)
        # ---------------------------------------------------------
        print("\n--- TEXT CONTINUATION SAMPLES GENERATED AT EACH THOUGHT DEPTH ---")
        gen_len = 60
        for t in [0, 1, 2, 4]:
            cur_tokens = tokens.clone()
            for _ in range(gen_len):
                if cur_tokens.size(1) >= max_seq_len:
                    break
                c_mask_gen = torch.triu(torch.full((cur_tokens.size(1), cur_tokens.size(1)), float('-inf'), device=device), diagonal=1)
                with torch.no_grad():
                    logits_list, _, _ = model.forward_steps(cur_tokens, causal_mask=c_mask_gen)
                    next_token_logits = logits_list[t][0, -1, :] / 0.8
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                    cur_tokens = torch.cat([cur_tokens, next_token.unsqueeze(0)], dim=1)
            
            gen_text = "".join([idx2char[i.item()] for i in cur_tokens[0, prompt_len:]])
            print(f"\n[Generated with Fixed Depth t={t}]:\n\"{gen_text}\"")

        analysis_results.append({
            "prompt": prompt,
            "evolution": step_evolution,
            "entropies": entropies
        })

    # Learned alpha schedule
    learned_alphas = torch.sigmoid(model.raw_alphas).squeeze().mean(dim=-1).detach().cpu().tolist()
    print("\n" + "=" * 80)
    print("  LEARNED TELEPORTATION SCHEDULE (IDENTITY RETENTION PRIOR)")
    print("=" * 80)
    for t, a in enumerate(learned_alphas):
        print(f"Step t={t} -> t={t+1}: Identity Retention alpha={a:.4f} | Relational Diffusion Flow (1-alpha)={1-a:.4f}")

    return {
        "case_studies": analysis_results,
        "learned_alphas": learned_alphas
    }


@app.local_entrypoint()
def main():
    print("Launching Gravimem Qualitative Interpretability Suite on Modal GPU...")
    res = run_qualitative_analysis.remote()
    print("\nQualitative Analysis Finished Successfully!")
