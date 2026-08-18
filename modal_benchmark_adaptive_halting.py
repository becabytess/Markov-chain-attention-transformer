"""
Gravimem: Adaptive Early-Exit & Compute Halting Benchmark
Experiment: "After training with mixed T, can the model identify when another hop is no longer worth the compute?"

Evaluates:
1. Confidence-based Halting (Max probability threshold tau)
2. State Velocity Settling Halting (||s^(t) - s^(t-1)|| / ||s^(t)|| <= epsilon)
3. Entropy Halting (H(p) <= tau_h)
4. Prediction Stability Halting (Top-1 argmax invariance)
5. Token Difficulty Profiling (Which tokens exit early vs late)
6. Compute Savings vs Perplexity Pareto Frontier
"""

import modal
import os

app = modal.App("gravimem-adaptive-halting")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "numpy",
    "requests"
)

@app.function(
    image=image,
    gpu="T4",
    timeout=600
)
def run_adaptive_halting_study():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  GRAVIMEM: ADAPTIVE EARLY-EXIT & COMPUTE HALTING BENCHMARK")
    print("=" * 80)
    print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Dataset Setup (TinyShakespeare)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        text = response.read().decode('utf-8')

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([char_to_ix[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split="train", batch_size=32, seq_len=128):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len - 1, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Optimized Fused Positional Jump Engine with Step-by-Step Trajectory
    class FusedPositionalJumpSurfer(nn.Module):
        def __init__(self, d_model=128, offsets=None, max_seq_len=256):
            super().__init__()
            if offsets is None:
                offsets = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127]
            self.offsets = sorted(list(set(offsets)))
            self.K = len(self.offsets)
            self.d_model = d_model
            self.max_seq_len = max_seq_len

            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.policy_net = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, self.K)
            )
            self.w_ih = nn.Linear(d_model, 3 * d_model)
            self.w_hh = nn.Linear(d_model, 3 * d_model)

            seq_indices = torch.arange(max_seq_len).unsqueeze(1)
            offset_tensor = torch.tensor(self.offsets).unsqueeze(0)
            target_indices = torch.clamp(seq_indices - offset_tensor, min=0)
            valid_mask = (seq_indices >= offset_tensor).float()

            self.register_buffer("target_indices", target_indices, persistent=False)
            self.register_buffer("valid_mask", valid_mask, persistent=False)

        def _step(self, s, V, B, L):
            logits = self.policy_net(s)
            mask = self.valid_mask[:L].unsqueeze(0)
            logits = logits + (1.0 - mask) * -1e9
            pi = F.softmax(logits, dim=-1)

            t_indices = self.target_indices[:L]
            flat_indices = t_indices.reshape(-1)
            V_gathered = V[:, flat_indices, :].view(B, L, self.K, self.d_model)
            v_context = torch.sum(pi.unsqueeze(-1) * V_gathered, dim=2)

            gates_i = self.w_ih(v_context)
            gates_h = self.w_hh(s)
            gates = gates_i + gates_h

            r_gate, z_gate, n_in = gates.chunk(3, dim=-1)
            r = torch.sigmoid(r_gate)
            z = torch.sigmoid(z_gate)
            c_gate = self.w_hh(r * s)
            c_in = gates_i.chunk(3, dim=-1)[2] + c_gate.chunk(3, dim=-1)[2]
            n = torch.tanh(c_in)
            s_next = (1.0 - z) * n + z * s
            return s_next, pi

        def forward(self, x, T=4, return_trajectory=False):
            B, L, D = x.shape
            V = self.v_proj(x)
            s = x
            trajectory = [s] if return_trajectory else None

            for _ in range(T):
                s, _ = self._step(s, V, B, L)
                if return_trajectory:
                    trajectory.append(s)

            if return_trajectory:
                return s, trajectory
            return s

    class GravimemLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, max_seq_len=256, offsets=None):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.surfer = FusedPositionalJumpSurfer(d_model=d_model, offsets=offsets, max_seq_len=max_seq_len)
            self.norm = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model)
            )
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, idx, T=4, return_all_steps=False):
            B, L = idx.shape
            pos = torch.arange(L, device=idx.device).unsqueeze(0)
            x = self.token_emb(idx) + self.pos_emb(pos)

            if return_all_steps:
                _, trajectory = self.surfer(x, T=T, return_trajectory=True)
                step_logits = []
                for s in trajectory[1:]: # skip step 0 (initial embedding)
                    h = s + self.mlp(self.norm(s))
                    step_logits.append(self.head(h))
                return step_logits, trajectory[1:]
            else:
                s = self.surfer(x, T=T)
                h = s + self.mlp(self.norm(s))
                return self.head(h)

    # 3. Train Model with Mixed-T (T in [1, 6])
    print("\n--- Training Gravimem with Mixed-T Strategy (T in [1, 6]) ---")
    torch.manual_seed(42)
    model = GravimemLM(vocab_size=vocab_size, d_model=128, max_seq_len=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    t0 = time.time()
    for step in range(1, 2001):
        model.train()
        x, y = get_batch("train", batch_size=32, seq_len=128)
        # Sample T uniformly from 1 to 6
        T_curr = int(torch.randint(1, 7, (1,)).item())
        logits = model(x, T=T_curr)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"  Step {step:4d}/2000 | Train Loss: {loss.item():.4f} | Elapsed: {time.time()-t0:.1f}s")

    # 4. Evaluation Function for Early-Exiting
    @torch.no_grad()
    def evaluate_halting(criterion_type="confidence", threshold=0.90, max_T=6):
        """
        Evaluates dynamic token-level early exiting on validation set.
        Criterion types:
          - 'confidence': exit when max softmax probability >= threshold
          - 'entropy': exit when Shannon entropy <= threshold
          - 'velocity': exit when ||s^(t) - s^(t-1)|| / ||s^(t)|| <= threshold
          - 'stability': exit when top-1 prediction is stable for 2 consecutive steps
          - 'fixed': fixed T hops
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        total_hops_used = 0
        hop_histogram = {t: 0 for t in range(1, max_T + 1)}

        val_batches = 20
        for _ in range(val_batches):
            x, y = get_batch("val", batch_size=32, seq_len=128)
            B, L = x.shape
            step_logits, trajectory = model(x, T=max_T, return_all_steps=True)
            # step_logits: list of T tensors, each [B, L, Vocab]
            # trajectory: list of T hidden states, each [B, L, D]

            # Track which tokens have exited
            exited = torch.zeros(B, L, dtype=torch.bool, device=device)
            final_logits = torch.zeros_like(step_logits[0])
            token_hops = torch.full((B, L), max_T, dtype=torch.long, device=device)

            for t_idx in range(max_T):
                t_val = t_idx + 1
                curr_logits = step_logits[t_idx]
                curr_probs = F.softmax(curr_logits, dim=-1)

                if criterion_type == "confidence":
                    max_probs, _ = torch.max(curr_probs, dim=-1)
                    halt_cond = max_probs >= threshold
                elif criterion_type == "entropy":
                    log_probs = F.log_softmax(curr_logits, dim=-1)
                    entropy = -torch.sum(curr_probs * log_probs, dim=-1)
                    halt_cond = entropy <= threshold
                elif criterion_type == "velocity":
                    if t_idx == 0:
                        halt_cond = torch.zeros(B, L, dtype=torch.bool, device=device)
                    else:
                        prev_s = trajectory[t_idx - 1]
                        curr_s = trajectory[t_idx]
                        diff_norm = torch.norm(curr_s - prev_s, dim=-1)
                        s_norm = torch.norm(curr_s, dim=-1) + 1e-6
                        rel_diff = diff_norm / s_norm
                        halt_cond = rel_diff <= threshold
                elif criterion_type == "stability":
                    if t_idx == 0:
                        halt_cond = torch.zeros(B, L, dtype=torch.bool, device=device)
                    else:
                        prev_top1 = torch.argmax(step_logits[t_idx - 1], dim=-1)
                        curr_top1 = torch.argmax(curr_logits, dim=-1)
                        halt_cond = (prev_top1 == curr_top1)
                elif criterion_type == "fixed":
                    halt_cond = torch.ones(B, L, dtype=torch.bool, device=device) if t_val == threshold else torch.zeros(B, L, dtype=torch.bool, device=device)
                else:
                    raise ValueError(f"Unknown criterion {criterion_type}")

                # Mask for newly halted tokens (or last step for non-halted tokens)
                newly_halted = (halt_cond & ~exited) | ((t_val == max_T) & ~exited)
                final_logits[newly_halted] = curr_logits[newly_halted]
                token_hops[newly_halted] = t_val
                exited = exited | halt_cond

            loss = F.cross_entropy(final_logits.view(-1, vocab_size), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += (B * L)
            total_hops_used += token_hops.sum().item()

            for t in range(1, max_T + 1):
                hop_histogram[t] += (token_hops == t).sum().item()

        avg_loss = total_loss / total_tokens
        avg_ppl = math.exp(avg_loss)
        avg_hops = total_hops_used / total_tokens
        compute_savings = (1.0 - (avg_hops / max_T)) * 100.0
        hop_distribution = {t: f"{(hop_histogram[t] / total_tokens)*100.0:.1f}%" for t in range(1, max_T + 1)}

        return {
            "val_loss": avg_loss,
            "ppl": avg_ppl,
            "avg_hops": avg_hops,
            "compute_savings": compute_savings,
            "distribution": hop_distribution
        }

    # 5. Run Comprehensive Halting Experiments
    print("\n" + "=" * 80)
    print("  EXPERIMENT A: PARETO FRONTIER - CONFIDENCE-BASED EARLY EXITING")
    print("=" * 80)
    print(f"{'Threshold (tau)':<16} | {'Avg Hops (T)':<12} | {'Savings (%)':<12} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 75)

    confidence_results = []
    for tau in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        res = evaluate_halting("confidence", threshold=tau, max_T=6)
        confidence_results.append((tau, res))
        print(f"tau = {tau:<10.2f} | {res['avg_hops']:<12.2f} | {res['compute_savings']:<11.1f}% | {res['val_loss']:<10.4f} | {res['ppl']:<10.2f}")

    print("\n" + "=" * 80)
    print("  EXPERIMENT B: DYNAMICAL STATE VELOCITY EARLY EXITING (||Delta s|| / ||s|| <= eps)")
    print("=" * 80)
    print(f"{'Epsilon (eps)':<16} | {'Avg Hops (T)':<12} | {'Savings (%)':<12} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 75)

    velocity_results = []
    for eps in [0.03, 0.05, 0.08, 0.12, 0.20]:
        res = evaluate_halting("velocity", threshold=eps, max_T=6)
        velocity_results.append((eps, res))
        print(f"eps = {eps:<10.2f} | {res['avg_hops']:<12.2f} | {res['compute_savings']:<11.1f}% | {res['val_loss']:<10.4f} | {res['ppl']:<10.2f}")

    print("\n" + "=" * 80)
    print("  EXPERIMENT C: PREDICTION INVARIANCE / TOP-1 STABILITY EXITING")
    print("=" * 80)
    res_stab = evaluate_halting("stability", threshold=0, max_T=6)
    print(f"Top-1 Invariance | Avg Hops: {res_stab['avg_hops']:.2f} | Savings: {res_stab['compute_savings']:.1f}% | Val Loss: {res_stab['val_loss']:.4f} | PPL: {res_stab['ppl']:.2f}")
    print(f"Hop Distribution: {res_stab['distribution']}")

    print("\n" + "=" * 80)
    print("  EXPERIMENT D: COMPARISON AGAINST FIXED HOP BASELINES (T=1, 2, 4, 6)")
    print("=" * 80)
    print(f"{'Policy':<25} | {'Avg Hops':<10} | {'Savings (%)':<12} | {'Val Loss':<10} | {'Perplexity':<10}")
    print("-" * 75)
    for fixed_t in [1, 2, 4, 6]:
        res = evaluate_halting("fixed", threshold=fixed_t, max_T=6)
        print(f"Fixed T = {fixed_t:<15} | {res['avg_hops']:<10.1f} | {res['compute_savings']:<11.1f}% | {res['val_loss']:<10.4f} | {res['ppl']:<10.2f}")

    # 6. Qualitative Token Case Study: Which Tokens Exit Early vs Late?
    print("\n" + "=" * 80)
    print("  EXPERIMENT E: TOKEN PROFILING - EASY TOKENS (T=1) vs HARD TOKENS (T=5..6)")
    print("=" * 80)
    x_sample, _ = get_batch("val", batch_size=1, seq_len=80)
    step_logits, _ = model(x_sample, T=6, return_all_steps=True)

    # Let's inspect confidence threshold tau=0.85
    tau = 0.85
    sample_chars = [ix_to_char[idx.item()] for idx in x_sample[0]]
    token_exit_hops = []

    for pos in range(len(sample_chars)):
        exit_t = 6
        for t_idx in range(6):
            p = F.softmax(step_logits[t_idx][0, pos], dim=-1)
            if p.max().item() >= tau:
                exit_t = t_idx + 1
                break
        token_exit_hops.append(exit_t)

    print("\nSample Character Stream Annotated with Required Thought Hops [Char: T_hops]:")
    formatted_stream = "".join([f"{repr(c)[1:-1]}(T={h}) " for c, h in zip(sample_chars[:40], token_exit_hops[:40])])
    print(formatted_stream)

    t1_tokens = [c for c, h in zip(sample_chars, token_exit_hops) if h == 1]
    t_deep_tokens = [c for c, h in zip(sample_chars, token_exit_hops) if h >= 4]

    print(f"\nTokens Exiting at T=1 (Instant Exit, ~0 compute): {repr(''.join(t1_tokens[:25]))}")
    print(f"Tokens Exiting at T>=4 (Deliberate Thinking Required): {repr(''.join(t_deep_tokens[:25]))}")

    print("\n" + "=" * 80)
    print("  ALL ADAPTIVE HALTING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return {
        "confidence_results": confidence_results,
        "velocity_results": velocity_results,
        "stability_result": res_stab
    }

@app.local_entrypoint()
def main():
    print("Launching Gravimem Adaptive Early-Exit Benchmark on Modal GPU...")
    res = run_adaptive_halting_study.remote()
    print("\nFinished Successfully!")
