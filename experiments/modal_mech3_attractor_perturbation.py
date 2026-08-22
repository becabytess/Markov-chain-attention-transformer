"""
Mechanistic Experiment 3: Attractor Basins & Noise Perturbation Recovery
Tests if Gravimem acts as a dynamical attractor that naturally damps out state perturbations
and recovers its clean unperturbed trajectory as hops progress.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-mech3-attractor-perturb", image=image)


@app.function(gpu="T4", timeout=3600)
def run_attractor_perturbation():
    import math
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  MECHANISTIC EXP 3: DYNAMICAL ATTRACTOR BASIN & PERTURBATION RECOVERY")
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
    d_mlp = 512
    num_steps = 1500
    max_T = 6

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 255]

    class AttractorGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, d_model=128, d_mlp=512, max_len=256):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.K = len(jump_offsets)
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

        def forward_with_perturbation(self, idx, T=6, perturb_step=1, noise_std=0.0):
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
            states = []
            logits_list = []

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

                if step == perturb_step and noise_std > 0.0:
                    noise = torch.randn_like(s) * noise_std
                    s = s + noise

                states.append(s)
                h_out = s + self.mlp(self.ln2(s))
                logits_list.append(self.head(self.ln_f(h_out)))

            return states, logits_list

    model = AttractorGravimem(vocab_size, jump_offsets).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    print("\n---> Training Standard Gravimem Model (Mixed T in [1, 6])...")
    for step in range(1, num_steps + 1):
        model.train()
        x, y = get_batch("train")
        T_curr = torch.randint(1, 7, (1,)).item()
        opt.zero_grad()
        _, logits_list = model.forward_with_perturbation(x, T=T_curr, noise_std=0.0)
        loss = F.cross_entropy(logits_list[-1].view(-1, vocab_size), y.view(-1))
        loss.backward()
        opt.step()
    print("     Model training complete!")

    print("\n---> Evaluating Attractor Perturbation Recovery Dynamics...")
    model.eval()
    noise_levels = [0.0, 0.1, 0.25, 0.5, 1.0]

    # Structure: noise -> step -> {error_ratio, cos_sim_clean, ppl}
    recovery_stats = {
        noise: {t: {"error_norm": 0.0, "clean_cos": 0.0, "ppl_sum": 0.0, "samples": 0} for t in range(1, max_T + 1)}
        for noise in noise_levels
    }

    with torch.no_grad():
        for _ in range(50):
            x_val, y_val = get_batch("val")

            # Reference clean trajectory
            clean_states, clean_logits = model.forward_with_perturbation(x_val, T=max_T, noise_std=0.0)
            clean_final = clean_states[-1]

            for noise in noise_levels:
                pert_states, pert_logits = model.forward_with_perturbation(x_val, T=max_T, perturb_step=1, noise_std=noise)

                initial_error = (pert_states[0] - clean_states[0]).norm(dim=-1).mean().item() + 1e-8

                for t in range(1, max_T + 1):
                    idx = t - 1
                    s_t = pert_states[idx]
                    curr_error = (s_t - clean_states[idx]).norm(dim=-1).mean().item()
                    cos_to_clean = F.cosine_similarity(s_t, clean_final, dim=-1).mean().item()

                    loss_t = F.cross_entropy(pert_logits[idx].view(-1, vocab_size), y_val.view(-1)).item()
                    ppl_t = math.exp(loss_t)

                    recovery_stats[noise][t]["error_norm"] += (curr_error / initial_error if noise > 0 else 0.0)
                    recovery_stats[noise][t]["clean_cos"] += cos_to_clean
                    recovery_stats[noise][t]["ppl_sum"] += ppl_t
                    recovery_stats[noise][t]["samples"] += 1

    print("\n" + "=" * 80)
    print("  ATTRACTOR PERTURBATION DYNAMICS: STATE RECOVERY & ERROR DECAY")
    print("=" * 80)
    print("Noise Level (sigma) | Step 1 (Perturbed) | Step 2 (1 Hop) | Step 4 (3 Hops) | Step 6 (5 Hops)")
    print("-" * 80)

    for noise in noise_levels:
        row = f"sigma = {noise:4.2f} (PPL)   |"
        for t in [1, 2, 4, 6]:
            avg_ppl = recovery_stats[noise][t]["ppl_sum"] / recovery_stats[noise][t]["samples"]
            row += f"    {avg_ppl:6.2f}     |"
        print(row)

    print("-" * 80)
    print("Noise Level (sigma) | Error Ratio t=1   | Error Ratio t=2| Error Ratio t=4 | Error Ratio t=6")
    print("-" * 80)
    for noise in noise_levels:
        if noise == 0.0:
            continue
        row = f"sigma = {noise:4.2f} (Error) |"
        for t in [1, 2, 4, 6]:
            avg_err = recovery_stats[noise][t]["error_norm"] / recovery_stats[noise][t]["samples"]
            row += f"     {avg_err:5.2f}x    |"
        print(row)
    print("=" * 80)

    return recovery_stats


@app.local_entrypoint()
def main():
    print("Launching Mechanistic Experiment 3 on dedicated Modal GPU...")
    res = run_attractor_perturbation.remote()
    print("Experiment 3 complete!")
