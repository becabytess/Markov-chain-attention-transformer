"""
Nightmare Benchmark 2 Follow-up: Gravimem with Deep Hops (T=10) on Dyck-4 Grammar
Tests if scaling from T=4 to T=10 hops allows 1-Layer Gravimem to resolve deep nested stack reduction.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-dyck-deep-hops", image=image)


@app.function(gpu="T4", timeout=3600)
def run_dyck_deep_hops():
    import math
    import random
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  NIGHTMARE BENCHMARK 2 FOLLOW-UP: GRAVIMEM WITH DEEP HOPS (T=10)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

    vocab_size = 16
    seq_len = 256
    batch_size = 32
    d_model = 128
    d_mlp = 512
    num_steps = 1500
    hops_T = 10

    open_to_close = {1: 2, 3: 4, 5: 6, 7: 8}
    open_brackets = [1, 3, 5, 7]

    def generate_dyck_sequence(target_len=256, max_depth=30):
        tokens = []
        stack = []
        depth_at_pos = []

        while len(tokens) < target_len - 1:
            curr_depth = len(stack)
            p_close = 0.0 if curr_depth == 0 else (0.45 if curr_depth < max_depth else 0.90)

            if len(tokens) + curr_depth >= target_len - 1:
                p_close = 1.0

            if random.random() < p_close and curr_depth > 0:
                last_open = stack.pop()
                expected_close = open_to_close[last_open]
                tokens.append(expected_close)
                depth_at_pos.append(curr_depth)
            else:
                b = random.choice(open_brackets)
                stack.append(b)
                tokens.append(b)
                depth_at_pos.append(len(stack))

        while stack and len(tokens) < target_len:
            last_open = stack.pop()
            tokens.append(open_to_close[last_open])
            depth_at_pos.append(len(stack) + 1)

        while len(tokens) < target_len:
            tokens.append(0)
            depth_at_pos.append(0)

        tokens = tokens[:target_len]
        depth_at_pos = depth_at_pos[:target_len]
        return tokens, depth_at_pos

    def generate_dyck_batch(batch_size, seq_len=256):
        x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        depths = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        for b in range(batch_size):
            tokens, depth_list = generate_dyck_sequence(seq_len, max_depth=30)
            x[b] = torch.tensor(tokens, device=device)
            depths[b] = torch.tensor(depth_list, device=device)

            for t in range(seq_len - 1):
                next_tok = tokens[t + 1]
                if next_tok in [2, 4, 6, 8]:
                    y[b, t] = next_tok

        return x, y, depths

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 255]

    class DyckGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=10, d_model=128, d_mlp=512, max_len=256):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.K = len(jump_offsets)
            self.T = T
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

        def forward(self, idx, T=None):
            if T is None:
                T = self.T
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
            for step in range(T):
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

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    model = DyckGravimem(vocab_size, jump_offsets, T=hops_T, d_model=d_model, d_mlp=d_mlp).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n---> Model: Gravimem (1 Layer, T={hops_T} Hops)")
    print(f"     Parameter Count: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        x, y, _ = generate_dyck_batch(batch_size, seq_len)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0 or step == num_steps:
            mask = (y != -100)
            preds = logits.argmax(dim=-1)
            acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
            print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Closing Acc: {acc:.1f}% | Elapsed: {time.time()-t0:.1f}s")

    model.eval()
    tier_correct = {"Overall": 0, "Depth 1-5": 0, "Depth 6-15": 0, "Depth 16-30": 0}
    tier_total = {"Overall": 0, "Depth 1-5": 0, "Depth 6-15": 0, "Depth 16-30": 0}

    with torch.no_grad():
        for _ in range(50):
            x_val, y_val, depth_val = generate_dyck_batch(32, seq_len)
            logits = model(x_val)
            mask = (y_val != -100)
            preds = logits.argmax(dim=-1)

            is_correct = (preds == y_val) & mask

            tier_correct["Overall"] += is_correct.sum().item()
            tier_total["Overall"] += mask.sum().item()

            m1 = mask & (depth_val <= 5)
            tier_correct["Depth 1-5"] += ((preds == y_val) & m1).sum().item()
            tier_total["Depth 1-5"] += m1.sum().item()

            m2 = mask & (depth_val > 5) & (depth_val <= 15)
            tier_correct["Depth 6-15"] += ((preds == y_val) & m2).sum().item()
            tier_total["Depth 6-15"] += m2.sum().item()

            m3 = mask & (depth_val > 15)
            tier_correct["Depth 16-30"] += ((preds == y_val) & m3).sum().item()
            tier_total["Depth 16-30"] += m3.sum().item()

    accs = {}
    print("\n" + "=" * 80)
    print(f"  GRAVIMEM (T={hops_T} HOPS) DYCK-4 EVALUATION RESULTS")
    print("=" * 80)
    for tier in tier_total:
        accs[tier] = (tier_correct[tier] / max(1, tier_total[tier])) * 100.0
        print(f"  => {tier:<12}: Accuracy = {accs[tier]:.2f}%")
    print("=" * 80)

    return {"params": params, "accs": accs, "time_s": time.time() - t0}


@app.local_entrypoint()
def main():
    print("Launching Gravimem Deep Hops (T=10) Dyck-4 on dedicated Modal GPU...")
    res = run_dyck_deep_hops.remote()
    print("Run complete!")
