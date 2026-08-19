"""
Mechanistic Experiment 2: Dynamic Course-Correction vs Static Routing Graph
Tests whether Gravimem requires dynamic trajectory-dependent routing (pi^(t) recomputed at each hop)
versus reusing a static frozen jump policy (pi^(1) frozen for all hops).
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-mech2-routing-dynamics", image=image)


@app.function(gpu="T4", timeout=3600)
def run_routing_dynamics():
    import math
    import random
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  MECHANISTIC EXP 2: DYNAMIC COURSE-CORRECTION VS STATIC ROUTING GRAPH")
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
    hops_T = 4

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1 : i + seq_len + 1] for i in ix]).to(device)
        return x, y

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 255]

    class RoutingVariantGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, routing_mode="dynamic", T=4, d_model=128, d_mlp=512, max_len=256):
            super().__init__()
            self.jump_offsets = jump_offsets
            self.K = len(jump_offsets)
            self.routing_mode = routing_mode  # "dynamic", "static_frozen", "uniform"
            self.T = T
            self.d_model = d_model

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)

            self.ln1 = nn.LayerNorm(d_model)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            if routing_mode in ["dynamic", "static_frozen"]:
                self.jump_policy = nn.Linear(d_model, self.K)
            else:
                self.jump_policy = None

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

        def forward(self, idx):
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
            frozen_pi = None

            for step in range(self.T):
                if self.routing_mode == "dynamic":
                    s_norm = self.ln1(s)
                    policy_logits = self.jump_policy(s_norm)
                    policy_logits = policy_logits.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                    pi = F.softmax(policy_logits, dim=-1)

                elif self.routing_mode == "static_frozen":
                    if step == 0:
                        s_norm = self.ln1(s)
                        policy_logits = self.jump_policy(s_norm)
                        policy_logits = policy_logits.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                        frozen_pi = F.softmax(policy_logits, dim=-1)
                    pi = frozen_pi

                elif self.routing_mode == "uniform":
                    valid_counts = valid_mask[:L].sum(dim=-1, keepdim=True).unsqueeze(0)  # [1, L, 1]
                    pi = valid_mask[:L].float().unsqueeze(0) / valid_counts.clamp(min=1)

                surfed_v = torch.sum(pi.unsqueeze(-1) * V_jumps, dim=2)
                surfed_out = self.out_proj(surfed_v)

                s_flat = s.view(B * L, self.d_model)
                out_flat = surfed_out.view(B * L, self.d_model)
                s_next = self.gru(out_flat, s_flat)
                s = s_next.view(B, L, self.d_model)

            x = s + self.mlp(self.ln2(s))
            return self.head(self.ln_f(x))

    modes = [
        ("Dynamic Course-Correction (Recompute pi^(t) every hop)", "dynamic"),
        ("Static Frozen Graph (Compute pi^(1) once, freeze for t>1)", "static_frozen"),
        ("Uniform Static Graph (Fixed 1/K uniform weights)", "uniform"),
    ]

    results = {}
    for name, mode in modes:
        print(f"\n---> Training Routing Mode: {name}")
        model = RoutingVariantGravimem(vocab_size, jump_offsets, routing_mode=mode, T=hops_T).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            x, y = get_batch("train")
            opt.zero_grad()
            loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(50):
                x_val, y_val = get_batch("val")
                loss_val = F.cross_entropy(model(x_val).view(-1, vocab_size), y_val.view(-1))
                val_losses.append(loss_val.item())

        mean_loss = sum(val_losses) / len(val_losses)
        ppl = math.exp(mean_loss)
        elapsed = time.time() - t0
        results[mode] = {"name": name, "val_loss": mean_loss, "ppl": ppl, "time_s": elapsed}
        print(f"     => Mode: {mode} | Val Loss: {mean_loss:.4f} | Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("  ROUTING DYNAMICS COMPARISON SUMMARY")
    print("=" * 80)
    print("Routing Mode                                    | Val Loss | Perplexity | Degradation vs Dynamic")
    print("-" * 80)
    dyn_ppl = results["dynamic"]["ppl"]
    for mode in ["dynamic", "static_frozen", "uniform"]:
        r = results[mode]
        deg = ((r["ppl"] - dyn_ppl) / dyn_ppl) * 100.0
        print(f"{r['name']:<48} |  {r['val_loss']:.4f}  |   {r['ppl']:5.2f}    |  {'+' if deg>=0 else ''}{deg:5.1f}%")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main():
    print("Launching Mechanistic Experiment 2 on dedicated Modal GPU...")
    res = run_routing_dynamics.remote()
    print("Experiment 2 complete!")
