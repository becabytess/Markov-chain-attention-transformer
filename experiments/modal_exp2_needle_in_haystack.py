"""
Experiment 2: Needle-In-A-Haystack / Long-Distance Associative Recall Benchmark
Evaluates 1-Layer Gravimem vs 4-Layer Standard Transformer on retrieving distant
key-value pairs buried under hundreds of random distractor tokens across needle depths
d in {16, 64, 128, 256, 384, 480}.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("gravimem-exp2-needle-in-haystack", image=image)


@app.function(gpu="T4", timeout=3600)
def run_needle_in_haystack():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 2: NEEDLE-IN-A-HAYSTACK ASSOCIATIVE RECALL BENCHMARK")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

    vocab_size = 256
    block_size = 512
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1200

    def generate_needle_batch(batch_size, seq_len, needle_depth=128):
        x = torch.randint(60, 255, (batch_size, seq_len), device=device)
        y = torch.zeros((batch_size,), dtype=torch.long, device=device)

        for b in range(batch_size):
            k_id = torch.randint(10, 30, (1,)).item()
            v_id = k_id + 20

            pos = seq_len - 2 - needle_depth
            pos = max(0, min(seq_len - 4, pos))

            x[b, pos] = k_id
            x[b, pos + 1] = v_id

            x[b, -2] = 1
            x[b, -1] = k_id
            y[b] = v_id

        return x, y

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

    class NeedleTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, causal_mask):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x, causal_mask)
            return self.head(self.ln_f(x[:, -1]))

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 450, 511]

    class NeedleGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, d_mlp=512, max_len=512):
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
            return self.head(self.ln_f(x[:, -1]))

    causal_mask = torch.full((block_size, block_size), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    models = {
        "Gravimem (1 Layer, T=4 Hops)": NeedleGravimem(vocab_size, jump_offsets, T=4, d_model=d_model, d_mlp=d_mlp).to(device),
        "Standard Transformer (4 Layers)": NeedleTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    depth_test_grid = [16, 64, 128, 256, 384, 480]
    accuracy_results = {}

    for name, model in models.items():
        print(f"\n---> Training Needle-In-A-Haystack on {name}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        for step in range(1, num_steps + 1):
            model.train()
            rand_depth = torch.randint(10, 480, (1,)).item()
            x, y = generate_needle_batch(batch_size, block_size, needle_depth=rand_depth)
            optimizer.zero_grad()

            if "Gravimem" in name:
                logits = model(x)
            else:
                logits = model(x, causal_mask)

            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 400 == 0 or step == num_steps:
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item() * 100
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Batch Acc: {acc:.1f}%")

        model.eval()
        depth_accs = {}
        with torch.no_grad():
            for depth in depth_test_grid:
                correct = 0
                total = 0
                for _ in range(20):
                    x_eval, y_eval = generate_needle_batch(32, block_size, needle_depth=depth)
                    if "Gravimem" in name:
                        logits = model(x_eval)
                    else:
                        logits = model(x_eval, causal_mask)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y_eval).sum().item()
                    total += len(y_eval)
                depth_accs[depth] = (correct / total) * 100.0
                print(f"     => Depth {depth:3d} tokens: Accuracy = {depth_accs[depth]:.1f}%")

        accuracy_results[name] = depth_accs

    print("\n" + "=" * 80)
    print("  EXPERIMENT 2: NEEDLE-IN-A-HAYSTACK RETRIEVAL ACCURACY TABLE")
    print("=" * 80)
    print(f"{'Model Architecture':<34} | " + " | ".join([f"d={d:3d}" for d in depth_test_grid]) + " | Mean Acc")
    print("-" * 88)
    for name, accs in accuracy_results.items():
        row = [f"{accs[d]:5.1f}%" for d in depth_test_grid]
        mean_acc = sum(accs.values()) / len(accs)
        print(f"{name:<34} | " + " | ".join(row) + f" | {mean_acc:6.1f}%")
    print("=" * 80)

    return accuracy_results


@app.local_entrypoint()
def main():
    print("Launching Experiment 2 (Needle-In-A-Haystack) on dedicated Modal GPU...")
    res = run_needle_in_haystack.remote()
    print("Experiment 2 complete!")
