"""
Experiment 3: Zero-Shot Context Length Extrapolation (Train L=256 -> Test L=256, 512, 1024)
Evaluates out-of-distribution length generalization for Gravimem vs Standard Transformer.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("gravimem-exp3-length-extrapolation", image=image)


@app.function(gpu="T4", timeout=3600)
def run_length_extrapolation():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 3: ZERO-SHOT CONTEXT LENGTH EXTRAPOLATION (256 -> 512, 1024)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_lm = data[:n_train]
    val_lm = data[n_train:]

    train_len = 256
    eval_lengths = [256, 512, 1024]
    max_eval_len = 1024
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 2000

    def get_batch(split, seq_len):
        d = train_lm if split == 'train' else val_lm
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

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

    class ExtrapTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_eval_len + 16, d_model)
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

        def forward(self, idx, causal_mask):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for block in self.blocks:
                x = block(x, causal_mask)
            return self.head(self.ln_f(x))

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1023]

    class ExtrapGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, d_mlp=512, max_len=1024):
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
            self.head.weight = self.tok_emb.weight

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

    causal_mask_1024 = torch.full((max_eval_len, max_eval_len), float('-inf'), device=device)
    causal_mask_1024 = torch.triu(causal_mask_1024, diagonal=1)

    models = {
        "Gravimem (1 Layer, T=4 Hops)": ExtrapGravimem(vocab_size, jump_offsets, T=4, d_model=d_model, d_mlp=d_mlp, max_len=max_eval_len).to(device),
        "Standard Transformer (4 Layers)": ExtrapTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    extrap_results = {}

    for name, model in models.items():
        print(f"\n---> Training strictly on short context L={train_len} for: {name}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        for step in range(1, num_steps + 1):
            model.train()
            x, y = get_batch('train', train_len)
            optimizer.zero_grad()

            if "Gravimem" in name:
                logits = model(x)
            else:
                logits = model(x, causal_mask_1024[:train_len, :train_len])

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 500 == 0 or step == num_steps:
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f}")

        model.eval()
        length_ppls = {}
        with torch.no_grad():
            for seq_l in eval_lengths:
                val_losses = []
                for _ in range(30):
                    x_eval, y_eval = get_batch('val', seq_l)
                    if "Gravimem" in name:
                        logits = model(x_eval)
                    else:
                        logits = model(x_eval, causal_mask_1024[:seq_l, :seq_l])
                    l = F.cross_entropy(logits.view(-1, vocab_size), y_eval.view(-1))
                    val_losses.append(l.item())
                mean_l = float(torch.tensor(val_losses).mean())
                ppl = math.exp(mean_l)
                length_ppls[seq_l] = {"val_loss": mean_l, "ppl": ppl}
                print(f"     => Eval Length L={seq_l:4d}: Val Loss = {mean_l:.4f} | PPL = {ppl:.2f}")

        extrap_results[name] = length_ppls

    print("\n" + "=" * 80)
    print("  EXPERIMENT 3: ZERO-SHOT LENGTH EXTRAPOLATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model Architecture':<34} | {'L=256 (Train)':<14} | {'L=512 (Zero-Shot)':<17} | {'L=1024 (Zero-Shot)':<18} | {'Degradation':<11}")
    print("-" * 102)
    for name, r in extrap_results.items():
        p256 = r[256]["ppl"]
        p512 = r[512]["ppl"]
        p1024 = r[1024]["ppl"]
        deg = (p1024 / p256 - 1.0) * 100.0
        print(f"{name:<34} | PPL: {p256:<8.2f} | PPL: {p512:<11.2f} | PPL: {p1024:<12.2f} | +{deg:6.1f}%")
    print("=" * 80)

    return extrap_results


@app.local_entrypoint()
def main():
    print("Launching Experiment 3 (Length Extrapolation) on dedicated Modal GPU...")
    res = run_length_extrapolation.remote()
    print("Experiment 3 complete!")
