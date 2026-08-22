"""
Gravimem vs Multi-Layer Standard Transformer Benchmark on Modal GPU:
Compares 1-Layer Gravimem (T=4 hops, 1x parameter budget) against:
- Standard Transformer 1-Layer (1x params)
- Standard Transformer 2-Layer (2x params)
- Standard Transformer 4-Layer (4x params)
- Standard Transformer 6-Layer (6x params)

Evaluated on TinyShakespeare at context L=512:
- Validation Loss & Perplexity
- Parameter Count
- GPU Memory Footprint
- Training / Inference Throughput
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("gravimem-vs-multilayer-transformers", image=image)


@app.function(gpu="T4", timeout=3600)
def run_multilayer_comparison():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM (1-LAYER) vs MULTI-LAYER TRANSFORMERS (1, 2, 4, 6 LAYERS)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Preparation (TinyShakespeare with Context L=512)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train_lm = data[:n_train]
    val_lm = data[n_train:]

    block_size = 512
    batch_size = 32   # 16,384 tokens per batch
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1500

    def get_lm_batch(split):
        d = train_lm if split == 'train' else val_lm
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # Standard Multi-Layer Transformer
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

    class MultiLayerTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size + 16, d_model)
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

    # Gravimem Model (1 Physical Layer with Multi-Scale Jumps and T Thought Steps)
    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 450, 511]

    class GravimemSurfer(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, n_heads=4, d_mlp=512, max_len=512):
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

            # PRECOMPUTE static jump indices & validity mask once as GPU buffers (zero per-step CPU loop overhead)
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

            # Fast vector gather from precomputed GPU buffers
            target_idx = self.target_indices[:L]
            valid_mask = self.valid_jump_mask[:L]

            flat_targets = target_idx.unsqueeze(0).expand(B, -1, -1)
            V_jumps = torch.gather(
                V.unsqueeze(2).expand(-1, -1, self.K, -1),
                1,
                flat_targets.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            )
            V_jumps = V_jumps.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(-1), 0.0)

            # Surfer State initialization
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

    causal_mask = torch.full((block_size, block_size), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    # Models to benchmark
    models = {
        "Gravimem (1 Layer, T=4 Hops)": GravimemSurfer(vocab_size, jump_offsets, T=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "Standard Transformer (1 Layer)": MultiLayerTransformer(vocab_size, n_layers=1, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "Standard Transformer (2 Layers)": MultiLayerTransformer(vocab_size, n_layers=2, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "Standard Transformer (4 Layers)": MultiLayerTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
        "Standard Transformer (6 Layers)": MultiLayerTransformer(vocab_size, n_layers=6, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n---> Training: {name}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"     Parameter Count: {total_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        for step in range(1, num_steps + 1):
            model.train()
            x, y = get_lm_batch('train')
            optimizer.zero_grad()

            if "Gravimem" in name:
                logits = model(x)
            else:
                logits = model(x, causal_mask)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 500 == 0 or step == num_steps:
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Elapsed: {time.time()-t0:.1f}s")

        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        throughput = (num_steps * batch_size * block_size) / elapsed

        # Validation Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(50):
                x_val, y_val = get_lm_batch('val')
                if "Gravimem" in name:
                    logits = model(x_val)
                else:
                    logits = model(x_val, causal_mask)
                l = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
                val_losses.append(l.item())

        mean_val_loss = float(torch.tensor(val_losses).mean())
        val_ppl = math.exp(mean_val_loss)

        results[name] = {
            "params": total_params,
            "val_loss": mean_val_loss,
            "val_ppl": val_ppl,
            "peak_vram_mb": peak_vram,
            "throughput_tok_s": throughput,
            "elapsed_s": elapsed
        }
        print(f"     => Val Loss: {mean_val_loss:.4f} | Perplexity: {val_ppl:.2f} | Peak VRAM: {peak_vram:.1f} MB | Tok/s: {throughput:,.0f}")

    print("\n" + "=" * 80)
    print("  FINAL HEAD-TO-HEAD COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model Architecture':<34} | {'Params':<9} | {'Val Loss':<8} | {'PPL':<6} | {'VRAM (MB)':<9} | {'Tokens/Sec':<11}")
    print("-" * 88)
    for name, r in results.items():
        print(f"{name:<34} | {r['params']:<9,d} | {r['val_loss']:<8.4f} | {r['val_ppl']:<6.2f} | {r['peak_vram_mb']:<9.1f} | {r['throughput_tok_s']:<11,.0f}")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main():
    print("Launching Gravimem vs Multi-Layer Transformers on Modal GPU...")
    res = run_multilayer_comparison.remote()
    print("Multi-Layer Benchmark complete!")
