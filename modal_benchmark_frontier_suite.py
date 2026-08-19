"""
Master Frontier Benchmark Suite on Modal GPU:
Runs 4 frontier experiments simultaneously in parallel containers:

1. Exp 1: Multi-Epoch Deep Convergence (5,000 Steps + Cosine LR Schedule)
   - Gravimem 1-Layer vs Standard Transformer 4-Layer over 75+ dataset passes.

2. Exp 2: Needle-In-A-Haystack / Long-Distance Associative Recall
   - Key-Value retrieval across L=512 and L=1024 with distractor noise at variable depths.

3. Exp 3: Zero-Shot Context Length Extrapolation
   - Train on L=256 -> Evaluate zero-shot on L=256, 512, 1024.

4. Exp 4: Extreme Context Scaling & OOM Frontier (L=256..4096)
   - Memory & throughput scaling curve, pinpointing OOM boundaries.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("gravimem-frontier-benchmark-suite", image=image)


# ==============================================================================
# EXPERIMENT 1: EXTENDED MULTI-EPOCH DEEP CONVERGENCE (5,000 STEPS)
# ==============================================================================
@app.function(gpu="T4", timeout=3600)
def run_exp1_deep_convergence():
    import math
    import time
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 1: EXTENDED MULTI-EPOCH DEEP CONVERGENCE (5,000 STEPS)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

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
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 5000

    def get_batch(split):
        d = train_lm if split == 'train' else val_lm
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    # Standard Transformer Block & LM
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

    # Gravimem Surfer (Optimized with Precomputed GPU Buffers)
    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 450, 511]

    class GravimemSurfer(nn.Module):
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

    causal_mask = torch.full((block_size, block_size), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    models = {
        "Gravimem (1 Layer, T=4 Hops)": GravimemSurfer(vocab_size, jump_offsets, T=4, d_model=d_model, d_mlp=d_mlp).to(device),
        "Standard Transformer (4 Layers)": MultiLayerTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n---> Running 5,000-Step Deep Training for: {name}")
        params = sum(p.numel() for p in model.parameters())
        print(f"     Parameter Count: {params:,}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        step_losses = []

        for step in range(1, num_steps + 1):
            model.train()
            x, y = get_batch('train')
            optimizer.zero_grad()

            if "Gravimem" in name:
                logits = model(x)
            else:
                logits = model(x, causal_mask)

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step_losses.append(loss.item())

            if step % 1000 == 0 or step == num_steps:
                recent_train_loss = sum(step_losses[-200:]) / len(step_losses[-200:])
                print(f"     Step {step:4d}/{num_steps} | Train Loss: {recent_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {time.time()-t0:.1f}s")

        elapsed = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Validation Evaluation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(50):
                x_val, y_val = get_batch('val')
                if "Gravimem" in name:
                    logits = model(x_val)
                else:
                    logits = model(x_val, causal_mask)
                l = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
                val_losses.append(l.item())

        mean_val_loss = float(torch.tensor(val_losses).mean())
        val_ppl = math.exp(mean_val_loss)
        final_train_loss = sum(step_losses[-200:]) / len(step_losses[-200:])

        results[name] = {
            "params": params,
            "train_loss": final_train_loss,
            "val_loss": mean_val_loss,
            "val_ppl": val_ppl,
            "overfit_gap": mean_val_loss - final_train_loss,
            "peak_vram_mb": peak_vram,
            "elapsed_s": elapsed,
        }
        print(f"     => Val Loss: {mean_val_loss:.4f} | PPL: {val_ppl:.2f} | Train Loss: {final_train_loss:.4f} | VRAM: {peak_vram:.1f}MB")

    return results


# ==============================================================================
# EXPERIMENT 2: NEEDLE-IN-A-HAYSTACK ASSOCIATIVE RECALL BENCHMARK
# ==============================================================================
@app.function(gpu="T4", timeout=3600)
def run_exp2_needle_in_haystack():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 2: NEEDLE-IN-A-HAYSTACK ASSOCIATIVE RECALL BENCHMARK")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    vocab_size = 256
    block_size = 512
    batch_size = 32
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1200

    # Key tokens: 10..29 (20 keys), Value tokens: 30..49 (20 values)
    # Query Prompt token: 1
    # Distractor tokens: 60..255
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

    # Models
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

        # Systematic Evaluation across all Needle Depths
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

    return accuracy_results


# ==============================================================================
# EXPERIMENT 3: ZERO-SHOT CONTEXT LENGTH EXTRAPOLATION (256 -> 512, 1024)
# ==============================================================================
@app.function(gpu="T4", timeout=3600)
def run_exp3_length_extrapolation():
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
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

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

        # Zero-shot evaluation across L=256, 512, 1024
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

    return extrap_results


# ==============================================================================
# EXPERIMENT 4: EXTREME CONTEXT SCALING & OOM FRONTIER (L=256..4096)
# ==============================================================================
@app.function(gpu="T4", timeout=3600)
def run_exp4_extreme_context():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 4: EXTREME CONTEXT SCALING & OOM FRONTIER (L=256..4096)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    context_lengths = [256, 512, 1024, 2048, 4096]
    batch_size = 8
    vocab_size = 256
    d_model = 128
    n_heads = 4
    d_mlp = 512

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

    class ScalingTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512, max_len=4096):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len + 16, d_model)
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
            return self.head(self.ln_f(x))

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4095]

    class ScalingGravimem(nn.Module):
        def __init__(self, vocab_size, jump_offsets, T=4, d_model=128, d_mlp=512, max_len=4096):
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

    causal_mask_4k = torch.full((4096, 4096), float('-inf'), device=device)
    causal_mask_4k = torch.triu(causal_mask_4k, diagonal=1)

    gravimem = ScalingGravimem(vocab_size, jump_offsets, T=4, d_model=d_model, d_mlp=d_mlp, max_len=4096).to(device)
    transformer = ScalingTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp, max_len=4096).to(device)

    scaling_results = {"gravimem": {}, "transformer": {}}

    for L in context_lengths:
        print(f"\n---> Profiling Context Length L = {L}")
        dummy_x = torch.randint(0, vocab_size, (batch_size, L), device=device)
        dummy_y = torch.randint(0, vocab_size, (batch_size, L), device=device)

        # 1. Gravimem Profile
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):
                out = gravimem(dummy_x)
                loss = F.cross_entropy(out.view(-1, vocab_size), dummy_y.view(-1))
                loss.backward()

            torch.cuda.synchronize()
            t0 = time.time()
            n_iters = 10
            for _ in range(n_iters):
                gravimem.zero_grad()
                out = gravimem(dummy_x)
                loss = F.cross_entropy(out.view(-1, vocab_size), dummy_y.view(-1))
                loss.backward()
            torch.cuda.synchronize()
            latency_ms = ((time.time() - t0) / n_iters) * 1000.0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tok_per_s = (batch_size * L) / (latency_ms / 1000.0)
            scaling_results["gravimem"][L] = {
                "vram_mb": peak_vram,
                "latency_ms": latency_ms,
                "tok_per_s": tok_per_s,
                "status": "OK"
            }
            print(f"     [Gravimem 1L]    L={L:4d} | VRAM: {peak_vram:7.1f} MB | Latency: {latency_ms:6.1f} ms | Tok/s: {tok_per_s:9,.0f}")
        except torch.cuda.OutOfMemoryError:
            scaling_results["gravimem"][L] = {"status": "OOM"}
            print(f"     [Gravimem 1L]    L={L:4d} | OOM (CUDA Out Of Memory)")

        # 2. Standard Transformer Profile
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):
                out = transformer(dummy_x, causal_mask_4k[:L, :L])
                loss = F.cross_entropy(out.view(-1, vocab_size), dummy_y.view(-1))
                loss.backward()

            torch.cuda.synchronize()
            t0 = time.time()
            n_iters = 10
            for _ in range(n_iters):
                transformer.zero_grad()
                out = transformer(dummy_x, causal_mask_4k[:L, :L])
                loss = F.cross_entropy(out.view(-1, vocab_size), dummy_y.view(-1))
                loss.backward()
            torch.cuda.synchronize()
            latency_ms = ((time.time() - t0) / n_iters) * 1000.0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            tok_per_s = (batch_size * L) / (latency_ms / 1000.0)
            scaling_results["transformer"][L] = {
                "vram_mb": peak_vram,
                "latency_ms": latency_ms,
                "tok_per_s": tok_per_s,
                "status": "OK"
            }
            print(f"     [Transformer 4L] L={L:4d} | VRAM: {peak_vram:7.1f} MB | Latency: {latency_ms:6.1f} ms | Tok/s: {tok_per_s:9,.0f}")
        except torch.cuda.OutOfMemoryError:
            scaling_results["transformer"][L] = {"status": "OOM"}
            print(f"     [Transformer 4L] L={L:4d} | OOM (CUDA Out Of Memory)")

    return scaling_results


# ==============================================================================
# MAIN ENTRYPOINT: SPAWN ALL 4 EXPERIMENTS IN PARALLEL
# ==============================================================================
@app.local_entrypoint()
def main():
    print("=" * 80)
    print("  LAUNCHING 4 FRONTIER EXPERIMENTS IN PARALLEL ON MODAL GPU CLOUD...")
    print("=" * 80)

    # Spawn all 4 functions concurrently on Modal GPU
    call1 = run_exp1_deep_convergence.spawn()
    call2 = run_exp2_needle_in_haystack.spawn()
    call3 = run_exp3_length_extrapolation.spawn()
    call4 = run_exp4_extreme_context.spawn()

    print("All 4 containers spawned concurrently! Awaiting parallel completion...")

    res1 = call1.get()
    print("\n[✓] Experiment 1 (Deep Convergence) Complete!")

    res2 = call2.get()
    print("[✓] Experiment 2 (Needle-In-A-Haystack) Complete!")

    res3 = call3.get()
    print("[✓] Experiment 3 (Length Extrapolation) Complete!")

    res4 = call4.get()
    print("[✓] Experiment 4 (Extreme Context Scaling) Complete!")

    print("\n" + "=" * 80)
    print("  ALL 4 FRONTIER EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return {
        "exp1_deep_convergence": res1,
        "exp2_needle_haystack": res2,
        "exp3_length_extrapolation": res3,
        "exp4_extreme_context": res4,
    }
