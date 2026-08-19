"""
Experiment 4: Extreme Context Scaling & OOM Frontier (L=256..4096)
Benchmarks peak VRAM memory and forward-backward latency on a dedicated Tesla T4 GPU,
pinpointing OOM thresholds for 4-Layer Standard Transformer vs 1-Layer Gravimem.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy", "requests")
)

app = modal.App("gravimem-exp4-extreme-context", image=image)


@app.function(gpu="T4", timeout=3600)
def run_extreme_context():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  EXPERIMENT 4: EXTREME CONTEXT SCALING & OOM FRONTIER (L=256..4096)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

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

    print("\n" + "=" * 80)
    print("  EXPERIMENT 4: EXTREME CONTEXT SCALING & OOM SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Context Length (L)':<20} | {'Gravimem VRAM':<15} | {'Transformer VRAM':<18} | {'Gravimem Speed':<16} | {'Transformer Speed':<17}")
    print("-" * 96)
    for L in context_lengths:
        g = scaling_results["gravimem"].get(L, {})
        t = scaling_results["transformer"].get(L, {})
        g_vram = f"{g.get('vram_mb', 0):.1f} MB" if g.get("status") == "OK" else "OOM"
        t_vram = f"{t.get('vram_mb', 0):.1f} MB" if t.get("status") == "OK" else "💥 OOM (Crash)"
        g_spd = f"{g.get('tok_per_s', 0):,.0f} tok/s" if g.get("status") == "OK" else "N/A"
        t_spd = f"{t.get('tok_per_s', 0):,.0f} tok/s" if t.get("status") == "OK" else "N/A"
        print(f"L = {L:<16d} | {g_vram:<15} | {t_vram:<18} | {g_spd:<16} | {t_spd:<17}")
    print("=" * 80)

    return scaling_results


@app.local_entrypoint()
def main():
    print("Launching Experiment 4 (Extreme Context Scaling) on dedicated Modal GPU...")
    res = run_extreme_context.remote()
    print("Experiment 4 complete!")
