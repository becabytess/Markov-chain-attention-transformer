"""
Nightmare Benchmark 1: Multi-Query Associative Recall (MQAR)
Evaluates 1-Layer Gravimem (T=4 hops, 342k params) vs 4-Layer Standard Transformer (867k params)
on retrieving multiple interleaved key-value pairs (16 pairs) buried under distractor noise at L=512.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-nightmare1-mqar", image=image)


@app.function(gpu="T4", timeout=3600)
def run_mqar_benchmark():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  NIGHTMARE BENCHMARK 1: MULTI-QUERY ASSOCIATIVE RECALL (MQAR)")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dedicated GPU Container: {torch.cuda.get_device_name(0)}")

    vocab_size = 256
    seq_len = 512
    batch_size = 32
    num_kv_pairs = 16
    num_queries = 8
    d_model = 128
    n_heads = 4
    d_mlp = 512
    num_steps = 1500

    def generate_mqar_batch(batch_size, seq_len=512, num_kv=16, num_q=8):
        """
        Generates batches with num_kv distinct key-value pairs scattered randomly
        across the first 400 positions, followed by num_q query keys at the end.
        """
        # Noise tokens in range [100, 255]
        x = torch.randint(100, 255, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

        query_marker = 1

        for b in range(batch_size):
            # Sample unique keys in [10, 49] and values in [50, 89]
            keys = torch.randperm(40)[:num_kv] + 10
            vals = keys + 40

            kv_positions = torch.randperm(350)[:num_kv * 2].sort().values
            for k_idx in range(num_kv):
                p_k = kv_positions[k_idx * 2].item()
                p_v = p_k + 1
                x[b, p_k] = keys[k_idx]
                x[b, p_v] = vals[k_idx]

            # Query phase in the final 100 positions
            query_indices = torch.randperm(num_kv)[:num_q]
            start_q_pos = seq_len - (num_q * 2) - 2
            for q_idx, k_orig_idx in enumerate(query_indices):
                q_pos = start_q_pos + (q_idx * 2)
                x[b, q_pos] = query_marker
                x[b, q_pos + 1] = keys[k_orig_idx]
                # Target to predict right after the key token is the corresponding value
                y[b, q_pos + 1] = vals[k_orig_idx]

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

    class MQARTransformer(nn.Module):
        def __init__(self, vocab_size, n_layers=4, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len + 16, d_model)
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

    jump_offsets = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 450, 511]

    class MQARGravimem(nn.Module):
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
            return self.head(self.ln_f(x))

    causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    models = {
        "Gravimem (1 Layer, T=4 Hops)": MQARGravimem(vocab_size, jump_offsets, T=4, d_model=d_model, d_mlp=d_mlp).to(device),
        "Standard Transformer (4 Layers)": MQARTransformer(vocab_size, n_layers=4, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device),
    }

    results = {}

    for name, model in models.items():
        print(f"\n---> Training MQAR on {name}")
        params = sum(p.numel() for p in model.parameters())
        print(f"     Parameter Count: {params:,}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

        t0 = time.time()
        for step in range(1, num_steps + 1):
            model.train()
            x, y = generate_mqar_batch(batch_size, seq_len, num_kv_pairs, num_queries)
            optimizer.zero_grad()

            if "Gravimem" in name:
                logits = model(x)
            else:
                logits = model(x, causal_mask)

            # Masked Cross-Entropy over query positions only
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 500 == 0 or step == num_steps:
                mask = (y != -100)
                preds = logits.argmax(dim=-1)
                acc = (preds[mask] == y[mask]).float().mean().item() * 100.0
                print(f"     Step {step:4d}/{num_steps} | Loss: {loss.item():.4f} | Query Accuracy: {acc:.1f}% | Elapsed: {time.time()-t0:.1f}s")

        # Rigorous multi-batch validation
        model.eval()
        total_queries = 0
        correct_queries = 0
        with torch.no_grad():
            for _ in range(50):
                x_val, y_val = generate_mqar_batch(32, seq_len, num_kv_pairs, num_queries)
                if "Gravimem" in name:
                    logits = model(x_val)
                else:
                    logits = model(x_val, causal_mask)

                mask = (y_val != -100)
                preds = logits.argmax(dim=-1)
                correct_queries += (preds[mask] == y_val[mask]).sum().item()
                total_queries += mask.sum().item()

        val_acc = (correct_queries / total_queries) * 100.0
        results[name] = {"params": params, "val_accuracy": val_acc, "time_s": time.time() - t0}
        print(f"     => FINAL MQAR EXACT-MATCH RECALL ACCURACY: {val_acc:.2f}%")

    print("\n" + "=" * 80)
    print("  NIGHTMARE BENCHMARK 1: MQAR MULTI-QUERY ACCURACY SUMMARY")
    print("=" * 80)
    print(f"{'Model Architecture':<34} | {'Parameters':<11} | {'MQAR Recall Acc':<16} | {'Training Time':<12}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<34} | {r['params']:<11,d} | {r['val_accuracy']:<15.2f}% | {r['time_s']:<11.1f}s")
    print("=" * 80)

    return results


@app.local_entrypoint()
def main():
    print("Launching Nightmare Benchmark 1 (MQAR) on dedicated Modal GPU...")
    res = run_mqar_benchmark.remote()
    print("Benchmark 1 Complete!")
