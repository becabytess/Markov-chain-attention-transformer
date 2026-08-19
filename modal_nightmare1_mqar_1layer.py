"""
Nightmare Benchmark 1 Follow-up: 1-Layer Standard Transformer on MQAR
Tests if a 1-Layer Standard Transformer can achieve 100% recall on MQAR (L=512, 16 key-value pairs).
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.0.0", "numpy")
)

app = modal.App("gravimem-mqar-1layer", image=image)


@app.function(gpu="T4", timeout=3600)
def run_mqar_1layer():
    import math
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  NIGHTMARE BENCHMARK 1 FOLLOW-UP: 1-LAYER TRANSFORMER ON MQAR")
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
        x = torch.randint(100, 255, (batch_size, seq_len), device=device)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
        query_marker = 1

        for b in range(batch_size):
            keys = torch.randperm(40)[:num_kv] + 10
            vals = keys + 40

            kv_positions = torch.randperm(350)[:num_kv * 2].sort().values
            for k_idx in range(num_kv):
                p_k = kv_positions[k_idx * 2].item()
                p_v = p_k + 1
                x[b, p_k] = keys[k_idx]
                x[b, p_v] = vals[k_idx]

            query_indices = torch.randperm(num_kv)[:num_q]
            start_q_pos = seq_len - (num_q * 2) - 2
            for q_idx, k_orig_idx in enumerate(query_indices):
                q_pos = start_q_pos + (q_idx * 2)
                x[b, q_pos] = query_marker
                x[b, q_pos + 1] = keys[k_orig_idx]
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

    class MQARTransformer1L(nn.Module):
        def __init__(self, vocab_size, n_layers=1, d_model=128, n_heads=4, d_mlp=512):
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

    causal_mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)

    model = MQARTransformer1L(vocab_size, n_layers=1, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n---> Model: Standard Transformer (1 Layer)")
    print(f"     Parameter Count: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)

    t0 = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        x, y = generate_mqar_batch(batch_size, seq_len, num_kv_pairs, num_queries)
        optimizer.zero_grad()
        logits = model(x, causal_mask)
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

    model.eval()
    total_queries = 0
    correct_queries = 0
    with torch.no_grad():
        for _ in range(50):
            x_val, y_val = generate_mqar_batch(32, seq_len, num_kv_pairs, num_queries)
            logits = model(x_val, causal_mask)
            mask = (y_val != -100)
            preds = logits.argmax(dim=-1)
            correct_queries += (preds[mask] == y_val[mask]).sum().item()
            total_queries += mask.sum().item()

    val_acc = (correct_queries / total_queries) * 100.0
    elapsed = time.time() - t0
    print(f"\n================================================================================")
    print(f"  1-LAYER TRANSFORMER MQAR FINAL RESULT: {val_acc:.2f}% (Time: {elapsed:.1f}s)")
    print(f"================================================================================")
    return {"val_accuracy": val_acc, "params": params, "time_s": elapsed}


@app.local_entrypoint()
def main():
    print("Launching 1-Layer Transformer MQAR on dedicated Modal GPU...")
    res = run_mqar_1layer.remote()
    print("Run completed!")
