"""
Modal Cloud Training Script for Gravimem Transformer v0.
Compares:
1. Standard 1-Layer Transformer Baseline (222k params)
2. Gravimem Transformer v0 with Recurrent Shared MLP (222k params, T=3 steps)
3. Standard 3-Layer Deep Transformer (660k params, 3 separate physical layers)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "requests")
)

app = modal.App("gravimem-transformer-v0", image=image)


@app.function(gpu="T4", timeout=1800)
def train_and_evaluate(
    settling_steps: int = 3,
    alpha_teleport: float = 0.15,
    max_steps: int = 2000,
    batch_size: int = 64,
    seq_len: int = 128,
    lr: float = 3e-3
):
    import math
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    print("=" * 70)
    print("  GRAVIMEM TRANSFORMER V0 - FULL RECURRENT EXPERIMENT")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Fetch TinyShakespeare in Cloud
    print("\n[Step 1/4] Fetching TinyShakespeare dataset in cloud...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
    print(f"Loaded {len(text):,} characters of text.")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - seq_len, (batch_size,))
        x = torch.stack([d[i:i+seq_len] for i in ix])
        y = torch.stack([d[i+1:i+seq_len+1] for i in ix])
        return x.to(device), y.to(device)

    # 2. Standard 1-Layer Transformer Baseline
    class Standard1LayerTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.d_model = d_model
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, idx, targets=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + self.mask[:L, :L], dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            x = self.ln_attn(x + self.out_proj(H))
            x = self.ln_mlp(x + self.mlp(x))
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # 3. Gravimem Transformer v0 with Recurrent Shared MLP Loop
    class GravimemRecurrentTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, T=3, alpha=0.15):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.T = T
            self.alpha = alpha

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, idx, targets=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            # Initialize Mass Matrix M = Identity
            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)

            # Outer Recurrent Loop across T Settling Steps
            for step in range(self.T):
                # 1. Fresh Attention Matrix P from current token states x
                Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + self.mask[:L, :L], dim=-1)

                # 2. Causal Markov Mass Update: M = (1 - alpha) * P @ M + alpha * I
                M = (1.0 - self.alpha) * torch.matmul(P, M) + self.alpha * I

                # 3. Value Mixing with accumulated Mass Matrix
                H = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                
                # 4. Residual updates with Shared Reusable Block
                x = self.ln_attn(x + self.out_proj(H))
                x = self.ln_mlp(x + self.mlp(x))

            x = self.ln_f(x)
            logits = self.head(x)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # 4. Standard 3-Layer Deep Transformer (Non-shared parameters for upper bound)
    class TransformerBlock(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln_attn = nn.LayerNorm(d_model)
            self.ln_mlp = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

        def forward(self, x, mask):
            B, L, D = x.shape
            Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + mask, dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, D)
            x = self.ln_attn(x + self.out_proj(H))
            x = self.ln_mlp(x + self.mlp(x))
            return x

    class Standard3LayerTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=128, n_heads=4, d_mlp=512, n_layers=3):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, idx, targets=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for layer in self.layers:
                x = layer(x, self.mask[:L, :L])
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # Generic Training Helper
    def train_model(model, name):
        print(f"\n--- Training {name} ---")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        losses = []

        for step in range(max_steps):
            xb, yb = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 500 == 0 or step == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch('val')
                    _, vloss = model(vx, vy)
                model.train()
                print(f"Step {step+1:4d}/{max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {vloss.item():.4f}")
                losses.append((step + 1, loss.item(), vloss.item()))

        # Generate sample
        model.eval()
        context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long, device=device)
        generated = context
        with torch.no_grad():
            for _ in range(220):
                c = generated[:, -seq_len:]
                logits, _ = model(c)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_tok), dim=1)

        sample_text = decode(generated[0].tolist())
        return losses, sample_text

    # Train Models
    std1_model = Standard1LayerTransformer(vocab_size).to(device)
    std1_losses, std1_sample = train_model(std1_model, "Standard 1-Layer (222k params)")

    grav_model = GravimemRecurrentTransformer(vocab_size, T=settling_steps, alpha=alpha_teleport).to(device)
    grav_losses, grav_sample = train_model(grav_model, f"Gravimem Recurrent v0 (222k params, T={settling_steps} steps)")

    std3_model = Standard3LayerTransformer(vocab_size, n_layers=3).to(device)
    std3_losses, std3_sample = train_model(std3_model, "Standard 3-Layer Deep (660k params)")

    print("\n" + "=" * 70)
    print("  FINAL 3-WAY ARCHITECTURAL COMPARISON")
    print("=" * 70)
    print(f"1. Standard 1-Layer (222k params): Val Loss = {std1_losses[-1][2]:.4f}")
    print(f"2. Gravimem v0      (222k params): Val Loss = {grav_losses[-1][2]:.4f}")
    print(f"3. Standard 3-Layer (660k params): Val Loss = {std3_losses[-1][2]:.4f}")

    print("\n--- SAMPLE GENERATION: Standard 1-Layer ---")
    print(std1_sample[:280])

    print("\n--- SAMPLE GENERATION: Gravimem v0 (Recurrent Shared MLP) ---")
    print(grav_sample[:280])

    print("\n--- SAMPLE GENERATION: Standard 3-Layer Deep ---")
    print(std3_sample[:280])

    return {
        "std_1layer_val_loss": std1_losses[-1][2],
        "gravimem_val_loss": grav_losses[-1][2],
        "std_3layer_val_loss": std3_losses[-1][2]
    }


@app.local_entrypoint()
def main():
    print("Launching 3-Way Gravimem Experiment on Modal Cloud GPU...")
    res = train_and_evaluate.remote(settling_steps=3, alpha_teleport=0.15, max_steps=2000)
    print("\nModal Experiment Complete!")
    print("Results:", res)
