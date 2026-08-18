"""
Gravimem Architecture Optimization & Research Suite on Modal Cloud GPU.
Systematically iterates and benchmarks:
- Exp 1: Baseline 1-Layer vs 3-Layer Deep
- Exp 2: Gravimem Pre-LN + Residual Highway Scaling
- Exp 3: Gravimem + Step-Embedding Conditioning (Time-aware Recurrence)
- Exp 4: Gravimem + Learned Per-Head Teleportation Alpha
- Exp 5: Combined Champion Architecture (Gravimem-Pro)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm", "requests")
)

app = modal.App("gravimem-transformer-research", image=image)


@app.function(gpu="T4", timeout=3600)
def run_architecture_suite(
    max_steps: int = 3000,
    batch_size: int = 64,
    seq_len: int = 128,
    lr: float = 2e-3
):
    import math
    import urllib.request
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 75)
    print("  GRAVIMEM TRANSFORMER ARCHITECTURAL OPTIMIZATION SUITE")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 1. Dataset in Cloud
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode('utf-8')
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

    # ----------------------------------------------------
    # Model 1: Standard 1-Layer Transformer (Baseline)
    # ----------------------------------------------------
    class Baseline1Layer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.d_model = d_model
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
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
            # Pre-LN
            x_norm = self.ln1(x)
            Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + self.mask[:L, :L], dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            x = x + self.out_proj(H)
            x = x + self.mlp(self.ln2(x))
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # ----------------------------------------------------
    # Model 2: Standard 3-Layer Deep (Upper Bound, 3x params)
    # ----------------------------------------------------
    class Baseline3Layer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512, n_layers=3):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'q': nn.Linear(d_model, d_model, bias=False),
                    'k': nn.Linear(d_model, d_model, bias=False),
                    'v': nn.Linear(d_model, d_model, bias=False),
                    'out': nn.Linear(d_model, d_model, bias=False),
                    'ln1': nn.LayerNorm(d_model),
                    'ln2': nn.LayerNorm(d_model),
                    'mlp': nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
                }) for _ in range(n_layers)
            ])
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
                x_norm = layer['ln1'](x)
                Q = layer['q'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = layer['k'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = layer['v'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + self.mask[:L, :L], dim=-1)
                H = (att @ V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                x = x + layer['out'](H)
                x = x + layer['mlp'](layer['ln2'](x))
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # ----------------------------------------------------
    # Model 3: Gravimem-Pro (Pre-LN + Step Embeddings + Learned Multi-Head Teleportation)
    # ----------------------------------------------------
    class GravimemPro(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.T = T

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            
            # Step Embeddings: informs the shared block what iteration it is in!
            self.step_emb = nn.Embedding(T, d_model)

            # Single Shared Block
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

            # Learned Teleportation Alpha per Head (initialized around 0.15)
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))  # sigmoid(-1.73) ~ 0.15

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight

            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.register_buffer("mask", mask)

        def forward(self, idx, targets=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)  # (H, 1, 1)

            # Residual scaling factor for recurrent depth
            res_scale = 1.0 / math.sqrt(self.T)

            for step in range(self.T):
                # Inject Step Embedding
                step_vec = self.step_emb(torch.tensor(step, device=idx.device)).unsqueeze(0).unsqueeze(0)
                x_step = x + step_vec

                # Pre-LN
                x_norm = self.ln1(x_step)
                Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                # Markov Transition Matrix P
                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k) + self.mask[:L, :L], dim=-1)

                # Causal Mass Update: M = (1 - alpha) * P @ M + alpha * I
                M = (1.0 - alpha) * torch.matmul(P, M) + alpha * I

                # Value Mixing with Mass
                H = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                x = x + res_scale * self.out_proj(H)
                x = x + res_scale * self.mlp(self.ln2(x))

            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # Generic Training Helper with Cosine Annealing
    def train_model(model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n>>> Training {name} ({num_params:,} parameters) for {max_steps} steps...")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-4)

        history = []
        for step in range(max_steps):
            xb, yb = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 500 == 0 or step == 0:
                model.eval()
                with torch.no_grad():
                    vx, vy = get_batch('val')
                    _, vloss = model(vx, vy)
                model.train()
                print(f"Step {step+1:4d}/{max_steps} | Train Loss: {loss.item():.4f} | Val Loss: {vloss.item():.4f}")
                history.append((step + 1, loss.item(), vloss.item()))

        # Sample generation
        model.eval()
        context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long, device=device)
        generated = context
        with torch.no_grad():
            for _ in range(250):
                c = generated[:, -seq_len:]
                logits, _ = model(c)
                probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_tok), dim=1)

        sample = decode(generated[0].tolist())
        return history, sample

    # Train all 3 models in cloud
    std1 = Baseline1Layer().to(device)
    std1_hist, std1_sample = train_model(std1, "Standard 1-Layer Transformer (222k params)")

    grav_pro = GravimemPro(T=3).to(device)
    grav_hist, grav_sample = train_model(grav_pro, "Gravimem-Pro (223k params, T=3 Steps + Pre-LN + Step-Emb + Learned Alpha)")

    std3 = Baseline3Layer(n_layers=3).to(device)
    std3_hist, std3_sample = train_model(std3, "Standard 3-Layer Deep Transformer (618k params)")

    print("\n" + "=" * 75)
    print("  FINAL RESEARCH SUITE RESULTS (3,000 STEPS)")
    print("=" * 75)
    print(f"1. Standard 1-Layer  (222,720 params): Val Loss = {std1_hist[-1][2]:.4f}")
    print(f"2. Gravimem-Pro      (223,104 params): Val Loss = {grav_hist[-1][2]:.4f}")
    print(f"3. Standard 3-Layer  (618,240 params): Val Loss = {std3_hist[-1][2]:.4f}")

    print("\n--- SAMPLE GENERATION: Gravimem-Pro ---")
    print(grav_sample[:350])

    return {
        "std1_val_loss": std1_hist[-1][2],
        "gravimem_pro_val_loss": grav_hist[-1][2],
        "std3_val_loss": std3_hist[-1][2],
        "std1_history": std1_hist,
        "grav_history": grav_hist,
        "std3_history": std3_hist
    }


@app.local_entrypoint()
def main():
    print("Launching Architecture Optimization Suite on Modal Cloud GPU...")
    res = run_architecture_suite.remote(max_steps=3000)
    print("\nOptimization Suite Finished!")
    print("Summary:", res)
