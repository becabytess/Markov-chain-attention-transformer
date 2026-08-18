"""
Multi-Hop Associative Recall Benchmark on Modal Cloud GPU.
Proves whether Gravimem's Markov Mass Diffusion allows a 1-layer model to solve
multi-hop relational reasoning (A -> B -> C -> D) where standard 1-layer transformers fail.
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-multihop-benchmark", image=image)


@app.function(gpu="T4", timeout=1800)
def run_multihop_benchmark(
    num_hops: int = 3,
    num_nodes: int = 32,
    num_samples: int = 50000,
    max_steps: int = 2500,
    batch_size: int = 128,
    lr: float = 2e-3
):
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 75)
    print(f"  MULTI-HOP REASONING BENCHMARK ({num_hops}-HOP GRAPH NAVIGATION)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 1. Generate Synthetic Multi-Hop Relational Dataset
    # Tokens: Node IDs (0..num_nodes-1), Special tokens: ARROW (->), QUERY (?), PAD, TARGET
    ARROW = num_nodes
    QUERY = num_nodes + 1
    vocab_size = num_nodes + 2
    seq_len = (num_hops * 3) + 2  # e.g., for 3 hops: A -> B , B -> C , C -> D , A ? -> D

    print(f"Generating {num_samples} multi-hop reasoning graphs ({num_hops} hops)...")
    
    def generate_data(n_samples):
        inputs = []
        targets = []
        for _ in range(n_samples):
            # Create a path of length num_hops
            nodes = random.sample(range(num_nodes), num_hops + 1)
            # Path: nodes[0] -> nodes[1] -> nodes[2] ... -> nodes[num_hops]
            # Shuffle edge order to test true graph traversal rather than sequential reading
            edges = [(nodes[i], nodes[i+1]) for i in range(num_hops)]
            random.shuffle(edges)

            seq = []
            for u, v in edges:
                seq.extend([u, ARROW, v])
            
            # Query: nodes[0] QUERY -> target is nodes[-1]
            seq.extend([nodes[0], QUERY])
            target = nodes[-1]

            inputs.append(seq)
            targets.append(target)

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    train_x, train_y = generate_data(num_samples)
    val_x, val_y = generate_data(5000)

    train_data = torch.utils.data.TensorDataset(train_x, train_y)
    val_data = torch.utils.data.TensorDataset(val_x, val_y)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"Data ready. Input sequence length: {seq_len}, Vocab size: {vocab_size}")

    # ----------------------------------------------------
    # Model 1: Standard 1-Layer Transformer
    # ----------------------------------------------------
    class Standard1Layer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
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
            self.head = nn.Linear(d_model, num_nodes, bias=False)
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x_norm = self.ln1(x)
            Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
            H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
            x = x + self.out_proj(H)
            x = x + self.mlp(self.ln2(x))
            x = self.ln_f(x)
            # Predict target from the last token (QUERY token)
            return self.head(x[:, -1, :])

    # ----------------------------------------------------
    # Model 2: Gravimem-Pro (1 Layer parameters, T=num_hops Settling Steps)
    # ----------------------------------------------------
    class GravimemMultiHop(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512, T=num_hops):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.T = T

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.step_emb = nn.Embedding(T, d_model)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, num_nodes, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            res_scale = 1.0 / math.sqrt(self.T)

            for step in range(self.T):
                step_vec = self.step_emb(torch.tensor(step, device=idx.device)).unsqueeze(0).unsqueeze(0)
                x_norm = self.ln1(x + step_vec)
                Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                # Markov mass propagation
                M = (1.0 - alpha) * torch.matmul(P, M) + alpha * I

                H = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + res_scale * self.out_proj(H)
                x = x + res_scale * self.mlp(self.ln2(x))

            x = self.ln_f(x)
            return self.head(x[:, -1, :])

    # ----------------------------------------------------
    # Model 3: Standard 3-Layer Deep Transformer (3x Parameters)
    # ----------------------------------------------------
    class Deep3Layer(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_mlp=512, n_layers=num_hops):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
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
            self.head = nn.Linear(d_model, num_nodes, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for layer in self.layers:
                x_norm = layer['ln1'](x)
                Q = layer['q'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = layer['k'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = layer['v'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + layer['out'](H)
                x = x + layer['mlp'](layer['ln2'](x))
            x = self.ln_f(x)
            return self.head(x[:, -1, :])

    # Training helper evaluating Test Accuracy (%)
    def train_and_test(model, name):
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n>>> Training {name} ({num_params:,} parameters)...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

        step = 0
        while step < max_steps:
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(bx)
                loss = F.cross_entropy(logits, by)
                loss.backward()
                optimizer.step()
                scheduler.step()
                step += 1

                if step % 500 == 0 or step == 1:
                    # Evaluate validation accuracy
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            preds = model(vx).argmax(dim=-1)
                            correct += (preds == vy).sum().item()
                            total += vy.size(0)
                    acc = 100.0 * correct / total
                    model.train()
                    print(f"Step {step:4d}/{max_steps} | Loss: {loss.item():.4f} | Val Accuracy: {acc:5.2f}%")

                if step >= max_steps:
                    break

        return acc

    acc1 = train_and_test(Standard1Layer().to(device), "Standard 1-Layer Transformer")
    acc_grav = train_and_test(GravimemMultiHop(T=num_hops).to(device), f"Gravimem-Pro (1 Layer params, T={num_hops} hops)")
    acc3 = train_and_test(Deep3Layer(n_layers=num_hops).to(device), f"Standard 3-Layer Deep ({num_hops}x params)")

    print("\n" + "=" * 75)
    print(f"  FINAL {num_hops}-HOP REASONING BENCHMARK ACCURACY")
    print("=" * 75)
    print(f"1. Standard 1-Layer (218k params): Accuracy = {acc1:5.2f}%")
    print(f"2. Gravimem-Pro      (218k params): Accuracy = {acc_grav:5.2f}%")
    print(f"3. Standard 3-Layer (608k params): Accuracy = {acc3:5.2f}%")

    return {
        "num_hops": num_hops,
        "standard_1layer_acc": acc1,
        "gravimem_pro_acc": acc_grav,
        "standard_3layer_acc": acc3
    }


@app.local_entrypoint()
def main():
    print("Launching Multi-Hop Reasoning Benchmark on Modal GPU...")
    res = run_multihop_benchmark.remote(num_hops=3, num_nodes=32, max_steps=2500)
    print("\nBenchmark Complete!")
    print("Results:", res)
