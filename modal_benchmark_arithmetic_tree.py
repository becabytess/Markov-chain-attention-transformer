"""
Binary Arithmetic Expression Tree Benchmark on Modal Cloud GPU.
Tests recursive expression tree evaluation:
e.g. ((a + b) * (c - d))
Where token dependencies form hierarchical Directed Acyclic Graphs (DAGs).
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-arithmetic-tree", image=image)


@app.function(gpu="T4", timeout=3600)
def run_tree_suite():
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: BINARY ARITHMETIC EXPRESSION TREE BENCHMARK")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Vocab: digits 0-9 (0..9), operators: + (10), - (11), * (12), (, ) (13, 14), = (15)
    # Target: modulo 10 result (0..9)
    PLUS, MINUS, MULT = 10, 11, 12
    LPAR, RPAR, EQUAL = 13, 14, 15
    VOCAB_SIZE = 16

    def generate_expr_tree(depth=2):
        """Recursively generates binary expression trees of given depth."""
        if depth == 0:
            val = random.randint(0, 9)
            return [val], val
        
        op = random.choice([PLUS, MINUS, MULT])
        left_toks, left_val = generate_expr_tree(depth - 1)
        right_toks, right_val = generate_expr_tree(depth - 1)

        if op == PLUS:
            val = (left_val + right_val) % 10
        elif op == MINUS:
            val = (left_val - right_val) % 10
        else:
            val = (left_val * right_val) % 10

        toks = [LPAR] + left_toks + [op] + right_toks + [RPAR]
        return toks, val

    def make_tree_dataset(n_samples, depth=2):
        xs, ys = [], []
        for _ in range(n_samples):
            toks, val = generate_expr_tree(depth=depth)
            toks = toks + [EQUAL]
            xs.append(toks)
            ys.append(val)
        
        max_l = max(len(s) for s in xs)
        padded = [s + [0] * (max_l - len(s)) for s in xs]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    # Depth 2: ((a + b) * (c - d)) -> requires 2 levels of composition
    # Depth 3: (((a+b)*(c-d)) + ((e-f)*(g+h))) -> requires 3 levels of composition
    print("Generating Depth-2 & Depth-3 Expression Tree Datasets...")
    x_tr2, y_tr2 = make_tree_dataset(50000, depth=2)
    x_val2, y_val2 = make_tree_dataset(5000, depth=2)
    x_val3, y_val3 = make_tree_dataset(5000, depth=3)

    loader2 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tr2, y_tr2), batch_size=128, shuffle=True
    )

    # -------------------------------------------------------------
    # Models
    # -------------------------------------------------------------
    class GravimemTree(nn.Module):
        def __init__(self, vocab_size=16, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, T=3):
            super().__init__()
            self.T = T
            self.d_k = d_model // n_heads
            self.n_heads = n_heads
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 10)

        def forward(self, idx, T=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            steps = T if T is not None else self.T

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)

            for _ in range(steps):
                x_norm = self.ln1(x)
                Q = self.q(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                M = (1.0 - alpha) * (P @ M) + alpha * I

                H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + 0.5 * self.out(H)
                x = x + 0.5 * self.mlp(self.ln2(x))

            x = self.ln_f(x)
            return self.head(x[:, -1, :])

    class StandardTree(nn.Module):
        def __init__(self, vocab_size=16, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, n_layers=1):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
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
            self.head = nn.Linear(d_model, 10)

        def forward(self, idx):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for l in self.layers:
                x_norm = l['ln1'](x)
                Q = l['q'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = l['k'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = l['v'](x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + l['out'](H)
                x = x + l['mlp'](l['ln2'](x))
            x = self.ln_f(x)
            return self.head(x[:, -1, :])

    tree_std1 = StandardTree(n_layers=1).to(device)
    tree_deep3 = StandardTree(n_layers=3).to(device)
    tree_grav = GravimemTree(T=3).to(device)

    def train_tree(model, name, is_grav=False):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        for step, (bx, by) in enumerate(loader2):
            if step > 2500:
                break
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    print("\nTraining on Depth-2 Arithmetic Trees...")
    train_tree(tree_std1, "Standard 1-Layer")
    train_tree(tree_grav, "Gravimem (1 layer, T=3)", is_grav=True)
    train_tree(tree_deep3, "Standard 3-Layer (3x params)")

    def eval_acc(model, tx, ty, T=None):
        model.eval()
        correct, total = 0, 0
        l = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=256)
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx, T=T) if T is not None else model(bx)
                correct += (logits.argmax(dim=-1) == by).sum().item()
                total += by.size(0)
        return 100.0 * correct / total

    print("\n" + "=" * 70)
    print("  RESULTS: Arithmetic Expression Tree Accuracy")
    print("=" * 70)
    print(f"1. Standard 1-Layer (205k params): Depth-2: {eval_acc(tree_std1, x_val2, y_val2):5.2f}% | Depth-3: {eval_acc(tree_std1, x_val3, y_val3):5.2f}%")
    print(f"2. Gravimem (1 layer, 205k params): Depth-2: {eval_acc(tree_grav, x_val2, y_val2, T=3):5.2f}% | Depth-3 (T=5): {eval_acc(tree_grav, x_val3, y_val3, T=5):5.2f}%")
    print(f"3. Standard 3-Layer (600k params): Depth-2: {eval_acc(tree_deep3, x_val2, y_val2):5.2f}% | Depth-3: {eval_acc(tree_deep3, x_val3, y_val3):5.2f}%")

    return {
        "std1": [eval_acc(tree_std1, x_val2, y_val2), eval_acc(tree_std1, x_val3, y_val3)],
        "gravimem": [eval_acc(tree_grav, x_val2, y_val2, T=3), eval_acc(tree_grav, x_val3, y_val3, T=5)],
        "std3": [eval_acc(tree_deep3, x_val2, y_val2), eval_acc(tree_deep3, x_val3, y_val3)]
    }


@app.local_entrypoint()
def main():
    print("Launching Arithmetic Tree Benchmark on Modal GPU...")
    res = run_tree_suite.remote()
    print("\nArithmetic Tree Benchmark Complete!")
    print("Results:", res)
