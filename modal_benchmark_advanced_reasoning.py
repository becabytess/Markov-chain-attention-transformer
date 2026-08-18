"""
Advanced Reasoning & Length Generalization Suite on Modal Cloud GPU.
Benchmarks Gravimem-Pro across 3 core cognitive capabilities:
1. Length Generalization (Train on 2-3 hops -> Test zero-shot on 4-5 hops by dynamically increasing T!)
2. Variable Tracking / Dataflow Dependency Tracing
3. Associative Recall with Heavy Distractor Swarms (Needle in a Haystack)
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-advanced-reasoning", image=image)


@app.function(gpu="T4", timeout=3600)
def run_advanced_suite():
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm import tqdm

    print("=" * 80)
    print("  GRAVIMEM-PRO: ADVANCED REASONING & GENERALIZATION SUITE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # Define Modular Architectures
    # -------------------------------------------------------------
    class GravimemPro(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, max_T=8):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.max_T = max_T

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.step_emb = nn.Embedding(max_T, d_model)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))

            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T=None):
            B, L = idx.shape
            T_steps = T if T is not None else 3
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)
            res_scale = 1.0 / math.sqrt(T_steps)

            for step in range(T_steps):
                step_idx = min(step, self.max_T - 1)
                step_vec = self.step_emb(torch.tensor(step_idx, device=idx.device)).unsqueeze(0).unsqueeze(0)
                x_norm = self.ln1(x + step_vec)
                Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                M = (1.0 - alpha) * torch.matmul(P, M) + alpha * I

                H = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + res_scale * self.out_proj(H)
                x = x + res_scale * self.mlp(self.ln2(x))

            x = self.ln_f(x)
            return self.head(x[:, -1, :])

    class Standard1Layer(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
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
            return self.head(x[:, -1, :])

    class StandardDeep(nn.Module):
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512, n_layers=3):
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
            self.head = nn.Linear(d_model, vocab_size, bias=False)

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

    # =========================================================================
    # EXPERIMENT 1: ZERO-SHOT LENGTH GENERALIZATION
    # (Train on 2-hop & 3-hop graphs -> Test on unseen 4-hop & 5-hop graphs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXP 1: ZERO-SHOT LENGTH GENERALIZATION (Train 2-3 hops -> Test 4-5 hops)")
    print("=" * 70)

    num_nodes = 32
    ARROW = num_nodes
    QUERY = num_nodes + 1
    vocab_size = num_nodes + 2

    def make_hops_data(n_samples, min_hops=2, max_hops=3):
        inputs, targets = [], []
        for _ in range(n_samples):
            hops = random.randint(min_hops, max_hops)
            nodes = random.sample(range(num_nodes), hops + 1)
            edges = [(nodes[i], nodes[i+1]) for i in range(hops)]
            random.shuffle(edges)
            seq = []
            for u, v in edges:
                seq.extend([u, ARROW, v])
            seq.extend([nodes[0], QUERY])
            inputs.append(seq)
            targets.append(nodes[-1])
        
        # Pad to max length
        max_l = 18  # 5 hops * 3 + 2 = 17
        padded = [s + [0]*(max_l - len(s)) for s in inputs]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    train_x, train_y = make_hops_data(40000, min_hops=2, max_hops=3)
    val_train_x, val_train_y = make_hops_data(3000, min_hops=2, max_hops=3)
    
    # Unseen 4-hop and 5-hop test sets
    test_4hop_x, test_4hop_y = make_hops_data(3000, min_hops=4, max_hops=4)
    test_5hop_x, test_5hop_y = make_hops_data(3000, min_hops=5, max_hops=5)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y), batch_size=128, shuffle=True
    )

    grav_gen = GravimemPro(vocab_size=vocab_size, max_seq_len=24, max_T=8).to(device)
    deep_gen = StandardDeep(vocab_size=vocab_size, max_seq_len=24, n_layers=3).to(device)
    std1_gen = Standard1Layer(vocab_size=vocab_size, max_seq_len=24).to(device)

    def train_simple(model, name, is_grav=False):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        for step, (bx, by) in enumerate(train_loader):
            if step > 2000:
                break
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx, T=3) if is_grav else model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    print("Training models on 2-3 hops...")
    train_simple(std1_gen, "Standard 1-Layer", is_grav=False)
    train_simple(grav_gen, "Gravimem-Pro", is_grav=True)
    train_simple(deep_gen, "Standard 3-Layer", is_grav=False)

    def eval_acc(model, tx, ty, T=None):
        model.eval()
        correct, total = 0, 0
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=256)
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx, T=T) if T is not None else model(bx)
                correct += (logits.argmax(dim=-1) == by).sum().item()
                total += by.size(0)
        return 100.0 * correct / total

    print("\n--- RESULTS: Length Generalization Accuracy ---")
    print(f"Standard 1-Layer (1 layer):  2-3 Hops: {eval_acc(std1_gen, val_train_x, val_train_y):5.2f}% | 4 Hops: {eval_acc(std1_gen, test_4hop_x, test_4hop_y):5.2f}% | 5 Hops: {eval_acc(std1_gen, test_5hop_x, test_5hop_y):5.2f}%")
    print(f"Standard 3-Layer (3 layers): 2-3 Hops: {eval_acc(deep_gen, val_train_x, val_train_y):5.2f}% | 4 Hops: {eval_acc(deep_gen, test_4hop_x, test_4hop_y):5.2f}% | 5 Hops: {eval_acc(deep_gen, test_5hop_x, test_5hop_y):5.2f}%")
    
    # Gravimem tested with dynamically expanding T at inference time!
    acc_g_train = eval_acc(grav_gen, val_train_x, val_train_y, T=3)
    acc_g_4hop = eval_acc(grav_gen, test_4hop_x, test_4hop_y, T=4)
    acc_g_5hop = eval_acc(grav_gen, test_5hop_x, test_5hop_y, T=5)
    print(f"Gravimem-Pro     (1 layer):  2-3 Hops: {acc_g_train:5.2f}% | 4 Hops (T=4): {acc_g_4hop:5.2f}% | 5 Hops (T=5): {acc_g_5hop:5.2f}%")

    # =========================================================================
    # EXPERIMENT 2: VARIABLE TRACKING / DATAFLOW DEPENDENCY CHAIN
    # x = 5, y = x + 3, z = y - 2, query: z -> 6
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXP 2: VARIABLE TRACKING / DATAFLOW COMPUTATION GRAPH")
    print("=" * 70)

    num_vars = 8
    num_vals = 16
    EQUAL = num_vars + num_vals
    PLUS = num_vars + num_vals + 1
    MINUS = num_vars + num_vals + 2
    Q_VAR = num_vars + num_vals + 3
    var_vocab = num_vars + num_vals + 4

    def make_var_data(n_samples, depth=3):
        inputs, targets = [], []
        for _ in range(n_samples):
            # v0 = val0
            # v1 = v0 + op1
            # v2 = v1 + op2 ...
            vars_chosen = random.sample(range(num_vars), depth + 1)
            val0 = random.randint(0, 5)
            curr_val = val0
            statements = []
            # First statement: var0 = val0
            statements.append([vars_chosen[0], EQUAL, num_vars + val0])
            for i in range(depth):
                op = random.choice([PLUS, MINUS])
                operand = random.randint(1, 3)
                if op == PLUS:
                    curr_val = min(num_vals - 1, curr_val + operand)
                else:
                    curr_val = max(0, curr_val - operand)
                statements.append([vars_chosen[i+1], EQUAL, vars_chosen[i], op, num_vars + operand])

            # Shuffle statement order
            random.shuffle(statements)
            seq = []
            for stmt in statements:
                seq.extend(stmt)
            seq.extend([vars_chosen[-1], Q_VAR])

            inputs.append(seq)
            targets.append(num_vars + curr_val)

        max_len = 28
        padded = [s + [0]*(max_len - len(s)) for s in inputs]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    vx_tr, vy_tr = make_var_data(40000, depth=3)
    vx_te, vy_te = make_var_data(5000, depth=3)

    v_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vx_tr, vy_tr), batch_size=128, shuffle=True
    )

    grav_var = GravimemPro(vocab_size=var_vocab, max_seq_len=32, max_T=6).to(device)
    std1_var = Standard1Layer(vocab_size=var_vocab, max_seq_len=32).to(device)
    deep_var = StandardDeep(vocab_size=var_vocab, max_seq_len=32, n_layers=4).to(device)

    def train_var(model, name, is_grav=False):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        for step, (bx, by) in enumerate(v_loader):
            if step > 2000:
                break
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx, T=4) if is_grav else model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    print("Training models on 3-step Variable Dataflow...")
    train_var(std1_var, "Standard 1-Layer", is_grav=False)
    train_var(grav_var, "Gravimem-Pro", is_grav=True)
    train_var(deep_var, "Standard 4-Layer", is_grav=False)

    print("\n--- RESULTS: Variable Dataflow Tracking Accuracy ---")
    print(f"1. Standard 1-Layer (208k params): Accuracy = {eval_acc(std1_var, vx_te, vy_te):5.2f}%")
    print(f"2. Gravimem-Pro      (208k params): Accuracy = {eval_acc(grav_var, vx_te, vy_te, T=4):5.2f}%")
    print(f"3. Standard 4-Layer (800k params): Accuracy = {eval_acc(deep_var, vx_te, vy_te):5.2f}%")

    # =========================================================================
    # EXPERIMENT 3: NEEDLE IN A HAYSTACK (Heavy Distractors Associative Recall)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  EXP 3: NEEDLE IN A HAYSTACK (Heavy Distractor Swarm Recall)")
    print("=" * 70)

    num_keys = 20
    num_vals = 20
    COLON = num_keys + num_vals
    ASK = num_keys + num_vals + 1
    hay_vocab = num_keys + num_vals + 2

    def make_haystack_data(n_samples, num_pairs=8):
        inputs, targets = [], []
        for _ in range(n_samples):
            keys = random.sample(range(num_keys), num_pairs)
            vals = [random.randint(0, num_vals - 1) for _ in range(num_pairs)]
            seq = []
            for k, v in zip(keys, vals):
                seq.extend([k, COLON, num_keys + v])
            target_idx = random.randint(0, num_pairs - 1)
            target_key = keys[target_idx]
            target_val = num_keys + vals[target_idx]
            seq.extend([target_key, ASK])
            inputs.append(seq)
            targets.append(target_val)

        max_len = num_pairs * 3 + 2
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    hx_tr, hy_tr = make_haystack_data(40000, num_pairs=8)
    hx_te, hy_te = make_haystack_data(5000, num_pairs=8)

    h_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(hx_tr, hy_tr), batch_size=128, shuffle=True
    )

    grav_hay = GravimemPro(vocab_size=hay_vocab, max_seq_len=32, max_T=4).to(device)
    std1_hay = Standard1Layer(vocab_size=hay_vocab, max_seq_len=32).to(device)

    def train_hay(model, is_grav=False):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        for step, (bx, by) in enumerate(h_loader):
            if step > 1500:
                break
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx, T=2) if is_grav else model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    print("Training models on Needle in a Haystack...")
    train_hay(std1_hay, is_grav=False)
    train_hay(grav_hay, is_grav=True)

    print("\n--- RESULTS: Needle in a Haystack Accuracy ---")
    print(f"1. Standard 1-Layer: Accuracy = {eval_acc(std1_hay, hx_te, hy_te):5.2f}%")
    print(f"2. Gravimem-Pro:     Accuracy = {eval_acc(grav_hay, hx_te, hy_te, T=2):5.2f}%")

    print("\n" + "=" * 80)
    print("  ALL ADVANCED REASONING EXPERIMENTS COMPLETE!")
    print("=" * 80)

    return {
        "length_gen": {
            "std1": [eval_acc(std1_gen, val_train_x, val_train_y), eval_acc(std1_gen, test_4hop_x, test_4hop_y), eval_acc(std1_gen, test_5hop_x, test_5hop_y)],
            "std3": [eval_acc(deep_gen, val_train_x, val_train_y), eval_acc(deep_gen, test_4hop_x, test_4hop_y), eval_acc(deep_gen, test_5hop_x, test_5hop_y)],
            "gravimem": [acc_g_train, acc_g_4hop, acc_g_5hop]
        },
        "variable_tracking": {
            "std1": eval_acc(std1_var, vx_te, vy_te),
            "gravimem": eval_acc(grav_var, vx_te, vy_te, T=4),
            "deep": eval_acc(deep_var, vx_te, vy_te)
        },
        "needle_in_haystack": {
            "std1": eval_acc(std1_hay, hx_te, hy_te),
            "gravimem": eval_acc(grav_hay, hx_te, hy_te, T=2)
        }
    }


@app.local_entrypoint()
def main():
    print("Launching Advanced Reasoning Benchmark Suite on Modal GPU...")
    res = run_advanced_suite.remote()
    print("\nAdvanced Benchmark Suite Complete!")
    print("Results:", res)
