"""
Stationary Markov Diffusion & Dynamic Pondering Suite on Modal Cloud GPU.
Tests:
1. Pure Stationary Operator (Zero-shot length generalization to 4, 5, 6, 7 hops without step-ID overfitting)
2. Dynamic Cauchy Halting (Surfer equilibrium halting when ||M^(t+1) - M^(t)||_1 < epsilon)
3. Dyck-2 Balanced Parentheses Hierarchy Parsing
"""

import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "tqdm")
)

app = modal.App("gravimem-stationary-pondering", image=image)


@app.function(gpu="T4", timeout=3600)
def run_pondering_suite():
    import math
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("=" * 80)
    print("  GRAVIMEM: STATIONARY MARKOV DIFFUSION & DYNAMIC PONDERING SUITE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Pure Stationary Gravimem (No Step-ID Bias for Generalization)
    # -------------------------------------------------------------
    class StationaryGravimem(nn.Module):
        """
        Pure Stationary Markov Operator:
        Transitions and MLP are stationary (no step embeddings),
        enabling zero-shot expansion of T at test time!
        """
        def __init__(self, vocab_size, max_seq_len=64, d_model=128, n_heads=4, d_mlp=512):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)

            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_mlp),
                nn.GELU(),
                nn.Linear(d_mlp, d_model)
            )

            # Per-head teleportation
            self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx, T_steps=3, eps_halt=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)

            I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            M = I.expand(B, self.n_heads, L, L)
            alpha = torch.sigmoid(self.raw_alpha)

            actual_steps = 0
            for step in range(T_steps):
                actual_steps += 1
                x_norm = self.ln1(x)
                Q = self.q_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.k_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                V = self.v_proj(x_norm).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

                P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                M_next = (1.0 - alpha) * torch.matmul(P, M) + alpha * I

                # Optional Cauchy Halting Condition
                if eps_halt is not None and step > 0:
                    delta = torch.norm(M_next - M, p=1, dim=(-2, -1)).mean().item()
                    if delta < eps_halt:
                        M = M_next
                        break

                M = M_next
                H = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, -1)
                
                # Stationary residual update (scale by 0.5 for stability)
                x = x + 0.5 * self.out_proj(H)
                x = x + 0.5 * self.mlp(self.ln2(x))

            x = self.ln_f(x)
            return self.head(x[:, -1, :]), actual_steps

    # =========================================================================
    # TASK 1: TRUE ZERO-SHOT LENGTH GENERALIZATION ON MULTI-HOP GRAPHS
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TASK 1: PURE STATIONARY MULTI-HOP GENERALIZATION (2-3 hops -> 4-6 hops)")
    print("=" * 70)

    num_nodes = 32
    ARROW = num_nodes
    QUERY = num_nodes + 1
    vocab_size = num_nodes + 2

    def make_hops(n_samples, min_hops=2, max_hops=3):
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
        
        max_l = 22  # 6 hops * 3 + 2 = 20
        padded = [s + [0]*(max_l - len(s)) for s in inputs]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    # Train on 2-3 hops
    tr_x, tr_y = make_hops(60000, min_hops=2, max_hops=3)
    val_23_x, val_23_y = make_hops(3000, min_hops=2, max_hops=3)
    
    # Test on unseen 4, 5, 6 hops
    te_4_x, te_4_y = make_hops(3000, min_hops=4, max_hops=4)
    te_5_x, te_5_y = make_hops(3000, min_hops=5, max_hops=5)
    te_6_x, te_6_y = make_hops(3000, min_hops=6, max_hops=6)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tr_x, tr_y), batch_size=128, shuffle=True
    )

    stat_model = StationaryGravimem(vocab_size=vocab_size, max_seq_len=24).to(device)
    opt = torch.optim.AdamW(stat_model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Multi-step training (10 epochs = 4680 steps)
    print("Training Stationary Gravimem for 3000 steps on 2-3 hops...")
    step = 0
    stat_model.train()
    for epoch in range(10):
        for bx, by in loader:
            step += 1
            if step > 3000:
                break
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            # Randomize T during training (T in [2, 3]) for scale invariance!
            T_train = random.choice([2, 3])
            logits, _ = stat_model(bx, T_steps=T_train)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stat_model.parameters(), 1.0)
            opt.step()
        if step > 3000:
            break

    def test_generalization(model, tx, ty, T):
        model.eval()
        correct, total = 0, 0
        l = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=256)
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx, T_steps=T)
                correct += (logits.argmax(dim=-1) == by).sum().item()
                total += by.size(0)
        return 100.0 * correct / total

    print("\n--- RESULTS: Pure Stationary Zero-Shot Generalization ---")
    acc_23 = test_generalization(stat_model, val_23_x, val_23_y, T=3)
    acc_4 = test_generalization(stat_model, te_4_x, te_4_y, T=4)
    acc_5 = test_generalization(stat_model, te_5_x, te_5_y, T=5)
    acc_6 = test_generalization(stat_model, te_6_x, te_6_y, T=6)

    print(f"Trained on 2-3 hops (T=3):   Accuracy = {acc_23:5.2f}%")
    print(f"Zero-Shot 4 hops    (T=4):   Accuracy = {acc_4:5.2f}%")
    print(f"Zero-Shot 5 hops    (T=5):   Accuracy = {acc_5:5.2f}%")
    print(f"Zero-Shot 6 hops    (T=6):   Accuracy = {acc_6:5.2f}%")

    # =========================================================================
    # TASK 2: DYNAMIC PONDERING (Cauchy Equilibrium Halting)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TASK 2: DYNAMIC CAUCHY EQUILIBRIUM PONDERING")
    print("=" * 70)

    def test_pondering(model, tx, ty, eps=0.01, max_T=8):
        model.eval()
        correct, total = 0, 0
        steps_list = []
        l = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=128)
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                logits, steps = model(bx, T_steps=max_T, eps_halt=eps)
                correct += (logits.argmax(dim=-1) == by).sum().item()
                total += by.size(0)
                steps_list.append(steps)
        avg_steps = sum(steps_list) / len(steps_list)
        acc = 100.0 * correct / total
        return acc, avg_steps

    acc_p_23, steps_23 = test_pondering(stat_model, val_23_x, val_23_y, eps=0.05)
    acc_p_4, steps_4 = test_pondering(stat_model, te_4_x, te_4_y, eps=0.05)
    acc_p_5, steps_5 = test_pondering(stat_model, te_5_x, te_5_y, eps=0.05)

    print(f"2-3 Hops with Adaptive Halting: Acc = {acc_p_23:5.2f}% | Avg Steps = {steps_23:.2f}")
    print(f"4   Hops with Adaptive Halting: Acc = {acc_p_4:5.2f}% | Avg Steps = {steps_4:.2f}")
    print(f"5   Hops with Adaptive Halting: Acc = {acc_p_5:5.2f}% | Avg Steps = {steps_5:.2f}")

    # =========================================================================
    # TASK 3: DYCK-2 BALANCED PARENTHESES HIERARCHICAL PARSING
    # Alphabet: '(', ')', '[', ']'
    # =========================================================================
    print("\n" + "=" * 70)
    print("  TASK 3: DYCK-2 FORMAL GRAMMAR & NESTED HIERARCHY PARSING")
    print("=" * 70)

    # 1: '(', 2: ')', 3: '[', 4: ']'
    PAIRS = {1: 2, 3: 4}
    OPEN_CHARS = [1, 3]

    def gen_dyck(max_depth=4, length=12):
        # Generate valid or invalid Dyck-2 string of fixed length
        is_valid = random.choice([True, False])
        if is_valid:
            # Build recursively valid
            def make_valid(rem_len):
                if rem_len <= 0:
                    return []
                # Choose pair
                op = random.choice(OPEN_CHARS)
                cl = PAIRS[op]
                inner_len = random.choice([i for i in range(0, rem_len, 2)])
                inner = make_valid(inner_len)
                rest = make_valid(rem_len - 2 - inner_len)
                return [op] + inner + [cl] + rest
            seq = make_valid(length)
            if len(seq) < length:
                seq += [0] * (length - len(seq))
            return seq[:length], 1
        else:
            # Random corrupt sequence
            seq = [random.randint(1, 4) for _ in range(length)]
            # Check if accidentally valid
            stack = []
            valid = True
            for ch in seq:
                if ch in OPEN_CHARS:
                    stack.append(ch)
                elif ch in PAIRS.values():
                    if not stack or PAIRS[stack.pop()] != ch:
                        valid = False
                        break
            if valid and len(stack) == 0:
                target = 1
            else:
                target = 0
            return seq, target

    def make_dyck_dataset(n_samples, length=12):
        xs, ys = [], []
        for _ in range(n_samples):
            x, y = gen_dyck(length=length)
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    dx_tr, dy_tr = make_dyck_dataset(40000, length=12)
    dx_val, dy_val = make_dyck_dataset(5000, length=12)
    dx_deep, dy_deep = make_dyck_dataset(5000, length=20)  # Generalization to length 20!

    d_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dx_tr, dy_tr), batch_size=128, shuffle=True
    )

    # Classification head for binary validity (vocab_size=6, binary out)
    class DyckClassifier(nn.Module):
        def __init__(self, is_gravimem=True, d_model=128, T=4):
            super().__init__()
            self.is_grav = is_gravimem
            self.T = T
            self.tok_emb = nn.Embedding(8, d_model)
            self.pos_emb = nn.Embedding(32, d_model)
            self.d_k = d_model // 4
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = nn.Linear(d_model, d_model, bias=False)
            self.v = nn.Linear(d_model, d_model, bias=False)
            self.out = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, d_model))
            self.head = nn.Linear(d_model, 2)

        def forward(self, idx, T=None):
            B, L = idx.shape
            pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            steps = T if T is not None else self.T

            if self.is_grav:
                I = torch.eye(L, device=idx.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
                M = I.expand(B, 4, L, L)
                for _ in range(steps):
                    x_n = self.ln1(x)
                    Q = self.q(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                    K = self.k(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                    V = self.v(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                    P = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                    M = 0.85 * (P @ M) + 0.15 * I
                    H = (M @ V).transpose(1, 2).contiguous().view(B, L, -1)
                    x = x + 0.5 * self.out(H)
                    x = x + 0.5 * self.mlp(self.ln2(x))
            else:
                x_n = self.ln1(x)
                Q = self.q(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                K = self.k(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                V = self.v(x_n).view(B, L, 4, self.d_k).transpose(1, 2)
                att = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
                H = (att @ V).transpose(1, 2).contiguous().view(B, L, -1)
                x = x + self.out(H)
                x = x + self.mlp(self.ln2(x))

            # Pool representation across sequence
            return self.head(x.mean(dim=1))

    dyck_grav = DyckClassifier(is_gravimem=True, T=4).to(device)
    dyck_std = DyckClassifier(is_gravimem=False).to(device)

    def train_dyck(model):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        for bx, by in d_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    print("Training Dyck-2 grammar parsers on length 12...")
    for _ in range(5):
        train_dyck(dyck_std)
        train_dyck(dyck_grav)

    def eval_dyck(model, tx, ty, T=None):
        model.eval()
        correct, total = 0, 0
        l = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=256)
        with torch.no_grad():
            for bx, by in l:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx, T=T)
                correct += (logits.argmax(dim=-1) == by).sum().item()
                total += by.size(0)
        return 100.0 * correct / total

    print("\n--- RESULTS: Dyck-2 Formal Language Parsing Accuracy ---")
    print(f"Standard 1-Layer: Length 12 (in-distribution): {eval_dyck(dyck_std, dx_val, dy_val):5.2f}% | Length 20 (OOD Deep Nesting): {eval_dyck(dyck_std, dx_deep, dy_deep):5.2f}%")
    print(f"Gravimem (T=4):   Length 12 (in-distribution): {eval_dyck(dyck_grav, dx_val, dy_val, T=4):5.2f}% | Length 20 (OOD Deep Nesting): {eval_dyck(dyck_grav, dx_deep, dy_deep, T=6):5.2f}%")

    print("\n" + "=" * 80)
    print("  STATIONARY & PONDERING SUITE COMPLETE!")
    print("=" * 80)

    return {
        "stationary_gen": {"2-3hops": acc_23, "4hops": acc_4, "5hops": acc_5, "6hops": acc_6},
        "adaptive_halting": {"2-3hops": [acc_p_23, steps_23], "4hops": [acc_p_4, steps_4], "5hops": [acc_p_5, steps_5]},
        "dyck2": {
            "std1_len12": eval_dyck(dyck_std, dx_val, dy_val),
            "std1_len20": eval_dyck(dyck_std, dx_deep, dy_deep),
            "grav_len12": eval_dyck(dyck_grav, dx_val, dy_val, T=4),
            "grav_len20": eval_dyck(dyck_grav, dx_deep, dy_deep, T=6)
        }
    }


@app.local_entrypoint()
def main():
    print("Launching Stationary & Pondering Suite on Modal GPU...")
    res = run_pondering_suite.remote()
    print("\nPondering Suite Complete!")
    print("Results:", res)
