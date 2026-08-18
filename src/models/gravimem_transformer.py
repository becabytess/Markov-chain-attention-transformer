"""
Gravimem Transformer v0 (Gravimem-Pro): PyTorch Implementation.
Replaces stacked physical layers with Markov Mass Settling iterations, Step Embeddings,
Learned Per-Head Teleportation, and a Shared Reusable MLP.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class GravimemMultiHeadMarkovBlock(nn.Module):
    """
    Multi-Head Gravimem-Pro Block with Markov Mass Settling.
    Features:
    - Pre-LayerNorm & 1/sqrt(T) Residual Highway Scaling
    - Step-Embedding Conditioning (Iteration-Aware Recurrence)
    - Learned Per-Head Teleportation Alpha (alpha_h = sigmoid(w_h))
    - Shared Reusable MLP
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        settling_steps: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.T = settling_steps

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-LN Normalizers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Step Embeddings: informs the shared block what iteration it is in
        self.step_emb = nn.Embedding(settling_steps, d_model)

        # Learned Teleportation Alpha per Head (initialized around alpha ~ 0.15)
        self.raw_alpha = nn.Parameter(torch.full((n_heads, 1, 1), -1.73))

        # Shared Reusable MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            causal_mask: Tensor of shape (seq_len, seq_len) with -inf for future tokens
        Returns:
            x_out: Updated token representations (batch, seq_len, d_model)
            M: Final settled mass matrix (batch, heads, seq_len, seq_len)
        """
        B, L, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        # 1. Initialize Mass Matrix M = Identity
        I = torch.eye(L, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        M = I.expand(B, H, L, L)
        alpha = torch.sigmoid(self.raw_alpha)  # (H, 1, 1)

        # Residual highway scale
        res_scale = 1.0 / math.sqrt(self.T)

        for step in range(self.T):
            # Inject Step Embedding for iteration-aware features
            step_vec = self.step_emb(torch.tensor(step, device=x.device)).unsqueeze(0).unsqueeze(0)
            x_step = x + step_vec

            # Pre-LN
            x_norm = self.ln1(x_step)
            Q = self.q_proj(x_norm).view(B, L, H, d_k).transpose(1, 2)
            K = self.k_proj(x_norm).view(B, L, H, d_k).transpose(1, 2)
            V = self.v_proj(x_norm).view(B, L, H, d_k).transpose(1, 2)

            # Causal Markov Transition Matrix P
            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
            if causal_mask is not None:
                scores = scores + causal_mask

            P = F.softmax(scores, dim=-1)
            P_drop = self.dropout(P)

            # Causal Markov Mass Settling: M^(t+1) = (1 - alpha) * P @ M^(t) + alpha * I
            M = (1.0 - alpha) * torch.matmul(P_drop, M) + alpha * I

            # Value Mixing with Mass Matrix
            H_mixed = torch.matmul(M, V).transpose(1, 2).contiguous().view(B, L, D)

            # Residual Highway Updates
            x = x + res_scale * self.dropout(self.out_proj(H_mixed))
            x = x + res_scale * self.mlp(self.ln2(x))

        return x, M


class GravimemTransformerLM(nn.Module):
    """
    Autoregressive Language Model powered by Gravimem-Pro Markov Dynamics.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        settling_steps: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Single Gravimem-Pro Recurrent Block
        self.block = GravimemMultiHeadMarkovBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            settling_steps=settling_steps,
            dropout=dropout
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, L = idx.shape
        assert L <= self.max_seq_len, f"Sequence length {L} exceeds max {self.max_seq_len}"

        pos = torch.arange(0, L, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        c_mask = self.causal_mask[:L, :L]
        x, M = self.block(x, causal_mask=c_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, M

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

