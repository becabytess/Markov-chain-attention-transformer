"""
Core Neural Layers for the Gravimem Gated Surfer Architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkovAttention(nn.Module):
    """
    Multi-Head Causal Markov Transition Attention.
    Computes row-stochastic transition probability matrix P:
        P_ij = Softmax(Q_i K_j^T / sqrt(d_k) + causal_mask)
    Supports both fluid continuous diffusion (M @ V) and stateful surfer trajectory routing.
    """
    def __init__(self, d_model: int, n_heads: int, alpha_init: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Learned teleportation prior logit (sigmoid(alpha) in (0, 1))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None):
        """
        Args:
            x: (B, L, d_model)
            causal_mask: (L, L) optional causal mask containing float('-inf') on upper triangle
        Returns:
            P: (B, n_heads, L, L) Transition probability matrix
            V: (B, n_heads, L, d_k) Value representations
        """
        B, L, _ = x.shape
        Q = self.q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask is not None:
            scores = scores + causal_mask[:L, :L]

        P = F.softmax(scores, dim=-1)
        return P, V


class GatedSurferBackpack(nn.Module):
    """
    Recurrent Gated Backpack Memory Cell for trajectory accumulation:
        V_gather = sum_j P_ij V_j
        s^(t+1) = GRUCell(W_out(V_gather), s^(t))
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRUCell(d_model, d_model)

    def forward(self, gathered_v: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gathered_v: (B*L, d_model) Projected gathered context from hop
            current_state: (B*L, d_model) Current backpack state vector
        Returns:
            new_state: (B*L, d_model) Updated backpack state vector
        """
        return self.gru(gathered_v, current_state)


class GravimemBlock(nn.Module):
    """
    Unified Gravimem Layer Block:
    1. LayerNorm + QKV Projection -> Transition Map P
    2. Recurrent Surfing Loop across T steps with Gated Backpack Memory
    3. Post-Settling FeedForward Network (MLP) + Residual Connection
    """
    def __init__(self, d_model: int, n_heads: int, d_mlp: int = None, default_T: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.default_T = default_T
        if d_mlp is None:
            d_mlp = 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MarkovAttention(d_model, n_heads)
        self.backpack = GatedSurferBackpack(d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None, T: int = None, return_all_steps: bool = False):
        """
        Args:
            x: (B, L, d_model) input representations
            causal_mask: (L, L) optional causal mask
            T: Number of reasoning / surfing hops (defaults to self.default_T)
            return_all_steps: If True, returns list of states [s^(0), s^(1), ..., s^(T)]
        Returns:
            out: (B, L, d_model) output representations (or list of outputs if return_all_steps is True)
        """
        if T is None:
            T = self.default_T

        B, L, _ = x.shape
        x_norm = self.ln1(x)
        P, V = self.attn(x_norm, causal_mask=causal_mask)

        # Surfer starts at initial embedding position
        s = x.view(-1, self.d_model)  # (B*L, d_model)
        all_states = [x]

        for step in range(T):
            gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
            gathered_proj = self.attn.out(gathered).view(-1, self.d_model)
            s = self.backpack(gathered_proj, s)
            s_reshaped = s.view(B, L, self.d_model)
            all_states.append(s_reshaped)

        if return_all_steps:
            final_outputs = []
            for state_t in all_states[1:]:
                out_t = state_t + self.mlp(self.ln2(state_t))
                final_outputs.append(out_t)
            return final_outputs

        # Standard final state forward pass
        final_state = all_states[-1]
        out = final_state + self.mlp(self.ln2(final_state))
        return out
