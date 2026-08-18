"""
Core Neural Layers for the Gravimem Gated Surfer Architecture.
Supports both:
1. PositionalJumpAttention (Sub-quadratic O(L * K) Multi-Scale Jump Routing)
2. MarkovAttention (Continuous Causal Markov Transition Attention)
3. GatedSurferBackpack (Recurrent GRU trajectory memory cell)
"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalJumpAttention(nn.Module):
    """
    Sub-Quadratic O(L * K) Positional Jump Routing.
    Instead of computing all-to-all O(L^2) attention, the surfer at position i
    predicts a dynamic probability distribution over K multi-scale jump offsets:
        offsets: [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128, 256, ...]
    Eliminates all-to-all attention dust and scales linearly with sequence length!
    """
    def __init__(self, d_model: int, jump_offsets: List[int] = None):
        super().__init__()
        if jump_offsets is None:
            # Default Fibonacci multi-scale jump menu
            jump_offsets = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 127]

        self.jump_offsets = jump_offsets
        self.num_jumps = len(jump_offsets)
        self.d_model = d_model

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.jump_policy = nn.Linear(d_model, self.num_jumps)

    def forward(self, x_norm: torch.Tensor, current_state: torch.Tensor):
        """
        Args:
            x_norm: (B, L, d_model) normalized token representations
            current_state: (B*L, d_model) current backpack states of surfers
        Returns:
            gathered_proj: (B*L, d_model) context gathered from chosen hops
            jump_probs: (B, L, K) distribution over jump choices
        """
        B, L, d = x_norm.shape
        device = x_norm.device

        # Pre-compute value vectors
        V = self.v_proj(x_norm)  # (B, L, d)

        # Build index mapping tensor for all K candidate jumps: (L, K)
        target_indices = torch.zeros((L, self.num_jumps), dtype=torch.long, device=device)
        valid_jump_mask = torch.zeros((L, self.num_jumps), dtype=torch.bool, device=device)
        for i in range(L):
            for k, offset in enumerate(self.jump_offsets):
                target_pos = i - offset
                if target_pos >= 0:
                    target_indices[i, k] = target_pos
                    valid_jump_mask[i, k] = True
                else:
                    target_indices[i, k] = 0
                    valid_jump_mask[i, k] = False

        # Vectorized gather across K positions: (B, L, K, d)
        gathered_candidates = V[:, target_indices]

        # Surfer state predicts jump probability distribution
        jump_logits = self.jump_policy(current_state).view(B, L, self.num_jumps)
        jump_logits = jump_logits.masked_fill(~valid_jump_mask.unsqueeze(0), float('-inf'))
        jump_probs = F.softmax(jump_logits, dim=-1)  # (B, L, K)

        # Weighted sum over K candidate landing sites
        gathered_V = torch.sum(jump_probs.unsqueeze(-1) * gathered_candidates, dim=2)  # (B, L, d)
        gathered_proj = self.out_proj(gathered_V).view(-1, d)  # (B*L, d)

        return gathered_proj, jump_probs


class MarkovAttention(nn.Module):
    """
    Multi-Head Causal Markov Transition Attention (O(L^2) dense mode).
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
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init))))

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None):
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
        s^(t+1) = GRUCell(gathered_context, s^(t))
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRUCell(d_model, d_model)

    def forward(self, gathered_context: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        return self.gru(gathered_context, current_state)


class GravimemBlock(nn.Module):
    """
    Unified Gravimem Layer Block supporting both Positional Jump Routing (O(L*K))
    and Dense Markov Attention (O(L^2)).
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_mlp: int = None,
        default_T: int = 4,
        routing_mode: str = "jump",
        jump_offsets: List[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.default_T = default_T
        self.routing_mode = routing_mode
        if d_mlp is None:
            d_mlp = 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)

        if routing_mode == "jump":
            self.jump_attn = PositionalJumpAttention(d_model, jump_offsets=jump_offsets)
        else:
            self.dense_attn = MarkovAttention(d_model, n_heads)

        self.backpack = GatedSurferBackpack(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model)
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None, T: int = None, return_all_steps: bool = False):
        if T is None:
            T = self.default_T

        B, L, _ = x.shape
        x_norm = self.ln1(x)

        s = x.view(-1, self.d_model)  # (B*L, d_model)
        all_states = [x]

        if self.routing_mode == "jump":
            for step in range(T):
                gathered_proj, _ = self.jump_attn(x_norm, current_state=s)
                s = self.backpack(gathered_proj, s)
                all_states.append(s.view(B, L, self.d_model))
        else:
            P, V = self.dense_attn(x_norm, causal_mask=causal_mask)
            for step in range(T):
                gathered = torch.einsum('bhij,bhjd->bhid', P, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
                gathered_proj = self.dense_attn.out(gathered).view(-1, self.d_model)
                s = self.backpack(gathered_proj, s)
                all_states.append(s.view(B, L, self.d_model))

        if return_all_steps:
            final_outputs = []
            for state_t in all_states[1:]:
                out_t = state_t + self.mlp(self.ln2(state_t))
                final_outputs.append(out_t)
            return final_outputs

        final_state = all_states[-1]
        out = final_state + self.mlp(self.ln2(final_state))
        return out
