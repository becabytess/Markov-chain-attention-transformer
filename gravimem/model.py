"""
Full Gravimem Model Architectures (Language Model & Classifier).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from gravimem.layers import GravimemBlock


class GravimemLM(nn.Module):
    """
    Autoregressive Gravimem Language Model.
    Supports:
    - routing_mode="jump": Sub-quadratic O(L * K) Positional Jump Surfer (Default, Best Quality)
    - routing_mode="dense": Causal Dense Markov Attention (O(L^2))
    - Dynamic anytime thought unrolling across T hops
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        default_T: int = 4,
        d_mlp: int = 512,
        routing_mode: str = "jump",
        jump_offsets: List[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.default_T = default_T
        self.routing_mode = routing_mode

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            GravimemBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                default_T=default_T,
                routing_mode=routing_mode,
                jump_offsets=jump_offsets
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)

    def forward(self, idx: torch.Tensor, T: int = None, return_all_steps: bool = False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal_mask = self.get_causal_mask(L, idx.device) if self.routing_mode == "dense" else None

        if return_all_steps and self.n_layers == 1:
            step_states = self.layers[0](x, causal_mask=causal_mask, T=T, return_all_steps=True)
            return [self.head(self.ln_f(s)) for s in step_states]

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask, T=T)

        logits = self.head(self.ln_f(x))
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None, T: int = None):
        """
        Autoregressive text generation with customizable surfing depth T.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits = self(idx_cond, T=T)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


class GravimemClassifier(nn.Module):
    """
    Gravimem Sequence / Graph Reasoning Classifier.
    """
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        default_T: int = 4,
        routing_mode: str = "jump",
        jump_offsets: List[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            GravimemBlock(
                d_model=d_model,
                n_heads=n_heads,
                default_T=default_T,
                routing_mode=routing_mode,
                jump_offsets=jump_offsets
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, idx: torch.Tensor, T: int = None):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x, causal_mask=None, T=T)

        return self.head(self.ln_f(x[:, -1, :]))
