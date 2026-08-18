"""
Full Gravimem Model Architectures (Language Model & Classifier) with Fused Accelerated Engine.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from gravimem.layers import GravimemBlock, FusedPositionalJumpSurfer


class GravimemLM(nn.Module):
    """
    Autoregressive Gravimem Language Model powered by the Fused Jump Surfer Engine.
    Sub-quadratic O(L * K) complexity with dynamic test-time thought depth (T).
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        d_model: int = 128,
        n_layers: int = 1,
        default_T: int = 4,
        d_mlp: Optional[int] = None,
        jump_offsets: Optional[List[int]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.default_T = default_T

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 16, d_model)

        self.layers = nn.ModuleList([
            GravimemBlock(
                d_model=d_model,
                max_seq_len=max_seq_len,
                d_mlp=d_mlp,
                default_T=default_T,
                jump_offsets=jump_offsets
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # Weight tying

    def forward(self, idx: torch.Tensor, T: Optional[int] = None, return_all_steps: bool = False):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        if return_all_steps and self.n_layers == 1:
            step_states = self.layers[0](x, T=T, return_all_steps=True)
            return [self.head(self.ln_f(s)) for s in step_states]

        for layer in self.layers:
            x = layer(x, T=T)

        logits = self.head(self.ln_f(x))
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        T: Optional[int] = None
    ):
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
    Gravimem Sequence / Graph Reasoning Classifier with Fused Engine.
    """
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = 128,
        n_layers: int = 1,
        default_T: int = 4,
        jump_offsets: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 16, d_model)

        self.layers = nn.ModuleList([
            GravimemBlock(
                d_model=d_model,
                max_seq_len=max_seq_len,
                default_T=default_T,
                jump_offsets=jump_offsets
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, idx: torch.Tensor, T: Optional[int] = None):
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x, T=T)

        return self.head(self.ln_f(x[:, -1, :]))
