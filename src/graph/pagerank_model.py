"""
Semantic PageRank Personality Space Model.
Builds the concept graph from semantic embeddings and computes Personalized PageRank mass
based on user search activity and graph connectivity.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from src.config import config
from src.ingestion.takeout_parser import TakeoutParser
from src.embeddings.embedder import Embedder


class SemanticPageRankSpace:
    """
    Computes graph centrality and Personalized PageRank mass over semantic concept nodes.
    """

    def __init__(
        self,
        labels: List[str],
        frequencies: np.ndarray,
        embeddings: np.ndarray,
        damping: float = 0.85,
        sim_threshold: float = 0.45,
        power: float = 1.5
    ):
        self.labels = labels
        self.N = len(labels)
        self.dim = embeddings.shape[1]
        self.damping = damping
        self.sim_threshold = sim_threshold
        self.power = power

        # Normalize embeddings to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.X0 = (embeddings / norms).astype(np.float32)

        # Personalization teleport vector based on user query frequencies
        freq_sum = np.sum(frequencies)
        if freq_sum > 0:
            self.v = (frequencies / freq_sum).astype(np.float32)
        else:
            self.v = np.ones(self.N, dtype=np.float32) / self.N

        self.W = None
        self.P = None
        self.pi = None

    def build_transition_matrix(self):
        """Constructs the semantic adjacency and row-stochastic transition matrix."""
        print(f"\n[1/3] Computing pairwise semantic similarity matrix ({self.N}x{self.N})...")
        S = np.dot(self.X0, self.X0.T)
        np.fill_diagonal(S, 0.0)

        print(f"[2/3] Applying threshold (>{self.sim_threshold}) and constructing transition graph...")
        # Positive affinity weights
        affinity = np.maximum(0.0, S - self.sim_threshold) ** self.power
        self.W = affinity

        # Row-normalize to create Markov transition probability matrix P
        row_sums = np.sum(affinity, axis=1, keepdims=True)
        
        # Handle dangling nodes (nodes with 0 outgoing edges teleport via v)
        dangling = (row_sums[:, 0] == 0)
        row_sums[dangling] = 1.0
        
        self.P = affinity / row_sums
        self.dangling_mask = dangling
        
        # Report graph density
        num_edges = np.count_nonzero(affinity)
        avg_degree = num_edges / self.N
        print(f" -> Graph built: {num_edges:,} semantic edges (Average degree: {avg_degree:.1f} connections/concept)")

    def compute_pagerank(self, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
        """Runs power iteration to solve the stationary distribution pi = d * P^T * pi + (1-d) * v."""
        if self.P is None:
            self.build_transition_matrix()

        print(f"\n[3/3] Running PageRank Power Iteration (damping = {self.damping})...")
        
        # Initialize uniformly
        pi = np.ones(self.N, dtype=np.float32) / self.N
        
        with tqdm(total=max_iter, desc="PageRank Power Iteration") as pbar:
            for iteration in range(max_iter):
                # Handle dangling nodes mass
                dangling_sum = np.sum(pi[self.dangling_mask])
                
                # Power iteration step: d * (P^T @ pi + dangling_sum * v) + (1 - d) * v
                pi_next = self.damping * (np.dot(self.P.T, pi) + dangling_sum * self.v) + (1.0 - self.damping) * self.v
                
                # Check convergence
                diff = np.linalg.norm(pi_next - pi, ord=1)
                pi = pi_next
                pbar.update(1)
                pbar.set_postfix({"L1 delta": f"{diff:.2e}"})
                
                if diff < tol:
                    print(f" -> Converged in {iteration + 1} iterations! (tolerance = {tol})")
                    break

        self.pi = pi
        return pi

    def get_ranked_concepts(self, top_k: int = 30) -> Dict[str, Any]:
        """Returns top and bottom concepts ranked by PageRank personality mass."""
        if self.pi is None:
            self.compute_pagerank()

        # Rescale PageRank mass to intuitive multiplier (mean = 1.0)
        norm_mass = self.pi * self.N
        sorted_indices = np.argsort(-self.pi)

        top_concepts = [
            {
                "rank": r + 1,
                "label": self.labels[idx],
                "pagerank_prob": float(self.pi[idx]),
                "relative_mass": float(norm_mass[idx]),
                "connections": int(np.count_nonzero(self.W[idx]))
            }
            for r, idx in enumerate(sorted_indices[:top_k])
        ]

        bottom_concepts = [
            {
                "rank": self.N - r,
                "label": self.labels[idx],
                "pagerank_prob": float(self.pi[idx]),
                "relative_mass": float(norm_mass[idx]),
                "connections": int(np.count_nonzero(self.W[idx]))
            }
            for r, idx in enumerate(reversed(sorted_indices[-top_k:]))
        ]

        return {
            "top_concepts": top_concepts,
            "bottom_concepts": bottom_concepts,
            "mass_array": norm_mass
        }
