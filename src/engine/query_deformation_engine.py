"""
Gravimem V0: Query-Deformation Semantic Memory Engine.
Implements the clean mathematical formulation:
- Frozen Canonical Reference Space X (x_i = E(c_i))
- Semantic Graph & Stationary PageRank prior p (sum(p) = 1)
- Query deformation via personal gravitational field: q* = normalize(q0 + eta * sum(m_i * K(sim) * (x_i - q0)))
- Fast query-driven mass reinforcement (sum(m) = 1)
- Slow EMA graph restructuring fusion: m_new = alpha * p_new + (1 - alpha) * m_old (alpha = 0.5)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class QueryEngineConfig:
    eta: float = 0.25           # Personalization coefficient (query deflection magnitude)
    sim_threshold: float = 0.40 # Similarity threshold for neighborhood kernel K
    kernel_power: float = 2.0   # Non-linear weighting power for K
    reinforce_rate: float = 0.15 # Fast mass reinforcement weight per query event
    damping: float = 0.85       # PageRank damping factor
    alpha_fusion: float = 0.5   # EMA fusion weight between structural PageRank and active mass


class QueryDeformationEngine:
    def __init__(
        self,
        labels: List[str],
        embeddings: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        config: Optional[QueryEngineConfig] = None
    ):
        self.labels = labels
        self.N = len(labels)
        self.dim = embeddings.shape[1]
        self.cfg = config or QueryEngineConfig()

        # 1. Canonical Reference Space (Unit Normalized, Frozen)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.X = (embeddings / norms).astype(np.float32)

        # 2. Activity Teleport Vector
        if frequencies is not None and np.sum(frequencies) > 0:
            self.v = (frequencies / np.sum(frequencies)).astype(np.float32)
        else:
            self.v = np.ones(self.N, dtype=np.float32) / self.N

        # Compute initial Structural PageRank p
        self.p = self.compute_pagerank()
        
        # 3. Active Personal Influence Distribution (sum to 1)
        self.m = self.p.copy()

    def _kernel(self, similarities: np.ndarray) -> np.ndarray:
        """K(sim) weighting function: zero below threshold, smooth non-linear above."""
        rel = np.maximum(0.0, similarities - self.cfg.sim_threshold) / (1.0 - self.cfg.sim_threshold + 1e-6)
        return (rel ** self.cfg.kernel_power).astype(np.float32)

    def compute_pagerank(self, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
        """Computes stationary distribution p on the semantic graph (sum(p) = 1)."""
        # Pairwise similarities
        S = np.dot(self.X, self.X.T)
        np.fill_diagonal(S, 0.0)

        # Graph adjacency weights
        W = self._kernel(S)
        row_sums = np.sum(W, axis=1, keepdims=True)
        
        dangling = (row_sums[:, 0] == 0)
        row_sums[dangling] = 1.0
        P = W / row_sums

        p = np.ones(self.N, dtype=np.float32) / self.N
        d = self.cfg.damping

        for _ in range(max_iter):
            dangling_sum = np.sum(p[dangling])
            p_next = d * (np.dot(P.T, p) + dangling_sum * self.v) + (1.0 - d) * self.v
            if np.linalg.norm(p_next - p, ord=1) < tol:
                break
            p = p_next

        # Enforce sum to 1
        p = p / np.sum(p)
        return p.astype(np.float32)

    def deform_query(self, q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes personalized query vector q* from raw query vector q0:
        Delta q = eta * sum_i m_i * K(sim(q0, x_i)) * (x_i - q0)
        q* = normalize(q0 + Delta q)
        """
        q0_norm = q0 / (np.linalg.norm(q0) + 1e-8)
        
        # Raw similarities: (N,)
        sims = np.dot(self.X, q0_norm)
        K_sims = self._kernel(sims)
        
        # Pull weights for each concept: w_i = m_i * K(sim(q0, x_i))
        pull_weights = self.m * K_sims  # (N,)
        
        # Vectorized directional delta:
        # sum_i pull_weights[i] * (X[i] - q0) = (pull_weights @ X) - sum(pull_weights) * q0
        weighted_X = np.dot(pull_weights, self.X)
        total_weight = np.sum(pull_weights)
        delta_q = self.cfg.eta * (weighted_X - total_weight * q0_norm)
        
        # Personalized query
        q_star_unnorm = q0_norm + delta_q
        q_star = q_star_unnorm / (np.linalg.norm(q_star_unnorm) + 1e-8)
        
        # Angular deflection in degrees
        cos_deflect = np.clip(np.dot(q0_norm, q_star), -1.0, 1.0)
        deflection_deg = float(np.arccos(cos_deflect) * (180.0 / np.pi))

        return q_star, sims, deflection_deg

    def search_canonical_space(self, q_star: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches untouched canonical space with personalized query q*."""
        q_star_norm = q_star / (np.linalg.norm(q_star) + 1e-8)
        scores = np.dot(self.X, q_star_norm)
        top_idx = np.argsort(-scores)[:top_k]

        return [
            {
                "label": self.labels[idx],
                "score": float(scores[idx]),
                "rank": r + 1,
                "mass": float(self.m[idx])
            }
            for r, idx in enumerate(top_idx)
        ]

    def reinforce_mass_from_interaction(self, q_star: np.ndarray):
        """
        Fast timescale: Query creates new relevance evidence.
        Delta m_i proportional to K(sim(q*, x_i)).
        Maintains sum(m) = 1.
        """
        scores = np.dot(self.X, q_star)
        K_scores = self._kernel(scores)
        
        if np.sum(K_scores) > 0:
            # Query relevance distribution
            query_dist = K_scores / np.sum(K_scores)
            # Update mass with fast reinforcement rate
            beta = self.cfg.reinforce_rate
            self.m = (1.0 - beta) * self.m + beta * query_dist
            self.m = self.m / np.sum(self.m)  # strictly normalize

    def slow_graph_fusion(self, new_frequencies: Optional[np.ndarray] = None):
        """
        Slow timescale: Rebuilds structural PageRank and blends with old mass using EMA:
        m_new = alpha * p_new + (1 - alpha) * m_old (alpha = 0.5)
        """
        if new_frequencies is not None:
            self.v = (new_frequencies / np.sum(new_frequencies)).astype(np.float32)
        
        p_new = self.compute_pagerank()
        self.p = p_new
        alpha = self.cfg.alpha_fusion
        self.m = alpha * p_new + (1.0 - alpha) * self.m
        self.m = self.m / np.sum(self.m)
