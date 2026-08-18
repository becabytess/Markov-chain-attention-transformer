"""
Vectorized Particle Dynamics Engine for Dynamic Semantic Memory (Gravimem V0).
Implements Tangent-Space Geodesic Semantic Attraction, Elastic Anchoring,
Repulsion, and Temporal Mass Decay.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from src.config import PhysicsConfig, config


class ParticleSystem:
    """
    Simulates a dynamic semantic space where concepts move as particles
    on the embedding hypersphere with bounded geometric deformation.
    """

    def __init__(
        self,
        labels: List[str],
        initial_embeddings: np.ndarray,
        physics_config: Optional[PhysicsConfig] = None
    ):
        self.labels = labels
        self.N = len(labels)
        self.dim = initial_embeddings.shape[1]
        self.cfg = physics_config or config.physics

        # Original static anchors (unit-normalized)
        norms0 = np.linalg.norm(initial_embeddings, axis=1, keepdims=True)
        norms0[norms0 == 0] = 1.0
        self.X0 = (initial_embeddings / norms0).astype(np.float32)

        # Active dynamic positions
        self.X = self.X0.copy()
        # Particle Masses (start at 1.0)
        self.M = np.ones(self.N, dtype=np.float32)
        # Activity / visit counters
        self.visit_counts = np.zeros(self.N, dtype=np.int32)
        
        self.step_count = 0

    def apply_event_query(self, query_vector: np.ndarray, similarity_threshold: float = 0.50) -> np.ndarray:
        """
        Applies a user search event:
        1. Computes similarity to concept particles.
        2. Applies soft-threshold mass update.
        3. Advances particle dynamics along the tangent space of the semantic manifold.
        """
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        # Cosine similarities to baseline anchors: (N,)
        similarities = np.dot(self.X0, q_norm)
        
        # Soft-threshold mass update
        rel_sim = np.maximum(0.0, similarities - similarity_threshold) / (1.0 - similarity_threshold + 1e-6)
        delta_m = self.cfg.alpha * (rel_sim ** 2)
        
        # Update mass with decay
        self.M = (self.M * self.cfg.gamma) + delta_m
        
        active_indices = np.where(delta_m > 0.05)[0]
        self.visit_counts[active_indices] += 1

        # Compute Tangent-Space Gravitational Attraction
        # Baseline similarity matrix
        S0 = np.dot(self.X0, self.X0.T)
        np.fill_diagonal(S0, 0.0)

        # Attract concepts within positive semantic affinity (S0 > 0.45)
        attract_kernel = np.maximum(0.0, S0 - 0.45) ** 1.5
        M_weights = attract_kernel * (self.M[None, :] ** 0.8) * self.cfg.G

        # Tangent pull vector on sphere: proj_x0_i(X0_j) = X0_j - (X0_j . X0_i) * X0_i
        tangent_pull = np.dot(M_weights, self.X0) - np.sum(M_weights * S0, axis=1, keepdims=True) * self.X0
        pull_norm = np.linalg.norm(tangent_pull, axis=1, keepdims=True) + 1e-8

        # Max displacement chord cap (~12 degrees deflection)
        max_disp = np.sin(np.radians(12.0))
        disp_scale = max_disp * (pull_norm / (self.cfg.k + pull_norm))

        # Update active position
        self.X = self.X0 + disp_scale * (tangent_pull / pull_norm)
        self.X = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
        
        self.step_count += 1
        return similarities

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns statistical metrics on current particle system deformation."""
        displacement = np.linalg.norm(self.X - self.X0, axis=1)
        angular_disp_deg = np.arccos(np.clip(np.sum(self.X * self.X0, axis=1), -1.0, 1.0)) * (180.0 / np.pi)

        return {
            "step_count": self.step_count,
            "avg_displacement": float(np.mean(displacement)),
            "max_displacement": float(np.max(displacement)),
            "avg_angular_deg": float(np.mean(angular_disp_deg)),
            "max_angular_deg": float(np.max(angular_disp_deg)),
            "avg_mass": float(np.mean(self.M)),
            "max_mass": float(np.max(self.M)),
            "most_massive_particles": [
                {"label": self.labels[idx], "mass": float(self.M[idx]), "visits": int(self.visit_counts[idx])}
                for idx in np.argsort(-self.M)[:5]
            ]
        }

    def compute_similarity_shifts(self, top_k_pairs: int = 15) -> List[Dict[str, Any]]:
        """
        Computes delta S_ij = sim(x_i, x_j) - sim(x_i^0, x_j^0)
        and returns the concept pairs with the largest positive geometric attraction.
        """
        S0 = np.dot(self.X0, self.X0.T)
        St = np.dot(self.X, self.X.T)

        delta_S = St - S0
        np.fill_diagonal(delta_S, -np.inf)

        triu_indices = np.triu_indices(self.N, k=1)
        shifts = delta_S[triu_indices]
        top_idx = np.argsort(-shifts)[:top_k_pairs]

        results = []
        for idx in top_idx:
            i = triu_indices[0][idx]
            j = triu_indices[1][idx]
            results.append({
                "concept_1": self.labels[i],
                "concept_2": self.labels[j],
                "orig_sim": float(S0[i, j]),
                "active_sim": float(St[i, j]),
                "delta_sim": float(delta_S[i, j]),
                "mass_1": float(self.M[i]),
                "mass_2": float(self.M[j]),
            })

        return results
