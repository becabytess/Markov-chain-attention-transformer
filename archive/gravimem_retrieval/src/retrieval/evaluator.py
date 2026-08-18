"""
Evaluation Engine for Dynamic Semantic Memory (Gravimem V0).
Executes the 70/30 chronological split benchmark comparing Static (X0) vs Dynamic (X) retrieval.
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from src.config import config
from src.embeddings.embedder import Embedder
from src.engine.particle_system import ParticleSystem


class RetrievalEvaluator:
    """Evaluates static vs dynamic semantic retrieval on chronological stream."""

    def __init__(
        self,
        events: List[Dict[str, Any]],
        embedder: Embedder,
        particle_system: ParticleSystem,
        train_ratio: float = 0.70
    ):
        self.events = events
        self.embedder = embedder
        self.particle_system = particle_system
        self.train_ratio = train_ratio
        
        split_idx = int(len(events) * train_ratio)
        self.train_events = events[:split_idx]
        self.test_events = events[split_idx:]
        
        print(f"Total stream events: {len(events)} | Train events: {len(self.train_events)} | Test events: {len(self.test_events)}")

    def run_training_replay(self) -> List[Dict[str, Any]]:
        """Replays train events chronologically, deforming the particle space."""
        print(f"\n--- Replaying {len(self.train_events)} training events through Particle Physics Engine ---")
        history_log = []

        for i, event in enumerate(tqdm(self.train_events, desc="Replaying History")):
            query = event.get("query") or event.get("title", "")
            if not query.strip():
                continue

            q_vec = self.embedder.embed_single(query)
            self.particle_system.apply_event_query(q_vec)

            if (i + 1) % 100 == 0 or (i + 1) == len(self.train_events):
                summary = self.particle_system.get_state_summary()
                summary["event_index"] = i + 1
                history_log.append(summary)

        print("\nTraining Replay Complete!")
        print("Final State Summary:", json.dumps(self.particle_system.get_state_summary(), indent=2))
        return history_log

    def evaluate_test_queries(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluates held-out future queries against both Static X0 and Dynamic X spaces.
        """
        print(f"\n--- Evaluating {len(self.test_events)} held-out future queries ---")
        
        static_ranks = []
        dynamic_ranks = []
        hybrid_ranks = []
        
        query_evaluations = []
        
        # Prepare normalized matrices
        X0_norm = self.particle_system.X0 / (np.linalg.norm(self.particle_system.X0, axis=1, keepdims=True) + 1e-8)
        X_norm = self.particle_system.X / (np.linalg.norm(self.particle_system.X, axis=1, keepdims=True) + 1e-8)
        
        # Hybrid embedding space
        lam = config.physics.lambda_retrieval
        X_hybrid = (1.0 - lam) * X0_norm + lam * X_norm
        X_hybrid_norm = X_hybrid / (np.linalg.norm(X_hybrid, axis=1, keepdims=True) + 1e-8)

        labels = self.particle_system.labels

        for i, event in enumerate(tqdm(self.test_events, desc="Evaluating Test Queries")):
            test_query = event.get("query") or event.get("title", "")
            if not test_query.strip():
                continue

            q_vec = self.embedder.embed_single(test_query)
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

            # Cosine similarities
            sim_static = np.dot(X0_norm, q_norm)
            sim_dynamic = np.dot(X_norm, q_norm)
            sim_hybrid = np.dot(X_hybrid_norm, q_norm)

            # Top-K indices
            idx_static = np.argsort(-sim_static)[:top_k]
            idx_dynamic = np.argsort(-sim_dynamic)[:top_k]
            idx_hybrid = np.argsort(-sim_hybrid)[:top_k]

            static_results = [
                {"label": labels[j], "score": float(sim_static[j]), "rank": r + 1}
                for r, j in enumerate(idx_static)
            ]
            dynamic_results = [
                {"label": labels[j], "score": float(sim_dynamic[j]), "rank": r + 1, "mass": float(self.particle_system.M[j])}
                for r, j in enumerate(idx_dynamic)
            ]
            hybrid_results = [
                {"label": labels[j], "score": float(sim_hybrid[j]), "rank": r + 1}
                for r, j in enumerate(idx_hybrid)
            ]

            query_evaluations.append({
                "test_query": test_query,
                "timestamp": event.get("timestamp_str"),
                "static_top_k": static_results,
                "dynamic_top_k": dynamic_results,
                "hybrid_top_k": hybrid_results,
            })

        # Calculate geometric shifts
        top_shifts = self.particle_system.compute_similarity_shifts(top_k_pairs=10)

        results = {
            "num_test_queries": len(query_evaluations),
            "final_particle_state": self.particle_system.get_state_summary(),
            "top_similarity_shifts": top_shifts,
            "sample_query_evaluations": query_evaluations[:20],
            "all_query_evaluations": query_evaluations
        }

        return results
