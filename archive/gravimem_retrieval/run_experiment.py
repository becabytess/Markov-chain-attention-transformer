"""
Main Experiment Runner for Dynamic Semantic Memory (Gravimem V0).
Executes the complete pipeline: Ingestion -> Embeddings -> Particle Dynamics -> 70/30 Evaluation -> 2D Projections.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import umap

from src.config import config
from src.ingestion.takeout_parser import TakeoutParser
from src.embeddings.embedder import Embedder
from src.engine.particle_system import ParticleSystem
from src.retrieval.evaluator import RetrievalEvaluator


def run_experiment(
    max_events: int = 500,
    train_ratio: float = 0.70,
    output_name: str = "experiment_results.json"
):
    print("=" * 60)
    print("  DYNAMIC SEMANTIC MEMORY (GRAVIMEM V0) EXPERIMENT")
    print("=" * 60)

    # 1. Ingestion
    print("\n[Step 1/5] Ingesting Takeout Search History...")
    parser = TakeoutParser()
    files = parser.find_takeout_files()
    if not files["search_file"]:
        raise FileNotFoundError(f"No search-history file found in {config.raw_dir}")

    all_events = parser.parse_search_history_html(files["search_file"])
    print(f"Loaded {len(all_events)} total search events from {files['search_file'].name}.")

    # Select the most recent max_events for the focused experiment
    if len(all_events) > max_events:
        events = all_events[-max_events:]
        print(f"Focused on the most recent {len(events)} chronological events.")
    else:
        events = all_events

    # 2. Particle Vocabulary
    print("\n[Step 2/5] Constructing Particle Vocabulary...")
    # Gather unique queries from the events to form the concept particle universe
    unique_queries = []
    seen = set()
    for e in events:
        q = e["query"].strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique_queries.append(q)

    print(f"Identified {len(unique_queries)} unique concept particles.")

    # 3. Embeddings Generation (X0)
    print("\n[Step 3/5] Generating Baseline Embeddings (X0)...")
    embedder = Embedder()
    embeddings_x0 = embedder.embed_texts(unique_queries)
    print(f"Generated {embeddings_x0.shape} baseline embedding matrix.")

    # 4. Initialize Particle Physics Engine & Run 70/30 Simulation
    print("\n[Step 4/5] Initializing Vectorized Particle System...")
    particle_system = ParticleSystem(
        labels=unique_queries,
        initial_embeddings=embeddings_x0,
        physics_config=config.physics
    )

    evaluator = RetrievalEvaluator(
        events=events,
        embedder=embedder,
        particle_system=particle_system,
        train_ratio=train_ratio
    )

    # Replay training portion (70%)
    history_log = evaluator.run_training_replay()

    # Evaluate test portion (30%)
    eval_results = evaluator.evaluate_test_queries(top_k=5)

    # 5. Compute 2D Projections for Interactive Visualizer
    print("\n[Step 5/5] Computing 2D UMAP Projections for Visualization...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    
    # Fit UMAP on original embeddings, transform both X0 and active X
    print("Fitting UMAP on original embeddings...")
    proj_x0 = reducer.fit_transform(particle_system.X0)
    print("Transforming active dynamic embeddings...")
    proj_x = reducer.transform(particle_system.X)

    # Build export payload
    particles_data = []
    for i, label in enumerate(unique_queries):
        particles_data.append({
            "id": i,
            "label": label,
            "mass": float(particle_system.M[i]),
            "visits": int(particle_system.visit_counts[i]),
            "x0_2d": [float(proj_x0[i, 0]), float(proj_x0[i, 1])],
            "x_2d": [float(proj_x[i, 0]), float(proj_x[i, 1])],
            "displacement": float(np.linalg.norm(particle_system.X[i] - particle_system.X0[i]))
        })

    experiment_payload = {
        "metadata": {
            "total_events": len(events),
            "num_particles": len(unique_queries),
            "train_ratio": train_ratio,
            "train_events_count": len(evaluator.train_events),
            "test_events_count": len(evaluator.test_events),
            "embedding_provider": embedder.provider,
            "embedding_model": embedder.model_name,
            "physics_config": {
                "G": config.physics.G,
                "k": config.physics.k,
                "beta": config.physics.beta,
                "R": config.physics.R,
                "r_repel": config.physics.r_repel,
                "alpha": config.physics.alpha,
                "gamma": config.physics.gamma,
                "lambda_retrieval": config.physics.lambda_retrieval
            }
        },
        "particles": particles_data,
        "history_log": history_log,
        "top_similarity_shifts": eval_results["top_similarity_shifts"],
        "sample_query_evaluations": eval_results["sample_query_evaluations"],
        "all_query_evaluations": eval_results["all_query_evaluations"]
    }

    out_file = config.processed_dir / output_name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_payload, f, indent=2, ensure_ascii=False)

    print(f"\n========================================================")
    print(f" Experiment completed successfully!")
    print(f" Results exported to: {out_file}")
    print(f"========================================================")

    return experiment_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dynamic Semantic Memory Experiment")
    parser.add_argument("--events", type=int, default=300, help="Number of recent search events to process")
    parser.add_argument("--split", type=float, default=0.70, help="Train/Test split ratio (default 0.70)")
    args = parser.parse_args()

    run_experiment(max_events=args.events, train_ratio=args.split)
