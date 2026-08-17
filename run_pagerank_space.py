"""
Runner for the Semantic PageRank Personality Space Experiment.
Builds the concept graph, runs power iteration, computes UMAP projections, and evaluates retrieval.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import umap
from collections import Counter

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from src.config import config
from src.ingestion.takeout_parser import TakeoutParser
from src.embeddings.embedder import Embedder
from src.graph.pagerank_model import SemanticPageRankSpace


def run_pagerank_pipeline(
    max_events: int = 1000,
    damping: float = 0.85,
    sim_threshold: float = 0.45,
    output_name: str = "pagerank_space.json"
):
    print("=" * 65)
    print("  SEMANTIC PAGERANK PERSONALITY SPACE PIPELINE")
    print("=" * 65)

    # 1. Ingest Data
    print("\n[Step 1/5] Ingesting Takeout Search History...")
    parser = TakeoutParser()
    files = parser.find_takeout_files()
    if not files["search_file"]:
        raise FileNotFoundError(f"No search-history file found in {config.raw_dir}")

    all_events = parser.parse_search_history_html(files["search_file"])
    print(f"Loaded {len(all_events)} total search events from {files['search_file'].name}.")

    if len(all_events) > max_events:
        events = all_events[-max_events:]
        print(f"Using the most recent {len(events)} chronological events.")
    else:
        events = all_events

    # 2. Vocabulary & Frequency Count
    print("\n[Step 2/5] Counting Concept Frequencies...")
    queries = [e["query"].strip() for e in events if e.get("query")]
    counter = Counter(queries)
    unique_queries = list(counter.keys())
    frequencies = np.array([counter[q] for q in unique_queries], dtype=np.float32)

    print(f"Identified {len(unique_queries)} unique concepts across {len(queries)} search events.")

    # 3. Embeddings Generation
    print(f"\n[Step 3/5] Generating/Loading Embeddings via {config.embedding_provider}...")
    embedder = Embedder()
    embeddings_x0 = embedder.embed_texts(unique_queries)
    print(f"Embeddings ready: shape {embeddings_x0.shape}")

    # 4. PageRank Model Execution
    print(f"\n[Step 4/5] Constructing Graph & Running PageRank Power Iteration...")
    model = SemanticPageRankSpace(
        labels=unique_queries,
        frequencies=frequencies,
        embeddings=embeddings_x0,
        damping=damping,
        sim_threshold=sim_threshold,
        power=1.5
    )

    model.compute_pagerank()
    rankings = model.get_ranked_concepts(top_k=25)

    print("\n" + "=" * 65)
    print("  TOP 20 PERSONALITY CORE CONCEPTS (Highest PageRank Mass)")
    print("=" * 65)
    for c in rankings["top_concepts"][:20]:
        print(f"#{c['rank']:2d} | Mass: {c['relative_mass']:5.2f}x | Links: {c['connections']:3d} | \"{c['label']}\"")

    print("\n" + "=" * 65)
    print("  BOTTOM 10 ISOLATED / DEAD-END CONCEPTS (Lowest PageRank Mass)")
    print("=" * 65)
    for c in rankings["bottom_concepts"][:10]:
        print(f"#{c['rank']:2d} | Mass: {c['relative_mass']:5.2f}x | Links: {c['connections']:3d} | \"{c['label']}\"")

    # 5. 2D UMAP Projection for Visualizer
    print("\n[Step 5/5] Computing 2D UMAP Projections...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj_2d = reducer.fit_transform(model.X0)

    particles_data = []
    norm_mass = rankings["mass_array"]
    for i, label in enumerate(unique_queries):
        particles_data.append({
            "id": i,
            "label": label,
            "mass": float(norm_mass[i]),
            "raw_frequency": int(frequencies[i]),
            "pagerank_prob": float(model.pi[i]),
            "x0_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],
            "x_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],  # Static geometry baseline
            "connections": int(np.count_nonzero(model.W[i]))
        })

    # Evaluate 70/30 split query retrieval with PageRank weighting
    split_idx = int(len(events) * 0.70)
    test_events = events[split_idx:]
    query_evaluations = []

    print(f"\nEvaluating PageRank-Weighted Retrieval on {len(test_events)} future test queries...")
    for event in test_events[:50]:
        q_text = event["query"].strip()
        q_vec = embedder.embed_single(q_text)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        # Raw semantic similarity
        sims_raw = np.dot(model.X0, q_norm)
        # PageRank mass-weighted score: sim * (1 + 0.5 * log(1 + mass))
        sims_weighted = sims_raw * (1.0 + 0.40 * np.log1p(norm_mass))

        idx_raw = np.argsort(-sims_raw)[:5]
        idx_weighted = np.argsort(-sims_weighted)[:5]

        static_results = [
            {"label": unique_queries[j], "score": float(sims_raw[j]), "rank": r + 1}
            for r, j in enumerate(idx_raw)
        ]
        weighted_results = [
            {"label": unique_queries[j], "score": float(sims_weighted[j]), "rank": r + 1, "mass": float(norm_mass[j])}
            for r, j in enumerate(idx_weighted)
        ]

        query_evaluations.append({
            "test_query": q_text,
            "timestamp": event.get("timestamp_str"),
            "static_top_k": static_results,
            "dynamic_top_k": weighted_results
        })

    payload = {
        "metadata": {
            "total_events": len(events),
            "num_particles": len(unique_queries),
            "model_type": "Semantic PageRank Personality Prior",
            "damping": damping,
            "sim_threshold": sim_threshold,
            "embedding_provider": embedder.provider,
            "embedding_model": embedder.model_name,
            "train_events_count": split_idx,
            "test_events_count": len(test_events)
        },
        "particles": particles_data,
        "top_concepts": rankings["top_concepts"],
        "bottom_concepts": rankings["bottom_concepts"],
        "all_query_evaluations": query_evaluations
    }

    out_file = config.processed_dir / "experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 65)
    print(f" Semantic PageRank Space built successfully!")
    print(f" Saved to: {out_file}")
    print("=" * 65)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Semantic PageRank Personality Space")
    parser.add_argument("--events", type=int, default=1000, help="Number of search events (default: 1000)")
    parser.add_argument("--damping", type=float, default=0.85, help="PageRank damping factor (default: 0.85)")
    args = parser.parse_args()

    run_pagerank_pipeline(max_events=args.events, damping=args.damping)
