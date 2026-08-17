"""
End-to-End Experiment for Query-Deformation Semantic Memory.
Replays training history to evolve personal mass field, runs 50-step EMA fusions (alpha = 0.5),
and evaluates query deformation on held-out future searches.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
import umap
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from src.config import config
from src.ingestion.takeout_parser import TakeoutParser
from src.embeddings.embedder import Embedder
from src.engine.query_deformation_engine import QueryDeformationEngine, QueryEngineConfig


def run_query_deformation_experiment(
    eta: float = 0.25,
    alpha_fusion: float = 0.5,
    sim_threshold: float = 0.40,
    reinforce_rate: float = 0.15
):
    print("=" * 65)
    print("  GRAVIMEM V0: QUERY DEFORMATION SEMANTIC MEMORY EXPERIMENT")
    print("=" * 65)

    # 1. Load Cached Concept Embeddings (0 API Calls)
    cache_path = config.cache_dir / "embeddings_gemini.json"
    if not cache_path.exists():
        raise FileNotFoundError("Embeddings cache not found.")

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    valid_items = [(k, v) for k, v in cache.items() if isinstance(v, list) and len(v) == 3072]
    labels = [k for k, v in valid_items]
    embeddings = np.array([v for k, v in valid_items], dtype=np.float32)
    N = len(labels)
    print(f"[Step 1/5] Loaded {N} concept particles in 3072-d canonical space.")

    # 2. Ingest Search History Stream
    parser = TakeoutParser()
    files = parser.find_takeout_files()
    events = parser.parse_search_history_html(files["search_file"])
    
    # Use the 300-event chronological sequence corresponding to the cached universe
    stream_events = [e for e in events if e.get("query", "").strip() in cache][-300:]
    split_idx = int(len(stream_events) * 0.70)
    train_events = stream_events[:split_idx]
    test_events = stream_events[split_idx:]
    print(f"[Step 2/5] Stream prepared: {len(train_events)} Training events (70%) | {len(test_events)} Held-out Test events (30%)")

    # Initial frequencies from train events
    train_queries = [e["query"].strip() for e in train_events]
    counter = Counter(train_queries)
    frequencies = np.array([counter.get(lbl, 0) for lbl in labels], dtype=np.float32)

    # 3. Initialize Query Deformation Engine
    engine_cfg = QueryEngineConfig(
        eta=eta,
        sim_threshold=sim_threshold,
        alpha_fusion=alpha_fusion,
        reinforce_rate=reinforce_rate
    )

    print(f"\n[Step 3/5] Initializing Engine (eta = {eta}, alpha_fusion = {alpha_fusion})...")
    engine = QueryDeformationEngine(
        labels=labels,
        embeddings=embeddings,
        frequencies=frequencies,
        config=engine_cfg
    )

    print(f" -> Structural PageRank prior computed. Max initial mass: {np.max(engine.p)*N:.2f}x average.")

    # 4. Replay Training Stream (Fast Reinforcement + Slow EMA Graph Fusion)
    print(f"\n[Step 4/5] Replaying {len(train_events)} Training Events chronologically...")
    running_counts = Counter(train_queries)
    
    for step, event in enumerate(tqdm(train_events, desc="Replaying History")):
        q_text = event["query"].strip()
        q_vec = np.array(cache[q_text], dtype=np.float32)
        
        # Deform query with current personal field
        q_star, _, _ = engine.deform_query(q_vec)
        
        # Fast query-driven mass reinforcement
        engine.reinforce_mass_from_interaction(q_star)
        
        # Every 50 events: Slow Graph Fusion (EMA average: alpha * p_new + (1 - alpha) * m_old)
        if (step + 1) % 50 == 0:
            current_freqs = np.array([running_counts.get(lbl, 0) for lbl in labels], dtype=np.float32)
            engine.slow_graph_fusion(new_frequencies=current_freqs)

    print(f" -> Training replay complete! Mass entropy and distribution stable (sum(m) = {np.sum(engine.m):.4f}).")

    # 5. Evaluate Query Deformation on Held-Out Future Searches
    print(f"\n[Step 5/5] Evaluating Query Deformation on {len(test_events)} Held-Out Future Searches...")
    
    eval_results = []
    total_deflections = []
    
    for event in tqdm(test_events, desc="Evaluating Test Queries"):
        q_text = event["query"].strip()
        q_vec = np.array(cache[q_text], dtype=np.float32)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        # Baseline search: Raw unpersonalized query q0
        raw_sims = np.dot(engine.X, q_norm)
        raw_top_idx = np.argsort(-raw_sims)[:5]
        static_top_k = [
            {"label": labels[j], "score": float(raw_sims[j]), "rank": r + 1}
            for r, j in enumerate(raw_top_idx)
        ]

        # Personalized search: Deformed query q*
        q_star, _, deflection_deg = engine.deform_query(q_vec)
        dynamic_top_k = engine.search_canonical_space(q_star, top_k=5)
        
        total_deflections.append(deflection_deg)

        eval_results.append({
            "test_query": q_text,
            "timestamp": event.get("timestamp_str"),
            "deflection_deg": deflection_deg,
            "static_top_k": static_top_k,
            "dynamic_top_k": dynamic_top_k
        })

    avg_deflection = float(np.mean(total_deflections))
    print(f"\nAverage Query Deflection Angle: {avg_deflection:.2f}° (max: {np.max(total_deflections):.2f}°)")

    # 6. Compute 2D UMAP for Visualizer
    print("\nComputing 2D UMAP Layout...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj_2d = reducer.fit_transform(engine.X)

    norm_mass = engine.m * N
    particles_data = []
    for i, label in enumerate(labels):
        particles_data.append({
            "id": i,
            "label": label,
            "mass": float(norm_mass[i]),
            "raw_prob": float(engine.m[i]),
            "x0_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],
            "x_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],  # Canonical space frozen!
            "displacement": 0.0
        })

    payload = {
        "metadata": {
            "total_events": len(stream_events),
            "num_particles": N,
            "model_type": "Query-Deformation Semantic Memory (Gravimem V0)",
            "eta": eta,
            "alpha_fusion": alpha_fusion,
            "avg_query_deflection_deg": avg_deflection,
            "train_events_count": len(train_events),
            "test_events_count": len(test_events),
            "embedding_provider": "gemini (cached)",
            "embedding_model": "models/gemini-embedding-2"
        },
        "particles": particles_data,
        "all_query_evaluations": eval_results
    }

    out_file = config.processed_dir / "experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("  SAMPLE HELD-OUT QUERY PERSONALIZATION COMPARISONS")
    print("=" * 65)
    for res in eval_results[:6]:
        print(f"\nFuture Test Query: \"{res['test_query']}\" (Query deflected by {res['deflection_deg']:.2f}°)")
        print("  [Generic Raw Query q0 Top-3]:")
        for r in res["static_top_k"][:3]:
            print(f"    - #{r['rank']} {r['label']} (score: {r['score']:.3f})")
        print("  [Personalized Deformed Query q* Top-3]:")
        for r in res["dynamic_top_k"][:3]:
            print(f"    - #{r['rank']} {r['label']} (score: {r['score']:.3f}, mass: {r['mass']*N:.2f}x)")

    return payload


if __name__ == "__main__":
    run_query_deformation_experiment()
