"""
Fast, Zero-API Cached PageRank Personality Space.
Runs entirely on the existing cached Gemini embeddings without making any new API calls.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
import umap

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

from src.config import config
from src.graph.pagerank_model import SemanticPageRankSpace

def run_cached_pagerank(damping: float = 0.85, sim_threshold: float = 0.45):
    print("=" * 65)
    print("  ZERO-API CACHED SEMANTIC PAGERANK PERSONALITY SPACE")
    print("=" * 65)

    # 1. Load cached embeddings
    cache_path = config.cache_dir / "embeddings_gemini.json"
    if not cache_path.exists():
        raise FileNotFoundError("Cache file not found.")

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # Filter out empty or non-vector entries
    valid_items = [(k, v) for k, v in cache.items() if isinstance(v, list) and len(v) == 3072]
    labels = [k for k, v in valid_items]
    embeddings = np.array([v for k, v in valid_items], dtype=np.float32)

    print(f"Loaded {len(labels)} cached concept embeddings (3072-d) with 0 API calls.")

    # 2. Count frequencies from Takeout search history
    from src.ingestion.takeout_parser import TakeoutParser
    parser = TakeoutParser()
    files = parser.find_takeout_files()
    events = parser.parse_search_history_html(files["search_file"])

    event_queries = [e["query"].strip() for e in events if e.get("query")]
    counter = Counter(event_queries)
    frequencies = np.array([counter.get(lbl, 1) for lbl in labels], dtype=np.float32)

    # 3. Construct Graph & Run PageRank
    model = SemanticPageRankSpace(
        labels=labels,
        frequencies=frequencies,
        embeddings=embeddings,
        damping=damping,
        sim_threshold=sim_threshold,
        power=1.5
    )

    model.compute_pagerank()
    rankings = model.get_ranked_concepts(top_k=30)

    print("\n" + "=" * 65)
    print("  TOP 25 PERSONALITY CORE CONCEPTS (Highest PageRank Mass)")
    print("=" * 65)
    for c in rankings["top_concepts"][:25]:
        print(f"#{c['rank']:2d} | Mass: {c['relative_mass']:5.2f}x | Links: {c['connections']:3d} | \"{c['label']}\"")

    print("\n" + "=" * 65)
    print("  BOTTOM 15 ISOLATED / SINK CONCEPTS (Lowest PageRank Mass)")
    print("=" * 65)
    for c in rankings["bottom_concepts"][:15]:
        print(f"#{c['rank']:2d} | Mass: {c['relative_mass']:5.2f}x | Links: {c['connections']:3d} | \"{c['label']}\"")

    # 4. Compute 2D UMAP Projections
    print("\n[Step 4/4] Computing 2D UMAP Projections for visualizer...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj_2d = reducer.fit_transform(model.X0)

    particles_data = []
    norm_mass = rankings["mass_array"]
    for i, label in enumerate(labels):
        particles_data.append({
            "id": i,
            "label": label,
            "mass": float(norm_mass[i]),
            "raw_frequency": int(frequencies[i]),
            "pagerank_prob": float(model.pi[i]),
            "x0_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],
            "x_2d": [float(proj_2d[i, 0]), float(proj_2d[i, 1])],
            "displacement": 0.0,
            "connections": int(np.count_nonzero(model.W[i]))
        })

    payload = {
        "metadata": {
            "total_events": len(events),
            "num_particles": len(labels),
            "model_type": "Semantic PageRank Personality Space",
            "damping": damping,
            "sim_threshold": sim_threshold,
            "embedding_provider": "gemini (cached)",
            "embedding_model": "models/gemini-embedding-2",
            "train_events_count": len(events),
            "test_events_count": 0
        },
        "particles": particles_data,
        "top_concepts": rankings["top_concepts"],
        "bottom_concepts": rankings["bottom_concepts"],
        "top_similarity_shifts": [],
        "all_query_evaluations": []
    }

    out_file = config.processed_dir / "experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nResults exported to: {out_file}")

if __name__ == "__main__":
    run_cached_pagerank()
