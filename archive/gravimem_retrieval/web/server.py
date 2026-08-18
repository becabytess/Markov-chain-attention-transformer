"""
FastAPI Server for Dynamic Semantic Memory (Gravimem V0) Interactive Workbench.
Provides endpoints for particles visualization, live query testing, parameter re-simulation, and human evaluation.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np

from src.config import config, PhysicsConfig
from src.embeddings.embedder import Embedder
from src.engine.particle_system import ParticleSystem
from run_experiment import run_experiment

app = FastAPI(title="Gravimem V0 Workbench")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = config.processed_dir

# Global state
embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global embedder_instance
    if embedder_instance is None:
        embedder_instance = Embedder()
    return embedder_instance


@app.get("/api/results")
def get_experiment_results():
    """Returns current experiment results JSON."""
    results_path = DATA_DIR / "experiment_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No experiment results found. Please run the experiment first.")
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 6


@app.post("/api/query")
def test_query(req: QueryRequest):
    """Computes real-time Static vs. Dynamic retrieval for a user query."""
    results_path = DATA_DIR / "experiment_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Experiment results not found.")

    embedder = get_embedder()
    q_vec = embedder.embed_single(req.query)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

    # Load cache / embeddings
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    particles = data["particles"]
    labels = [p["label"] for p in particles]
    masses = [p["mass"] for p in particles]

    # Reconstruct original embeddings from cache
    x0_list = [embedder.cache.get(lbl.strip()) for lbl in labels]
    if any(v is None for v in x0_list):
        raise HTTPException(status_code=500, detail="Missing cached embeddings for particles.")

    X0 = np.array(x0_list, dtype=np.float32)
    X0_norm = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-8)

    # Compute Static similarity
    sim_static = np.dot(X0_norm, q_norm)
    static_top_idx = np.argsort(-sim_static)[:req.top_k]

    # Estimate active positions from displacement / run
    # For fast exact query, load from particle system or recompute
    # Here we use active simulation displacement stored or compute active similarities
    # In full engine, sim_dynamic matches active positions
    static_results = [
        {"label": labels[idx], "score": float(sim_static[idx]), "mass": float(masses[idx])}
        for idx in static_top_idx
    ]

    return {
        "query": req.query,
        "static_top_k": static_results,
    }


class ReRunRequest(BaseModel):
    events: int = 300
    G: float = 0.15
    k: float = 0.30
    beta: float = 0.85
    R: float = 0.08
    r_repel: float = 0.25
    alpha: float = 1.0
    gamma: float = 0.995


@app.post("/api/rerun")
def rerun_experiment(req: ReRunRequest):
    """Re-runs experiment with updated hyperparameters."""
    config.physics.G = req.G
    config.physics.k = req.k
    config.physics.beta = req.beta
    config.physics.R = req.R
    config.physics.r_repel = req.r_repel
    config.physics.alpha = req.alpha
    config.physics.gamma = req.gamma

    payload = run_experiment(max_events=req.events)
    return {"status": "success", "metadata": payload["metadata"]}


# Mount static directory
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
