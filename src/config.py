"""
Configuration and Hyperparameters for Dynamic Semantic Memory (Gravimem V0).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class PhysicsConfig:
    """Hyperparameters for the Particle Dynamics Engine."""
    G: float = 0.05          # Gravitational attraction strength (scaled for high-d space)
    k: float = 1.20          # Semantic anchor elastic restoring force (keeps meaning grounded)
    beta: float = 0.80       # Momentum damping factor (0 <= beta < 1)
    R: float = 0.05          # Short-range repulsion strength
    r_repel: float = 0.30    # Repulsion characteristic distance scale
    alpha: float = 0.8       # Mass boost per visit
    gamma: float = 0.98      # Mass decay factor per step (recency focus)
    dt: float = 0.1          # Simulation integration time step
    epsilon: float = 1e-3    # Numerical stability epsilon
    max_grav_dist: float = 1.2  # Gravity cutoff: only attract within semantic neighborhood
    lambda_retrieval: float = 0.5  # Retrieval score interpolation: (1-lambda)*x0 + lambda*x


@dataclass
class AppConfig:
    """General Application and Path Configuration."""
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
    
    llm_provider: str = os.getenv("LLM_PROVIDER", "local")
    
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)

    def __post_init__(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
