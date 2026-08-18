"""
Unified Embedding Engine with Local Caching & Rate-Limiting.
Supports Gemini Embedding 2 and local sentence-transformers fallback.
"""

import os
import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from tqdm import tqdm

from src.config import config


class Embedder:
    """Manages embedding generation, caching, and similarity computation."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        self.provider = provider or config.embedding_provider
        self.model_name = model_name or (
            "models/gemini-embedding-2" if self.provider == "gemini" else config.local_embedding_model
        )
        self.cache_dir = cache_dir or config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"embeddings_{self.provider}.json"
        
        self.cache: Dict[str, List[float]] = self._load_cache()
        self._local_model = None

    def _load_cache(self) -> Dict[str, List[float]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: could not load cache: {e}")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: could not save cache: {e}")

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {config.local_embedding_model}...")
            self._local_model = SentenceTransformer(config.local_embedding_model)
        return self._local_model

    def embed_texts(self, texts: List[str], batch_size: int = 20, delay_per_batch: float = 0.5) -> np.ndarray:
        """Embeds a list of texts into a numpy array (N, d), using cache for known texts."""
        embeddings: List[List[float]] = []
        texts_to_fetch = []
        indices_to_fetch = []

        # Check cache first
        for i, t in enumerate(texts):
            clean_t = t.strip()
            if clean_t in self.cache:
                embeddings.append(self.cache[clean_t])
            else:
                embeddings.append(None)
                texts_to_fetch.append(clean_t)
                indices_to_fetch.append(i)

        if texts_to_fetch:
            print(f"Generating embeddings for {len(texts_to_fetch)} new texts using {self.provider} ({self.model_name})...")

            if self.provider == "gemini":
                api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")

                endpoint = f"https://generativelanguage.googleapis.com/v1beta/{self.model_name}:embedContent?key={api_key}"
                
                for idx, text in enumerate(tqdm(texts_to_fetch, desc="Gemini Embeddings")):
                    retries = 3
                    success = False
                    while retries > 0 and not success:
                        try:
                            payload = {
                                "model": self.model_name,
                                "content": {"parts": [{"text": text}]}
                            }
                            res = requests.post(endpoint, json=payload, timeout=15)
                            if res.status_code == 200:
                                vec = res.json()["embedding"]["values"]
                                self.cache[text] = vec
                                orig_idx = indices_to_fetch[idx]
                                embeddings[orig_idx] = vec
                                success = True
                            elif res.status_code == 429:
                                print("Rate limited (429), waiting 5s...")
                                time.sleep(5.0)
                                retries -= 1
                            else:
                                print(f"Error {res.status_code}: {res.text}")
                                retries -= 1
                                time.sleep(1.0)
                        except Exception as ex:
                            print(f"Exception during embedding: {ex}")
                            retries -= 1
                            time.sleep(1.0)
                    
                    if not success:
                        # Fallback to zero vector or raise
                        print(f"Warning: Failed to embed '{text}' after retries. Using fallback zero vector.")
                        vec = [0.0] * 3072
                        self.cache[text] = vec
                        embeddings[indices_to_fetch[idx]] = vec

                    time.sleep(0.1)  # small rate-limit safety pause

                self._save_cache()

            else:
                # Local sentence-transformers fallback
                local_model = self._get_local_model()
                encoded = local_model.encode(texts_to_fetch, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
                for idx, vec in enumerate(encoded):
                    vec_list = vec.tolist()
                    self.cache[texts_to_fetch[idx]] = vec_list
                    orig_idx = indices_to_fetch[idx]
                    embeddings[orig_idx] = vec_list
                self._save_cache()

        arr = np.array(embeddings, dtype=np.float32)
        # Normalize vectors to unit length
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_single(self, text: str) -> np.ndarray:
        """Embeds a single string and returns normalized 1D vector."""
        res = self.embed_texts([text])
        return res[0]


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between two sets of vectors (assumed unit normalized)."""
    # If normalized, dot product = cosine similarity
    return np.dot(a, b.T)
