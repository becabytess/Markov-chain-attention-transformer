"""
Spherical Tangent Dynamics Verification Script.
Tests localized geodesic attraction with strong anchor spring to ensure
subtle, realistic semantic deformation (+0.05 to +0.20 delta sim).
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.stdout.reconfigure(encoding='utf-8')

# Load cached embeddings from Gemini
with open('data/processed/experiment_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

labels = [p['label'] for p in d['particles']]
with open('data/cache/embeddings_gemini.json', 'r', encoding='utf-8') as f:
    cache = json.load(f)

X0_list = [cache[lbl.strip()] for lbl in labels]
X0 = np.array(X0_list, dtype=np.float32)
# Unit normalize
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)

N, dim = X0.shape
print(f"Loaded {N} particles in {dim}-d space.")

# Hyperparameters for spherical tangent dynamics
G = 0.02          # Gentle attraction
k = 0.80          # Strong anchor spring restoring force
beta = 0.80       # Momentum
alpha = 0.5       # Mass boost per event
gamma = 0.99      # Mass decay
dt = 0.1

X = X0.copy()
V = np.zeros_like(X)
M = np.ones(N, dtype=np.float32)

# Load training events
with open('data/raw/Takeout/YouTube and YouTube Music/history/search-history.html', 'r', encoding='utf-8') as f:
    pass

# Let's run a test simulation on 210 train events
from src.ingestion.takeout_parser import TakeoutParser
parser = TakeoutParser()
events = parser.parse_search_history_html(parser.find_takeout_files()['search_file'])[-300:]
train_events = events[:210]

for step, e in enumerate(train_events):
    q_str = e['query'].strip()
    if q_str not in cache:
        continue
    q_vec = np.array(cache[q_str], dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # 1. Cosine similarity
    sims = np.dot(X, q_vec)
    # Mass update: only for relevant concepts
    rel = np.maximum(0.0, sims - 0.55) / 0.45
    delta_m = alpha * (rel ** 2)
    M += delta_m
    
    # 2. Tangent Space Forces
    # Pairwise dot product (cosine sim)
    S = np.dot(X, X.T)
    np.fill_diagonal(S, -1.0)
    
    # Only attract within positive semantic relationship (e.g. sim > 0.45)
    attract_mask = np.maximum(0.0, S - 0.40)
    
    # Tangent direction from i to j: proj_{X_i}(X_j) = X_j - (X_j . X_i) * X_i
    # Shape: (N, N, dim)
    # Vectorized: F_grav on particle i is sum_j G * M[i]*M[j] * attract_mask[i,j] * (X[j] - S[i,j]*X[i])
    M_weight = (M[:, None] * M[None, :]) * attract_mask * G
    
    # Sum over j of M_weight[i,j] * X[j]
    weighted_X = np.dot(M_weight, X)
    # Sum over j of M_weight[i,j] * S[i,j]
    weighted_S = np.sum(M_weight * S, axis=1, keepdims=True)
    F_grav = weighted_X - weighted_S * X
    
    # Anchor force in tangent space: X0[i] - (X0[i] . X[i]) * X[i]
    anchor_dots = np.sum(X0 * X, axis=1, keepdims=True)
    F_anchor = k * (X0 - anchor_dots * X)
    
    # Total Tangent Force
    F_total = F_grav + F_anchor
    
    # Integrate velocity & position
    acc = F_total / M[:, None]
    V = beta * V + acc * dt
    X_new = X + V * dt
    # Re-normalize to sphere
    X = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)
    
    M = 1.0 + (M - 1.0) * gamma

# Check results!
S0 = np.dot(X0, X0.T)
St = np.dot(X, X.T)
delta_S = St - S0
np.fill_diagonal(delta_S, -999)

print("\n=== TOP SIMILARITY SHIFTS ===")
triu = np.triu_indices(N, k=1)
shifts = delta_S[triu]
for idx in np.argsort(-shifts)[:10]:
    i, j = triu[0][idx], triu[1][idx]
    print(f"[{labels[i]}] <-> [{labels[j]}] | S0: {S0[i,j]:.3f} -> St: {St[i,j]:.3f} (ΔS: +{delta_S[i,j]:.3f}) | mass: {M[i]:.1f}, {M[j]:.1f}")

# Check displacement angles
angular_disp_deg = np.arccos(np.clip(np.sum(X * X0, axis=1), -1.0, 1.0)) * (180.0 / np.pi)
print(f"\nAverage angular deflection: {np.mean(angular_disp_deg):.2f}° (max: {np.max(angular_disp_deg):.2f}°)")
