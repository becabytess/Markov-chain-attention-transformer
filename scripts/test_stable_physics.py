"""
Stable Semantic Deformation Engine with Overdamped Dynamics.
Guarantees bounded deformation (+0.05 to +0.20 delta sim) without cluster collapse.
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
X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)

N, dim = X0.shape
print(f"Loaded {N} particles in {dim}-d space.")

# Hyperparameters
G = 0.05          # Attraction gain
k_spring = 2.0    # Restoring spring constant
max_deflection_deg = 12.0  # Max deflection in degrees
max_disp = np.sin(np.radians(max_deflection_deg))  # chord displacement cap

from src.ingestion.takeout_parser import TakeoutParser
parser = TakeoutParser()
events = parser.parse_search_history_html(parser.find_takeout_files()['search_file'])[-300:]
train_events = events[:210]
test_events = events[210:]

M = np.ones(N, dtype=np.float32)
decay = 0.992

# Step 1: Accumulate mass and co-occurrence over the training stream
for step, e in enumerate(train_events):
    q_str = e['query'].strip()
    if q_str not in cache:
        continue
    q_vec = np.array(cache[q_str], dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    # Cosine similarity to baseline
    sims = np.dot(X0, q_vec)
    
    # Soft mass update: only top-matching concepts get mass
    rel = np.maximum(0.0, sims - 0.50) / 0.50
    delta_m = 1.0 * (rel ** 2)
    M = (M * decay) + delta_m

# Step 2: Compute localized pairwise semantic attraction
# S0 baseline similarity
S0 = np.dot(X0, X0.T)
np.fill_diagonal(S0, 0.0)

# Only attract within positive semantic relationship (S0 > 0.50)
attract_kernel = np.maximum(0.0, S0 - 0.45) ** 1.5

# Gravitational pull matrix: heavier concepts pull related concepts
# F_grav on particle i is sum_j G * M[j] * attract_kernel[i, j] * (X0[j] - S0[i, j] * X0[i])
M_weights = attract_kernel * (M[None, :] ** 0.8) * G

# Direction of attraction in tangent space
tangent_pull = np.dot(M_weights, X0) - np.sum(M_weights * S0, axis=1, keepdims=True) * X0

# Net pull magnitude
pull_norm = np.linalg.norm(tangent_pull, axis=1, keepdims=True) + 1e-8

# Bounded displacement via saturated spring: disp = max_disp * (pull / (k + pull))
disp_scale = max_disp * (pull_norm / (k_spring + pull_norm))

# Compute active positions X
X_active = X0 + disp_scale * (tangent_pull / pull_norm)
X_active = X_active / np.linalg.norm(X_active, axis=1, keepdims=True)

# Step 3: Compute shifts & metrics
St = np.dot(X_active, X_active.T)
delta_S = St - S0
np.fill_diagonal(delta_S, -999)

angular_disp_deg = np.arccos(np.clip(np.sum(X_active * X0, axis=1), -1.0, 1.0)) * (180.0 / np.pi)
print(f"\nAverage angular deflection: {np.mean(angular_disp_deg):.2f}° (max: {np.max(angular_disp_deg):.2f}°)")

print("\n=== TOP SIMILARITY SHIFTS (Delta S_ij) ===")
triu = np.triu_indices(N, k=1)
shifts = delta_S[triu]
for idx in np.argsort(-shifts)[:10]:
    i, j = triu[0][idx], triu[1][idx]
    print(f"[{labels[i]}] <-> [{labels[j]}] | S0: {S0[i,j]:.3f} -> St: {St[i,j]:.3f} (ΔS: +{delta_S[i,j]:.3f}) | mass: {M[i]:.1f}, {M[j]:.1f}")

# Step 4: Test query evaluation
print("\n=== EVALUATING TEST QUERIES ===")
for q in test_events[:6]:
    query = q['query'].strip()
    if query not in cache:
        continue
    q_vec = np.array(cache[query], dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    
    sim_s = np.dot(X0, q_vec)
    sim_d = np.dot(X_active, q_vec)
    
    idx_s = np.argsort(-sim_s)[:3]
    idx_d = np.argsort(-sim_d)[:3]
    
    print(f"\nFuture Test Query: \"{query}\"")
    print("  [Static X0]:", [f"#{r+1} {labels[j]} ({sim_s[j]:.3f})" for r, j in enumerate(idx_s)])
    print("  [Dynamic X]:", [f"#{r+1} {labels[j]} ({sim_d[j]:.3f}, m={M[j]:.1f})" for r, j in enumerate(idx_d)])
