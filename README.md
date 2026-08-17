# Gravitational Memory (Gravimem) 🪐

> **Dynamic Semantic Memory via Query Deformation & Semantic Graph PageRank**
> *An ongoing research project on dynamic, personalized knowledge geometry beyond static embeddings and spaced repetition.*

---

## 1. The Core Insight

Conventional retrieval treats embeddings as static points in a high-dimensional vector space:
$$x_{\text{concept}} = E(\text{concept})$$

In human cognition, however, our semantic representation of knowledge evolves as we interact with and revisit concepts. When a researcher spends weeks exploring *gradient descent*, *derivatives*, *backpropagation*, and *neural networks*, their active interpretation of *calculus* acquires a contextual bias toward optimization and learning algorithms.

Rather than permanently distorting the underlying concepts or relying on graph edges alone, **Gravimem** models personalization as a **gravitational field acting on incoming queries**:

* The **knowledge space** remains pristine and canonical ($x_i$).
* The user's **accumulated activity** forms a mass distribution ($m_i$).
* An incoming query ($q_0$) passes through this personal field and is **deflected toward active conceptual attractors** ($q^*$).

$$\boxed{\text{Canonical Knowledge Space } X \quad \times \quad \text{Personal Mass Field } m \quad \Longrightarrow \quad \text{Query Lens } q^*}$$

---

## 2. Mathematical Specification

### Step 1: The Canonical Reference Hypersphere
Let $\{c_1, \dots, c_N\}$ be a universe of $N$ concepts. Each concept is mapped to a unit vector via an embedding model $E(\cdot)$:
$$x_i = \frac{E(c_i)}{\|E(c_i)\|} \in \mathbb{S}^{d-1}$$

The canonical matrix $X = [x_1, \dots, x_N]^T \in \mathbb{R}^{N \times d}$ remains **strictly frozen**, ensuring long-term semantic stability without coordinate collapse.

---

### Step 2: Semantic Graph & Structural PageRank Prior
We construct a Markov transition graph where edge weights reflect semantic proximity above a noise threshold $\theta$:
$$W_{ij} = \max\left(0, \frac{x_i \cdot x_j - \theta}{1 - \theta}\right)^p \quad (i \neq j)$$

Let $P$ be the row-stochastic transition matrix:
$$P_{ij} = \frac{W_{ij}}{\sum_k W_{ik}}$$

We compute the stationary distribution $p \in \mathbb{R}^N$ via PageRank with teleportation vector $v$ (proportional to historical query activity):
$$p = d \cdot P^T p + (1 - d) \cdot v$$
$$\sum_{i=1}^N p_i = 1$$

Here, $p_i$ represents the **structural centrality** of concept $i$ across the user's intellectual network. Densely connected knowledge clusters (e.g., ML, math, signal processing) mutually amplify their stationary probability, while isolated one-off topics decay.

---

### Step 3: The Personal Field & Active Mass
The active mass vector $m \in \mathbb{R}^N$ represents the user's current personal influence distribution, initialized to the structural prior:
$$m^{(0)} = p, \quad \text{with } \sum_{i=1}^N m_i = 1$$

---

### Step 4: Gravitational Query Deflection
When a user submits a query, we compute its baseline embedding:
$$q_0 = \frac{E(\text{query})}{\|E(\text{query})\|}$$

The personal mass field exerts a directional force on $q_0$, pulling it toward semantically relevant, high-mass concepts:
$$\Delta q = \eta \sum_{i=1}^N m_i \cdot K(q_0 \cdot x_i) \cdot (x_i - q_0)$$
$$q^* = \frac{q_0 + \Delta q}{\|q_0 + \Delta q\|}$$

where:
* $K(s) = \max\left(0, \frac{s - \theta}{1 - \theta}\right)^2$ is the affinity kernel.
* $\eta$ is the personalization strength parameter.
* $q^*$ is the personalized query vector.

---

### Step 5: Canonical Retrieval
The deflected query searches the untouched reference space:
$$\text{score}_i = q^* \cdot x_i$$

The retrieved concepts naturally align with the user's active context without corrupting generic semantic relationships.

---

### Step 6: Fast Interaction Reinforcement
Every interaction produces immediate relevance evidence. Concepts matching the active query receive a local reinforcement boost:
$$\Delta m_i = \beta \cdot K(q^* \cdot x_i)$$
$$m \leftarrow \frac{m + \Delta m}{\sum_{k=1}^N (m_k + \Delta m_k)}$$

This creates an immediate "hot trail" for active workflows without rebuilding the graph.

---

### Step 7: Slow Structural Fusion (EMA)
When new concepts or substantial activity accumulate, the Markov graph is restructured and fresh PageRank $p^{\text{new}}$ is computed. The engine fuses the macro structural state with the micro query trail via an Exponential Moving Average (EMA):
$$\boxed{m^{\text{new}} = \alpha \cdot p^{\text{new}} + (1 - \alpha) \cdot m^{\text{old}}}$$

Setting $\alpha = 0.5$ balances topological centrality with recent empirical relevance while strictly maintaining $\sum m_i = 1$.

---

## 3. The Dual-Timescale Dynamics

```
                  ┌────────────────────────────────────────┐
                  │      Activity Stream / Interactions    │
                  └───────────────────┬────────────────────┘
                                      │
              Fast Timescale          │          Slow Timescale
             (Per-Query Loop)         │        (Periodic Rebuild)
                     │                │                │
                     ▼                │                ▼
             Query Lens (q*)          │        New Semantic Graph
                     │                │                │
                     ▼                │                ▼
            Canonical Retrieval       │        PageRank Prior (p)
                     │                │                │
                     ▼                │                │
             Mass Reinforcement       │                │
                     │                │                │
                     └───────────────►├◄───────────────┘
                                      ▼
                         EMA Fusion (α = 0.5)
                        m = α·p + (1-α)·m_old
                                      │
                                      ▼
                           Active Mass Field (m)
```

---

## 4. Key Properties & Theorems

1. **Zero Semantic Drift:** Because $X$ is frozen, the geometry of human knowledge is preserved indefinitely.
2. **Strict Normalization:** Both $p$ and $m$ are probability simplices ($\sum m_i = 1$), eliminating runaway amplification and unbounded mass explosion.
3. **Graph-Regularized Personalization:** Isolated noise searches (e.g., one-off queries) cannot capture large mass because they lack graph centrality ($W_{ij} \approx 0$).
4. **Smooth Deflection Limit:** As $\eta \to 0$, $q^* \to q_0$ (pure generic search). As $\eta > 0$, queries experience a smooth angular bias bounded by the local tangent plane.

---

## 5. Ongoing Research Directions

* **Multi-Scale Temporal Half-Life:** Differentiating short-term curiosity spikes from foundational career-long knowledge.
* **Hierarchical Concept Trees:** Graph coarsening and multi-resolution PageRank for scaling to $N > 10^6$ concepts.
* **Personalized RAG & Memory Agents:** Using query deflection fields to condition context retrieval in personal AI assistants.
* **Cross-Modal Gravitational Fields:** Extending $q^*$ deflection across multimodal embeddings (code, text, images, audio).
