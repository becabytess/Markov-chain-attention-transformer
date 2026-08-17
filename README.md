# Gravitational Memory (Gravimem) 🪐

> **Dynamic Semantic Memory via Query Deformation & Semantic Graph PageRank.**
> *An ongoing research project exploring dynamic, personalized knowledge geometry beyond static embeddings and spaced repetition.*

---

## 1. Overview & Core Hypothesis

Traditional semantic search treats concept embeddings as static vectors:
$$\text{concept } c_i \longrightarrow \text{fixed embedding } x_i^0$$

In human cognition, however, our semantic interpretation of a concept evolves as we study, explore, and revisit interconnected domains:
> When studying *gradient descent*, *backpropagation*, and *neural networks*, a person's conceptual representation of *calculus* acquires a contextual bias toward optimization and learning algorithms without losing its underlying meaning.

**Gravimem** explores a dual-timescale architecture:
1. **Frozen Canonical Reference Space ($X$):** Preserves general semantic truth without distortion.
2. **Personal Influence Field ($m$):** Captures long-term structural interests via Semantic PageRank blended with fast, interaction-driven query trails.
3. **The Gravitational Query Lens ($q^*$):** Bends incoming query vectors toward the user's active attractors, retrieving personalized context from the pristine canonical space.

---

## 2. Mathematical Formulation

### A. Semantic Graph & Structural PageRank Prior ($p$)
Given canonical embeddings $x_1, \dots, x_N \in \mathbb{S}^{d-1}$:
$$W_{ij} = \max\left(0, \frac{\operatorname{sim}(x_i, x_j) - \theta_{\text{sim}}}{1 - \theta_{\text{sim}}}\right)^p$$
$$p = \operatorname{PageRank}(W, \text{teleport}=v), \quad \sum_{i=1}^N p_i = 1$$
where $v$ is the user's activity frequency distribution.

### B. Query Gravitational Deflection ($q^*$)
When a query arrives with initial embedding $q_0 = E(\text{query})$:
$$\Delta q = \eta \sum_{i=1}^N m_i \cdot K\left(\operatorname{sim}(q_0, x_i)\right) \cdot (x_i - q_0)$$
$$q^* = \frac{q_0 + \Delta q}{\|q_0 + \Delta q\|}$$
where:
* $m_i$ is the active personal influence mass ($\sum m_i = 1$).
* $K(s) = \max\left(0, \frac{s - \theta}{1 - \theta}\right)^2$ is a neighborhood affinity kernel.
* $\eta$ is the personalization coefficient.

### C. Fast Interaction Reinforcement
Each query event creates immediate evidence:
$$\Delta m_i = \beta \cdot K\left(\operatorname{sim}(q^*, x_i)\right)$$
$$m \leftarrow \frac{m + \Delta m}{\sum (m + \Delta m)}$$

### D. Slow Structural Fusion (EMA)
When the graph topology is updated with new knowledge, the new structural PageRank $p^{\text{new}}$ is merged with the active trail:
$$m^{\text{new}} = \alpha \cdot p^{\text{new}} + (1 - \alpha) \cdot m^{\text{old}}$$

---

## 3. Repository Structure

```
gravitational-memory/
├── data/
│   └── sample/               # Sample activity stream for reproduction
├── src/
│   ├── config.py             # Hyperparameters & environment settings
│   ├── ingestion/            # Chronological activity parsers (Takeout/JSON/HTML)
│   ├── embeddings/           # Unified embedder (Local BGE / Gemini / Cloudflare)
│   ├── graph/                # Semantic PageRank graph model
│   ├── engine/               # Query Deformation & Particle dynamics engines
│   └── retrieval/            # Benchmark evaluation suites
├── web/
│   ├── server.py             # FastAPI backend
│   └── static/               # Interactive 2D UMAP canvas & A/B retrieval comparator
├── run_query_deformation_experiment.py # End-to-end experiment pipeline
├── requirements.txt
└── README.md
```

---

## 4. Quickstart

### 1. Installation
```bash
git clone https://github.com/becabytess/gravitational-memory.git
cd gravitational-memory
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(By default, the engine runs 100% locally with zero API keys required, or you can plug in your Gemini/Cloudflare API keys).*

### 3. Run Experiment Pipeline
```bash
python run_query_deformation_experiment.py
```

### 4. Launch Interactive Workbench
```bash
uvicorn web.server:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** to explore the dynamic semantic space, inspect concept mass distributions, and test query deflections side-by-side.

---

## 5. Research Roadmap & Open Questions

We welcome research contributions and experiments in:
* **Multi-Scale Temporal Decay:** Dynamic half-life for short-term curiosity vs. foundational career skills.
* **Hierarchical Concept Abstraction:** Automatic hierarchical grouping of fine-grained concept particles.
* **Multi-Modal Personalization:** Extending the query lens to image, audio, and code embeddings.
* **Personalized RAG:** Evaluating context retrieval improvements in LLM agent memory architectures.

---

## License
MIT License
