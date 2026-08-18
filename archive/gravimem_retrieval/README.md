# Gravitational Memory (Gravimem) 🪐

> **A Markov-Based Fluid Memory Algorithm Inspired by Google PageRank**
> *An ongoing research project on dynamic, personalized knowledge geometry beyond static embeddings and spaced repetition.*

---

## 1. The Core Insight

Conventional retrieval treats embeddings as static points in a vector space:
$$x_{\text{concept}} = E(\text{concept})$$

In human cognition, however, our semantic representation of knowledge evolves as we interact with and revisit concepts. When a researcher spends weeks exploring *gradient descent*, *derivatives*, *backpropagation*, and *neural networks*, their active interpretation of *calculus* acquires a contextual bias toward optimization and learning algorithms.

Rather than permanently distorting the underlying concepts or relying on graph edges alone, **Gravimem** models personalization as a **gravitational field acting on incoming queries**:

* The **knowledge space** remains pristine and canonical ($x_i$).
* The user's **accumulated activity** forms a mass distribution ($m_i$).
* An incoming query ($q_0$) passes through this personal field and is **deflected toward active conceptual attractors** ($q^*$).

$$\boxed{\text{Canonical Knowledge Space } X \quad \times \quad \text{Personal Mass Field } m \quad \Longrightarrow \quad \text{Query Lens } q^*}$$

---

## 2. The Algorithm

### Step 1: The Canonical Reference Space
Let $\{c_1, \dots, c_N\}$ be our universe of concepts. Each concept is embedded as a fixed unit vector:
$$x_i = \frac{E(c_i)}{\|E(c_i)\|} \in \mathbb{S}^{d-1}$$

The canonical space $X$ is **never modified**.

---

### Step 2: The Random Surfer on Knowledge (Structural Prior $p$)
To discover which concepts form the central backbone of the user's curiosity, we simulate a **Random Surfer** traversing the semantic space:

1. **Neighbor Transition Probabilities:**
   From any concept $i$, the surfer looks at all neighboring concepts $j$. The probability of jumping to neighbor $j$ is proportional to how semantically similar they are:
   $$P(j \mid i) = \frac{\text{sim}(x_i, x_j)}{\sum_k \text{sim}(x_i, x_k)}$$
   *(Very distant, unrelated concepts with similarity below a threshold receive zero transition probability).*

2. **The Surfer's Walk:**
   * **85% of the time:** The surfer jumps to a semantic neighbor according to $P(j \mid i)$.
   * **15% of the time (Random Exploration):** The surfer gets bored and teleports to a concept based on the user's past search history.

3. **Structural Mass ($p_i$):**
   The percentage of time the surfer spends visiting concept $i$ becomes its structural mass $p_i$ ($\sum p_i = 1$). 
   * Densely connected clusters (e.g. *Audio ML*, *CNNs*, *Optimization*, *Algorithms*) mutually amplify each other and get **high mass**.
   * Disconnected or one-off topics quickly lose the surfer and get **low mass**.

---

### Step 3: The Active Personal Mass ($m$)
The active mass distribution $m$ tracks the user's current state of personal influence, initialized to the structural mass:
$$m = p, \quad \sum_{i=1}^N m_i = 1$$

---

### Step 4: Gravitational Query Deflection
When the user enters a search query, it is embedded as $q_0$:
$$q_0 = \frac{E(\text{query})}{\|E(\text{query})\|}$$

The personal mass field pulls the query vector toward concepts that are both **semantically related** and **high in personal mass**:
$$\Delta q = \eta \sum_{i=1}^N m_i \cdot K(\text{sim}(q_0, x_i)) \cdot (x_i - q_0)$$
$$q^* = \frac{q_0 + \Delta q}{\|q_0 + \Delta q\|}$$

where:
* $K(\text{sim})$ weights the pull so distant unrelated concepts exert zero force.
* $\eta$ is the personalization strength.
* $q^*$ is the personalized, deflected query vector.

---

### Step 5: Search the Canonical Space
We search the untouched canonical space using the deflected query:
$$\text{score}_i = q^* \cdot x_i$$

The retrieved results are naturally biased toward what the user currently cares about without corrupting general semantic truth.

---

### Step 6: Fast Interaction Reinforcement
Every query creates immediate relevance evidence. Concepts matching the active query receive a quick mass boost:
$$\Delta m_i \propto K(\text{score}_i)$$
$$m \leftarrow \frac{m + \Delta m}{\sum_k (m_k + \Delta m_k)}$$

This creates a "hot trail" of active focus without needing to recalculate the whole graph.

---

### Step 7: Slow Structural Fusion (EMA)
When new concepts are added or significant history accumulates, the random surfer is rerun on the updated graph to get a fresh structural prior $p^{\text{new}}$. 

We merge the structural prior with the active query trail using an equal Exponential Moving Average ($\alpha = 0.5$):
$$\boxed{m^{\text{new}} = 0.5 \cdot p^{\text{new}} + 0.5 \cdot m^{\text{old}}}$$

* $p^{\text{new}}$ brings in the updated topological structure.
* $m^{\text{old}}$ preserves the user's recent interaction trail.
* The total mass remains strictly normalized ($\sum m_i = 1$).

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
             Query Lens (q*)          │         Random Surfer
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
                        m = 0.5·p + 0.5·m_old
                                      │
                                      ▼
                           Active Mass Field (m)
```

---

## 4. Key Properties

1. **Zero Semantic Drift:** The canonical space $X$ is frozen. Knowledge never distorts or collapses.
2. **Strictly Bounded ($\sum m_i = 1$):** Mass is a probability distribution. It cannot explode or drift to infinity.
3. **Graph-Regularized:** Isolated one-off searches cannot capture high mass because the random surfer quickly moves away from them.
4. **Smooth Deflection:** Personalization acts as a smooth, localized gravitational lens that bends queries toward active attractors.

---

## 5. Ongoing Research Directions

* **Multi-Scale Temporal Half-Life:** Differentiating short-term curiosity spikes from foundational career-long knowledge.
* **Hierarchical Concept Trees:** Graph coarsening and multi-resolution PageRank for scaling to $N > 10^6$ concepts.
* **Personalized RAG & Memory Agents:** Using query deflection fields to condition context retrieval in personal AI assistants.
* **Cross-Modal Gravitational Fields:** Extending $q^*$ deflection across multimodal embeddings (code, text, images, audio).
