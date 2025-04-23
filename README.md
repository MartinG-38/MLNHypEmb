# MLNHypEmb

**MLNHypEmb** is a Python library for embedding multilayer (multiplex) networks into hyperbolic space, with a focus on providing a layer-wise embedding using the Poincaré disk model. This approach enables the geometric analysis and visualization of complex multilayered systems, capturing both intra-layer and inter-layer structures.

---

## Features

- **Layer-wise Hyperbolic Embedding:**  
  Embed each layer of a multiplex network independently or jointly in the Poincaré disk, preserving both local and global network geometry.

- **Flexible Network Support:**  
  Handles both synthetic (e.g., Stochastic Block Model) and real-world multiplex networks, including cases where layers have different numbers of nodes.

- **Multiple Embedding Methods:**  
  Supports Isomap and Spectral embedding techniques for dimensionality reduction.

- **Customizable Radius Attribution:**  
  Several strategies for assigning node radii in hyperbolic space (degree-based, order-based, log-degree).

- **Edge Pre-weighting and Coupling:**  
  Options for edge weighting and inter-layer coupling to control embedding influence.

- **Visualization Tools:**  
  Functions for visualizing embedded networks in the Poincaré disk, including community coloring and perturbation analysis.

---

## Repository Structure

```
MLNHypEmb/
├── src/
│   ├── EmbeddingMethod.py         # Main class for symmetric multilayer hyperbolic embedding
│   ├── EmbeddingMethodUnsym.py    # Embedding for asymmetric (different node count) layers
│   ├── OpenData.py                # Data loading and synthetic network generation (SBM, perturbations)
│   ├── Tools.py                   # Hyperbolic geometry utilities (distance, alignment, barycenter, etc.)
│   ├── VizData.py                 # Visualization utilities for hyperbolic embeddings
├── Multiplex_kaptail_example.ipynb    # Example: Real-world Kapferer tailor shop multiplex network
├── Multiplex_SBM_example.ipynb        # Example: Synthetic SBM multiplex network
├── README.md
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/MLNHypEmb.git
cd MLNHypEmb
pip install -r requirements.txt
```

Dependencies include: `numpy`, `networkx`, `scikit-learn`, `matplotlib`, `pandas`

---

## Quick Start

### 1. Generate or Load a Multiplex Network

You can generate a synthetic SBM multiplex network or load your own data:

```python
from src.OpenData import generate_sbm_multilayer
G_tot, mat_tot, G_global, node_com = generate_sbm_multilayer(n_nodes=100, n_layers=3, n_communities=3)
```

### 2. Embed the Network in Hyperbolic Space

```python
from src.EmbeddingMethod import MlHypEmb
emb = MlHypEmb(preweight=True, method='Isomap', radius='degree')
emb.Load_data(matrices=mat_tot, mu_mat=30)
emb.Embedding(n_jobs=-1)
```

### 3. Visualize the Embedding

```python
import matplotlib.pyplot as plt
import networkx as nx

for i, G in enumerate(G_tot):
    pos = emb.embeddings[i]
    pos_dict = {node: pos[j, :] for j, node in enumerate(G.nodes())}
    plt.figure()
    nx.draw_networkx_nodes(G, pos_dict, node_size=50)
    nx.draw_networkx_edges(G, pos_dict, alpha=0.2)
    plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, color='gray', linestyle=':'))
    plt.axis('equal')
    plt.title(f'Layer {i}')
    plt.show()
```

---

## Notebooks

- **Multiplex_kaptail_example.ipynb**  
  Step-by-step analysis of the Kapferer tailor shop dataset, including embedding, visualization, perturbation, and coupling analysis.

- **Multiplex_SBM_example.ipynb**  
  Demonstrates generation, embedding, and visualization of synthetic SBM multiplex networks.

---

## Modules Overview

- `EmbeddingMethod.py`: Main class for symmetric multilayer embedding.
- `EmbeddingMethodUnsym.py`: Embedding for multiplex networks with different node counts per layer.
- `OpenData.py`: Functions for loading data and generating synthetic multiplex networks (SBM, perturbations).
- `Tools.py`: Hyperbolic geometry utilities (distance, barycenter, alignment, etc.).
- `VizData.py`: Visualization helpers for Poincaré disk and network layers.

---

## Citation

If you use MLNHypEmb in your research, please cite:

```
@software{guillemaud2025mlnhypemb,
  author = {Martin Guillemaud},
  title = {MLNHypEmb: Hyperbolic Embedding for Multilayer Networks},
  year = {2025},
  url = {https://github.com/yourusername/MLNHypEmb}
}
```

---

## License

MIT License

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the author at martin.guillemaud@icm-institute.org.

