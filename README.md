# MLNHypEmb

**MLNHypEmb** is a Python library for embedding multilayer (multiplex) networks into hyperbolic space, with a focus on providing a layer-wise embedding using the Poincaré disk model. This approach enables the geometric analysis and visualization of complex multilayered systems, capturing both intra-layer and inter-layer structures.

The method is detailed in the paper : https://doi.org/10.48550/arXiv.2505.20378 

---

## Features

- **Layer-wise Hyperbolic Embedding in the Poincaré Disk:**  
    Perform hyperbolic embedding for each layer of a multiplex or multilayer network, either independently or jointly, while preserving both local and global geometric structures.

- **Support for Multiplex and Multilayer Networks:**  
    Compatible with networks having identical or distinct node sets across layers, with or without inter-layer connections.

- **Synthetic Network Generation with SBM:**  
    Includes a multilayer Stochastic Block Model (SBM) generator for testing and benchmarking on synthetic networks.

- **Advanced Hyperbolic Geometry Tools:**  
    Features barycenter estimation in the Poincaré disk using the Klein disk model and hyperbolic Gaussian estimation for point distributions via the tangent plane.

- **Visualization and Analysis Tools:**  
    Provides functions for visualizing embeddings in the Poincaré disk, including community detection, perturbation analysis, and layer-specific visualizations.

---

## Repository Structure

```
MLNHypEmb/
├── src/
│   ├── EmbeddingMethod.py
│   ├── EmbeddingMethodUnsym.py
│   ├── OpenData.py
│   ├── Tools.py
│   ├── VizData.py
├── Data/                                         # Example data for the Kapferer tailor shop (see notebook, data from http://casos.cs.cmu.edu/computational_tools/datasets/sets/kaptail/)
├── Multilayers_kaptail_example.ipynb             # Example: Real-world Kapferer tailor shop multilayer network
├── Multilayers_SBM_example.ipynb                 # Example: Synthetic SBM multilayer network
├── Multilayers_SBM_unsymmetric_example.ipynb     # Example: SBM multilayer network with asymmetric layers
├── Hyperbolic_gaussian_estimation_example.ipynb  # Example: Hyperbolic Gaussian estimation in the Poincaré disk
├── README.md                                     # Project documentation
├── LICENSE                                       # BSD 3-Clause License
├── CITATION.cff                                  # Citation metadata file
├── requirements.txt                              # Python dependencies
```

*The `Data/` directory contains the example datasets used in the Kapferer tailor shop notebook.  
Kapferer data source: http://casos.cs.cmu.edu/computational_tools/datasets/sets/kaptail/*

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

- **Multilayers_kaptail_example.ipynb**  
  Step-by-step analysis of the Kapferer tailor shop dataset, including embedding, visualization, perturbation, and coupling analysis.

- **Multilayers_SBM_example.ipynb**  
  Demonstrates generation, embedding, and visualization of synthetic SBM multilayer networks.

- **Multilayers_SBM_unsymmetric_example.ipynb**  
  Example of embedding and analyzing SBM multilayer networks with asymmetric (different node count) layers.

- **Hyperbolic_gaussian_estimation_example.ipynb**  
  Demonstrates estimation and visualization of Gaussian distributions in hyperbolic space (Poincaré disk).

---

## Modules Overview

- `EmbeddingMethod.py`: Main class for symmetric multilayer embedding.
- `EmbeddingMethodUnsym.py`: Embedding for multiplex networks with different node counts per layer.
- `OpenData.py`: Functions for loading data and generating synthetic multiplex networks (SBM, perturbations).
- `Tools.py`: Hyperbolic geometry utilities (distance, barycenter, alignment, etc.).
- `VizData.py`: Visualization helpers for Poincaré disk and network layers.

---

## Citation

If you use MLNHypEmb in your research or project, please cite as mentionned in the Licence:

```
@software{guillemaud2025mlnhypemb,
  author = {Martin Guillemaud},
  title = {MLNHypEmb: Hyperbolic Embedding for Multilayer Networks},
  year = {2025},
  url = {https://github.com/yourusername/MLNHypEmb}
}
```

**If you use this library for published work, please also cite the associated paper.**

```
@misc{guillemaud2025hyperbolicembeddingmultilayernetworks,
      title={Hyperbolic embedding of multilayer networks}, 
      author={Martin Guillemaud and Vera Dinkelacker and Mario Chavez},
      year={2025},
      archivePrefix={arXiv},
      url={https://doi.org/10.48550/arXiv.2505.20378}, 
}
```

---

## License

This project is licensed under the BSD 3-Clause License.

- **Copyright (c) 2025, Martin Guillemaud, Sorbonne Universite, INRIA**
- All rights reserved.

See LICENSE file for more information

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the author at martin.guillemaud [at] gmail [dot] com.

