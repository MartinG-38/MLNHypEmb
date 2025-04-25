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
│   ├── EmbeddingMethod.py
│   ├── EmbeddingMethodUnsym.py
│   ├── OpenData.py
│   ├── Tools.py
│   ├── VizData.py
├── Data/                                         # Example data for the Kapferer tailor shop (see notebook, data from http://casos.cs.cmu.edu/computational_tools/datasets/sets/kaptail/)
├── Multiplex_kaptail_example.ipynb               # Example: Real-world Kapferer tailor shop multiplex network
├── Multiplex_SBM_example.ipynb                   # Example: Synthetic SBM multiplex network
├── Multiplex_SBM_unsymmetric_example.ipynb       # Example: SBM multiplex network with asymmetric layers
├── Hyperbolic_gaussian_estimation_example.ipynb  # Example: Hyperbolic Gaussian estimation in the Poincaré disk
├── README.md                                     # Project documentation
├── LICENSE                                       # BSD 3-Clause License
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

- **Multiplex_kaptail_example.ipynb**  
  Step-by-step analysis of the Kapferer tailor shop dataset, including embedding, visualization, perturbation, and coupling analysis.

- **Multiplex_SBM_example.ipynb**  
  Demonstrates generation, embedding, and visualization of synthetic SBM multiplex networks.

- **Multiplex_SBM_unsymmetric_example.ipynb**  
  Example of embedding and analyzing SBM multiplex networks with asymmetric (different node count) layers.

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

If you use MLNHypEmb in your research, please cite:

```
@software{guillemaud2025mlnhypemb,
  author = {Martin Guillemaud},
  title = {MLNHypEmb: Hyperbolic Embedding for Multilayer Networks},
  year = {2025},
  url = {https://github.com/yourusername/MLNHypEmb}
}
```

**If you use this library for published work, please also cite the associated paper (citation will be provided here once published).**

---

## License

This project is licensed under the BSD 3-Clause License.

- **Copyright (c) 2025, Martin Guillemaud**
- All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. **Redistributions of source code** must retain the above copyright notice, this list of conditions and the following disclaimer.
2. **Redistributions in binary form** must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. **Neither the name of the author nor the names of its contributors** may be used to endorse or promote products derived from this software without specific prior written permission.

**Disclaimer:**  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Any commercial use, redistribution, or modification of this library must comply with the terms above. If you use this code as part of a larger software or research project, proper attribution is required.

---

## Contact

For questions, suggestions, or contributions, please open an issue or contact the author at martin.guillemaud [at] gmail [dot] com.

