This repository will later contain all the necessary codes and examples highlighting how the Hotelling trace criterion (HTC) can be used to predict the performance and optimize the physics-based extreme learning machines (ELMs).
For the moment, I only place the basic Python function that performs the full analysis of the physics-based ELMs and computes the corresponding metrics.
Some guidelines and usage examples are to be added later.

## Hotelling Trace Criterion (HTC)

To quantify the class separability of the hidden representation, we evaluate the Hotelling trace criterion (HTC) computed from the hidden layer matrix $\mathbf{H}$. Let $\mathbf{H} \in \mathbb{R}^{M \times d}$ denote the hidden layer matrix, where $M$ is the number of samples and $d$ is the dimensionality of the hidden representation. Each row of $\mathbf{H}$ corresponds to a feature vector $\mathbf{h}_i \in \mathbb{R}^{d}$, for $i = 1, \dots, M$.

Each sample belongs to one of $C$ classes indexed by $c \in \{1, \dots, C\}$. Let $m_c$ denote the number of samples in class $c$. The mean hidden representation of class $c$ is defined as

$$
\boldsymbol{\mu}_c = \frac{1}{m_c} \sum_{i \in c} \mathbf{h}_i
$$

where $i \in c$ denotes the set of indices of samples belonging to class $c$.

The global mean of the hidden layer is

$$
\boldsymbol{\mu} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{h}_i
$$

The within-class scatter matrix is defined as

$$
\mathbf{S}_W =
\sum_{c=1}^{C} \sum_{i \in \mathcal{C}_c}
(\mathbf{h}_i - \boldsymbol{\mu}_c)(\mathbf{h}_i - \boldsymbol{\mu}_c)^T
$$

The between-class scatter matrix is defined as

$$
\mathbf{S}_B =
\sum_{c=1}^{C}
m_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T
$$

The Hotelling trace criterion is given by

$$
J_F = \mathrm{tr}\left(\mathbf{S}_W^{-1}\mathbf{S}_B\right)
$$

In practice, for numerical stability, $\mathbf{S}_W$ is often regularized before inversion (e.g., $\mathbf{S}_W + \varepsilon \mathbf{I}$). A larger value of $J_F$ indicates better class separability in the hidden representation.
