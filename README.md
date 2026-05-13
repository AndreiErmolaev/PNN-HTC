This repository will later contain all the necessary codes and examples highlighting how the Hotelling trace criterion (HTC) can be used to predict the performance and optimize the physics-based extreme learning machines (ELMs).
For the moment, I only place the basic Python function that performs the full analysis of the physics-based ELMs and computes the corresponding metrics.
Some guidelines and usage examples are to be added later.

\subsection*{Hotelling Trace Criterion (HTC)}

To quantify the class separability of the hidden representation, we evaluate the Hotelling trace criterion (HTC) computed from the hidden layer matrix $\mathbf{H}$. Let $\mathbf{H} \in \mathbb{R}^{M \times d}$ denote the hidden layer matrix, where $M$ is the number of samples and $d$ is the dimensionality of the hidden representation. Each row of $\mathbf{H}$ corresponds to a hidden feature vector $\mathbf{h}_i \in \mathbb{R}^{d}$, for $i = 1, \dots, M$.

Each sample belongs to one of $C$ classes indexed by $c \in \{1, \dots, C\}$. Let $m_c$ denote the number of samples in class $c$. The mean hidden representation of class $c$ is defined as
\begin{equation}
\boldsymbol{\mu}_c = \frac{1}{m_c} \sum_{i \in \mathcal{C}_c} \mathbf{h}_i ,
\end{equation}
where $\mathcal{C}_c$ denotes the set of indices of samples belonging to class $c$.

The global mean of the hidden layer is
\begin{equation}
\boldsymbol{\mu} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{h}_i .
\end{equation}

The within-class scatter matrix measures the variance of samples around their respective class means:
\begin{equation}
\mathbf{S}_W =
\sum_{c=1}^{C} \sum_{i \in \mathcal{C}_c}
(\mathbf{h}_i - \boldsymbol{\mu}_c)(\mathbf{h}_i - \boldsymbol{\mu}_c)^T .
\end{equation}

The between-class scatter matrix measures the dispersion of the class means relative to the global mean:
\begin{equation}
\mathbf{S}_B =
\sum_{c=1}^{C}
m_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T .
\end{equation}

The Hotelling trace criterion evaluates the separability of the hidden representation by comparing the between-class and within-class scatter matrices:
\begin{equation}
J_F = \mathrm{tr}\left(\mathbf{S}_W^{-1}\mathbf{S}_B\right).
\end{equation}

In practice, for numerical stability, $\mathbf{S}_W$ is often regularized before inversion (e.g., $\mathbf{S}_W + \varepsilon \mathbf{I}$). A larger value of $J_F$ indicates that the class means are well separated relative to the within-class variability of the hidden representations. This metric provides a quantitative measure of the class separability induced by the hidden layer transformation $\mathbf{H}$.
