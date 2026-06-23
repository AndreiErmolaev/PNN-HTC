# Hotelling Trace Criterion (HTC) for Physics-Based Extreme Learning Machines

This repository accompanies our paper on the Hotelling Trace Criterion (HTC) as a predictive metric for physics-based extreme learning machines (ELMs). It provides the core Python function that performs the full HTC analysis of a physics-based ELM and computes the associated metrics. Usage examples and guidelines will be added.

## Hotelling Trace Criterion (HTC)

The HTC, also known as the Lawley–Hotelling trace, is a scalar measure of class separability and corresponds to the criterion maximized by Linear Discriminant Analysis (LDA). It is computed directly from the hidden-layer matrix $\mathbf{H}$.

Let $\mathbf{H} \in \mathbb{R}^{M \times N}$, where $M$ is the number of samples and $N$ the dimensionality (number of neurons). Each row of $\mathbf{H}$ is a hidden-layer state vector $\mathbf{h}_m^{T}$, for $m = 1, \dots, M$. Each sample belongs to one of $K$ classes, with $M_k$ samples in class $k$.

The class-mean and global-mean representations are

$$
\boldsymbol{\mu}_k = \frac{1}{M_k} \sum_{m \in k} \mathbf{h}_m ,
\qquad
\boldsymbol{\mu} = \frac{1}{M} \sum_{m=1}^{M} \mathbf{h}_m = \frac{1}{M}\sum_{k=1}^{K} M_k \boldsymbol{\mu}_k .
$$

In analogy with multiclass LDA, the between-class and within-class scatter matrices are

$$
\mathbf{S}_B = \sum_{k=1}^{K} M_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^{T} ,
$$

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{m \in k} (\mathbf{h}_m - \boldsymbol{\mu}_k)(\mathbf{h}_m - \boldsymbol{\mu}_k)^{T} .
$$

The HTC is the scalar separability metric obtained by normalizing the between-class scatter by the within-class scatter:

$$
\mathrm{HTC} = \mathrm{tr}\left(\mathbf{S}_W^{-1}\mathbf{S}_B\right) = \sum_{i}\lambda_{i} ,
$$

where $\mathrm{tr}$ is the matrix trace and the $\lambda_i$ are the non-zero eigenvalues of $\mathbf{S}_W^{-1}\mathbf{S}_B$, each representing the ratio of between-class to within-class variation along a principal direction. A larger HTC indicates greater separation between class means relative to the within-class spread. For numerical stability, $\mathbf{S}_W$ is regularized before inversion (e.g. $\mathbf{S}_W + \varepsilon \mathbf{I}$).
