# Hotelling Trace Criterion (HTC) for analysis and performance evaluation of Physical Neural Networks (PNN)

This repository accompanies our paper on the Hotelling Trace Criterion (HTC) as a predictive metric for physics-based extreme learning machines (ELMs): https://arxiv.org/abs/2606.31588. It provides the core Python function that performs the full HTC analysis of a ELM and computes the associated metrics. Usage examples and guidelines are highlighted below.

## Hotelling Trace Criterion (HTC)

The HTC, also known as the Lawley–Hotelling trace, is a scalar measure of class separability and corresponds to the criterion maximized by Linear Discriminant Analysis. It is computed directly from the hidden-layer matrix $\mathbf{H}$.

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
\qquad
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{m \in k} (\mathbf{h}_m - \boldsymbol{\mu}_k)(\mathbf{h}_m - \boldsymbol{\mu}_k)^{T} .
$$

The HTC is the scalar separability metric obtained by normalizing the between-class scatter by the within-class scatter:

$$
\mathrm{HTC} = \mathrm{tr}\left(\mathbf{S}_W^{-1}\mathbf{S}_B\right) = \sum_{i}\lambda_{i} ,
$$

where $\mathrm{tr}$ is the matrix trace and the $\lambda_i$ are the non-zero eigenvalues of $\mathbf{S}_W^{-1}\mathbf{S}_B$, each representing the ratio of between-class to within-class variation along a principal direction. A larger HTC indicates greater separation between class means relative to the within-class spread. For numerical stability, $\mathbf{S}_W$ is regularized before inversion (e.g. $\mathbf{S}_W + \varepsilon \mathbf{I}$).


## Experimental and numerical simulation data

The `Data` folder contains the three datasets corresponding
to two different PNN systems: fiber-based ELM experimental data, fiber-based ELM
numerical simulation data, and VCSEL-based PNN experimental data. Each file holds the
output hidden-layer state of a corresponding PNN — its response to a given input (see the paper for
details). Within each folder, the data is split into training and test sets obtained
for different choices of physical parameters (for different nonlinear parameters in the case of 
fiber-based PNN and for different wavelength in the case of VCSEL-based PNN).

The hidden-layer dimensions are:

| System                        | Training   | Test      |
|-------------------------------|------------|-----------|
| Fiber-based ELM (exp. & sim.) | 9000 × 600 | 1000 × 600 |
| VCSEL (exp.)                  | 4500 × 600 | 500 × 600  |

where the first number is the number of samples and the second the number of
hidden-layer neurons. The data was preprocessed to match dimensionality across the two
platforms by linearly interpolating each hidden-layer representation to N = 600 neurons.
All data is obtained for the MNIST handwritten-digit benchmark.

The execution code loads all datasets and processes them using the `elm_classifier`
function (see the `Utils` folder), which essentially computes the HTC, performs ridge regression
on the output hidden layer, and computes the MSE. Finally, it produces a plot of test
accuracy, MSE and HTC as a function of the physical parameter of each system.
