# Coupled nonlinear oscillator network (CNON)

This folder contains a minimal pipeline that regenerates the results of the paper obtained with the numerical coupled nonlinear oscillator network (CNON) PNN:

- the CNON part of **Fig. 4**: test MSE versus HTC of the single-hidden-layer CNON on MNIST and fashion-MNIST, with the random-choice floor, the isotropic lower bound and the spectral description of each task;
- **Fig. 5**: layer-resolved HTC of a 3-hidden-layer CNON trained by backpropagation;
- the test accuracies of the 2- and 3-hidden-layer CNON quoted in the text.

Figure and equation numbers refer to the current version of the manuscript.

## Model

Each hidden layer $l = 1, \dots, L$ of $N$ oscillators evolves according to

$$
\frac{d\mathbf{h}^{[l]}}{dt} = -\alpha\,\mathbf{h}^{[l]} + \sin\left(\mathbf{W}_\mathrm{rec}^{[l]}\mathbf{h}^{[l]} + \mathbf{W}_\mathrm{in}^{[l]}\mathbf{h}^{[l-1]} + \mathbf{b}^{[l]}\right),
\qquad
\mathbf{y} = \mathbf{W}_\mathrm{out}\,\mathbf{h}^{[L]} + \mathbf{b}_\mathrm{out},
$$

where $\mathbf{h}^{[0]}$ is the input image and $\alpha$ the damping coefficient. The input is clamped while the network is integrated with explicit Euler steps ($dt = 0.1$, $\mathbf{h} = 0$ at $t = 0$). In the ELM configuration, integration stops when $\max|d\mathbf{h}/dt| < 10^{-4}$ over the batch, or after 100 steps. In the backpropagation configuration, it runs for a fixed 50 steps.

`cnon.py` is a port of Anas Skalli's model code (see [Credits](#credits)). With the same random seed it builds the same network and produces bit-identical states, gradients and parameter updates.

## Pipeline

| Script | Output | Paper |
|---|---|---|
| `run_alpha_sweep.py --dataset mnist` / `fashion` | `results/alpha_sweep_<dataset>.npz` | CNON points and spectra of Fig. 4 |
| `run_backprop.py --layers 3` / `2` | `results/backprop_L<L>.npz` | Fig. 5, accuracy of 2 vs 3 layers |
| `make_figures.py` | `figures/fig4_cnon.*`, `figures/fig5_cnon.*` | Figs. 4 and 5 |
| `verify.py` | `results/verification.md`, `figures/check_fig4.png`, `figures/check_fig5.png` | comparison with the published values |

`cnon.py` holds the model. `htc.py` holds the scatter matrices, the HTC, the discriminant spectrum $\lambda_i$, and the MSE–HTC corridor with its spectral fit. The ELM readout and the HTC of the $\alpha$-sweep are computed with `../Utils/elm_classifier.py`, the function also used for the HNLF and LA-VCSEL data.

### Damping sweep (Fig. 4)

- 30 damping coefficients $\alpha$, log-spaced in $[0.1, 5]$. A new random CNON is built for each value, with seed `seed + i`.
- Network: one hidden layer with $N = 600$ and adaptive integration.
- Data: the first 10 000 images of the MNIST or fashion-MNIST training set.
  - Training uses 8 000 images, the next 1 000 are the validation set, and the last 1 000 are the test set.
- Ridge readout without intercept. $\lambda_R$ is chosen from 100 log-spaced values in $[10^{-10}, 10^{6}]$ by the validation MSE.
- The HTC is computed on the pooled training and test hidden states (9 000 samples), with $\mathbf{S}_W$ regularised by $\varepsilon = 10^{-8}$.
- The spectral curves use the normalised spectra $p_i = \lambda_i/\mathrm{HTC}$ of all operating points of a task:
  - $\mathrm{MSE}(\mathrm{HTC}) = \langle K^{-2}\sum_i (1 + p_i\,\mathrm{HTC})^{-1}\rangle + d$, with the offset $d$ fitted in $\log_{10}$;
  - $\eta = (K-1)^{-1}\sum_i \bar p_i^{-1}$.

### Backpropagation (Fig. 5)

- Network: $L = 3$ (and $L = 2$) hidden layers of $N = 50$, $\alpha = 3$, 50 Euler steps. All weights are trained.
- Training: Adam (learning rate $10^{-3}$) with the MSE loss on one-hot targets, batch size 256, 20 epochs on the 60 000 MNIST training images.
- After every epoch the script records, on the 10 000 test images:
  - test accuracy;
  - test MSE of the network's own readout;
  - HTC of every hidden layer computed from $\mathbf{H}_\mathrm{test}$.

## Usage

```bash
pip install -r requirements.txt
python run_alpha_sweep.py --dataset mnist
python run_alpha_sweep.py --dataset fashion
python run_backprop.py --layers 3
python run_backprop.py --layers 2
python make_figures.py
python verify.py
```

torchvision downloads MNIST and fashion-MNIST into `CNON/.data`; use `--data-root` to point to an existing copy. On a 6-core laptop CPU (no GPU), each sweep takes about 8 min and each backpropagation run 5–8 min. The saved results in `results/` let you redraw the figures and rerun the verification without recomputing.

## Verification

The published CNON values come from several independent, unseeded runs, collected with their notebook provenance in `reference/published_cnon.json`. A new run can therefore agree with them only within the spread between those runs: about 2 % median and 15 % maximum pointwise difference in the HTC, and 1.5 % and 7 % in the test MSE.

`verify.py` compares the results in `results/` with these values. All 27 checks pass (`results/verification.md`); `figures/check_fig4.png` and `figures/check_fig5.png` overlay the reproduced and published data.

- **Fig. 4, MNIST:**
  - Against the published run, the HTC differs by 2.0 % (median) / 9.4 % (max) and the test MSE by 1.8 % / 5.6 %.
  - Best test MSE: 0.0296 at $\alpha \approx 0.7$ (published 0.0291–0.0294, same $\alpha$).
  - $\eta = 13.3$ (published CNON spectra: 13.3).
- **Fig. 4, fashion-MNIST:**
  - HTC differs by 1.7 % / 11.8 % and test MSE by 1.6 % / 5.3 %.
  - $\eta = 30.8$ (published CNON spectra: 31.0).
- **Fig. 5:**
  - The first hidden layer's HTC peaks after 4 epochs and then decreases by 34 % (published: epoch 3, 33 %). The HTC of layers 2 and 3 grows about 3-fold and saturates.
  - Layer 3 / layer 1 HTC ratio at epoch 20: 4.87 (published 4.87).
  - Test MSE at epoch 20: 0.0057 (published 0.0060).
  - Log–log correlation with the test MSE: $r = -0.998$ (layer 2) and $-0.997$ (layer 3).
- **Accuracy after 20 epochs:** 96.49 % (2 layers) and 96.56 % (3 layers), against the 96.3 % and 96.8 % quoted in the text. The difference between the two depths is within the epoch-to-epoch fluctuation of about 0.2 %.

## Implementation details

These follow the code that produced the published data and are kept unchanged here:

- $\mathbf{W}_\mathrm{rec}$:
  - Its sparsity pattern is symmetric: 10 % of the upper-triangular entries, plus all self-couplings. The weights themselves are drawn independently from $U(-1, 1)$.
  - The masked weights are scaled to a spectral radius of 0.8, and the mask is applied again in the forward pass and to the gradients.
  - Self-couplings that were also selected therefore enter with a factor 4, which makes the effective spectral radius 0.81–0.92 at $N = 600$ and 1.1–1.5 at $N = 50$ (at initialisation).
- For $\alpha \le 1.5$ (21 of the 30 values) the integration stops at the 100-step limit ($t = 10$) before the steady-state criterion is met. For $\alpha \lesssim 1$ all inputs are still evolving at that point: the median per-sample $\max|d\mathbf{h}/dt|$ is between 0.02 and 1.3. `run_alpha_sweep.py` records, for every $\alpha$:
  - the number of steps;
  - the final $\max|d\mathbf{h}/dt|$;
  - the fraction of samples above the threshold.
- The single-hidden-layer CNON of the original runs was a two-layer network read out at its first layer. The unused layer only changes the step at which the integration stops; the effect on the HTC and the test MSE is below $10^{-6}$ (relative).
- The published runs were not seeded. A seeded run agrees with them within the spread between independent runs, not exactly.

## Credits

The CNON model is adapted from the implementation by Anas Skalli, [github.com/ASkalli/nonlinear_oscillator_network](https://github.com/ASkalli/nonlinear_oscillator_network): the classes `SparseLinear`, `Oscillator_RNN_dyn` and `Oscillator_RNN_parallel` of `Utils/NN_utils.py`. See also [github.com/ASkalli](https://github.com/ASkalli).

## Citation

A. V. Ermolaev, M. Hary, A. Skalli, G. Genty, M. Gebski, T. Czyszanowski, S. Reitzenstein, J. A. Lott, J. M. Dudley and D. Brunner, *Power law scaling for classification accuracy in physical neural networks*, [arXiv:2606.31588](https://arxiv.org/abs/2606.31588) (2026).
