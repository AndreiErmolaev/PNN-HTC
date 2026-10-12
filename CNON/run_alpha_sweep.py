"""
Damping sweep of the single-hidden-layer CNON used as an ELM (Fig. 4 of the paper).

For each of 30 damping coefficients alpha in [0.1, 5] a new random CNON with
N = 600 oscillators is built, the first 10 000 training images of MNIST or
fashion-MNIST are injected, and the steady-state hidden layer H is analysed with
`Utils/elm_classifier.py` exactly as for the HNLF and LA-VCSEL data: ridge
readout (lambda_R scanned on a validation set), test MSE, and HTC on the pooled
training and test states. The discriminant spectrum lambda_i is stored for the
spectral description of Fig. 4.

Usage:
    python run_alpha_sweep.py --dataset mnist
    python run_alpha_sweep.py --dataset fashion
"""

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
import torch
import torchvision
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Utils"))
from elm_classifier import elm_classifier  # noqa: E402

from cnon import CNON, spectral_radius  # noqa: E402
from htc import discriminant_spectrum  # noqa: E402

ALPHAS = np.logspace(np.log10(0.1), np.log10(5), 30)
DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}

N_IMAGES = 10000        # first images of the official training set
N_TRAIN = 9000          # of which the last 1000 are used for validation
ELM_KWARGS = dict(normalize=False, lambda_reg=None, scan_lambda=True, n_lambda=100,
                  lambda_min=1e-10, lambda_max=1e6, objective='mse', PCvar=0.95,
                  validation=True, permutation=False, val_size=1000, sampling=600,
                  verbose=False)


def load_images(dataset, root, n):
    ds = DATASETS[dataset](root=root, train=True, download=True)
    X = ds.data[:n].reshape(n, -1).float().div(255)     # identical to ToTensor()
    y = ds.targets[:n].numpy()
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--n-neurons", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0,
                        help="the network for the i-th alpha is built with seed + i")
    parser.add_argument("--alpha-indices", type=int, nargs="+", default=None,
                        help="run only these alpha indices (e.g. for a quick test)")
    parser.add_argument("--data-root", default=os.path.join(HERE, ".data"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(HERE, "results", "alpha_sweep_{}.npz".format(args.dataset))
    X, y = load_images(args.dataset, args.data_root, N_IMAGES)
    indices = args.alpha_indices if args.alpha_indices is not None else range(len(ALPHAS))

    keys = ["alpha", "train_accuracy", "test_accuracy", "train_mse", "test_mse",
            "train_crossent", "test_crossent", "lambda_used", "HTC", "n_components",
            "n_steps", "max_delta", "residual_median", "frac_unconverged", "rho_eff",
            "seconds"]
    res_all = {k: [] for k in keys}
    lam_all = []
    t_start = time.time()
    for i in indices:
        alpha = ALPHAS[i]
        t0 = time.time()
        torch.manual_seed(args.seed + i)
        model = CNON(n_neurons=args.n_neurons, n_layers=1, alpha=alpha, dt=0.1,
                     max_steps=100, eps_int=1e-4, adaptive=True)
        model.eval()
        with torch.no_grad():
            _, h, n_steps, max_delta = model(X)
            rho_eff = float(spectral_radius(model.W_rec[0].effective_weight()))
            # per-sample max|dh/dt| at the final state: how far each input is
            # from a fixed point when the integration stops
            f = -alpha * h[0] + torch.sin(model.W_input(X) + model.W_rec[0](h[0]))
            residual = f.abs().max(dim=1).values.numpy()
        H = h[0].numpy()

        with contextlib.redirect_stdout(io.StringIO()):     # silence unconditional prints
            res = elm_classifier(H[:N_TRAIN], H[N_TRAIN:], y[:N_TRAIN], y[N_TRAIN:],
                                 **ELM_KWARGS)
        lam_all.append(discriminant_spectrum(res["S_W"], res["S_B"]))

        row = dict(alpha=alpha, HTC=res["HTC_criterion"], n_steps=n_steps,
                   max_delta=max_delta, residual_median=float(np.median(residual)),
                   frac_unconverged=float(np.mean(residual >= 1e-4)), rho_eff=rho_eff,
                   seconds=time.time() - t0)
        for k in keys:
            res_all[k].append(row[k] if k in row else res[k])
        print("alpha={:.3f}  steps={:3d}  max|dh|={:.1e}  unconverged={:4.1f}%  "
              "lambda_R={:.1e}  acc train/test={:.1f}/{:.1f}%  MSE test={:.4f}  "
              "HTC={:.2f}  ({:.0f} s)".format(
                  alpha, n_steps, max_delta, 100 * row["frac_unconverged"],
                  res["lambda_used"], 100 * res["train_accuracy"],
                  100 * res["test_accuracy"], res["test_mse"], res["HTC_criterion"],
                  row["seconds"]), flush=True)

    meta = dict(dataset=args.dataset, n_images=N_IMAGES, n_train=N_TRAIN,
                n_neurons=args.n_neurons, seed=args.seed, dt=0.1, max_steps=100,
                eps_int=1e-4, elm_kwargs=ELM_KWARGS,
                class_counts=np.bincount(y, minlength=10).tolist(),
                total_seconds=time.time() - t_start,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, sklearn=sklearn.__version__,
                              torch=torch.__version__, torchvision=torchvision.__version__),
                threads=torch.get_num_threads(), machine=platform.processor())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, lam=np.array(lam_all), meta=json.dumps(meta),
             **{k: np.array(v) for k, v in res_all.items()})
    print("saved", out, "({:.1f} min)".format(meta["total_seconds"] / 60))


if __name__ == "__main__":
    main()
