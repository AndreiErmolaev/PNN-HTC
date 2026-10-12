"""
Layer-resolved HTC of a multilayer CNON trained by backpropagation (Fig. 5 of the paper).

A CNON with L hidden layers of N = 50 oscillators (damping alpha = 3, 50 Euler
steps) is trained end-to-end on the full MNIST training set (60 000 images) with
the MSE loss on one-hot targets. After every epoch the network is evaluated on
the 10 000 test images: test accuracy, test MSE of the network's own readout,
and the HTC of each hidden layer computed from the test hidden-layer matrix
H_test. Running it for L = 3 and L = 2 also gives the accuracy comparison
quoted in the text.

Usage:
    python run_backprop.py --layers 3
    python run_backprop.py --layers 2
"""

import argparse
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from cnon import CNON
from htc import hotelling_trace, scatter_matrices

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mnist(root, train):
    ds = datasets.MNIST(root=root, train=train, download=True)
    return ds.data.reshape(len(ds.data), -1).float().div(255), ds.targets   # = ToTensor()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--n-neurons", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=os.path.join(HERE, ".data"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(HERE, "results", "backprop_L{}.npz".format(args.layers))
    X_train, y_train = load_mnist(args.data_root, train=True)
    X_test, y_test = load_mnist(args.data_root, train=False)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    Y_test = F.one_hot(y_test, num_classes=10).float()

    torch.manual_seed(args.seed)
    model = CNON(n_neurons=args.n_neurons, n_layers=args.layers, alpha=args.alpha,
                 dt=0.1, max_steps=args.steps, adaptive=False, train_reservoir=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    hist = {k: [] for k in ["train_loss", "train_accuracy", "test_accuracy", "test_mse",
                            "htc_test", "seconds"]}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred, _, _, _ = model(xb)
            loss = criterion(y_pred, F.one_hot(yb, num_classes=10).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == yb).sum().item()

        model.eval()
        with torch.no_grad():
            y_pred, h, _, _ = model(X_test)
        htc = [hotelling_trace(*scatter_matrices(h_l.double().numpy(), y_test.numpy()))
               for h_l in h]
        row = dict(train_loss=total_loss / len(loader),
                   train_accuracy=correct / len(X_train),
                   test_accuracy=(y_pred.argmax(dim=1) == y_test).float().mean().item(),
                   test_mse=((y_pred - Y_test)**2).mean().item(),
                   htc_test=htc, seconds=time.time() - t0)
        for k, v in row.items():
            hist[k].append(v)
        print("epoch {:2d}  loss={:.4f}  acc train/test={:.2f}/{:.2f}%  MSE test={:.5f}  "
              "HTC per layer={}  ({:.0f} s)".format(
                  epoch, row["train_loss"], 100 * row["train_accuracy"],
                  100 * row["test_accuracy"], row["test_mse"],
                  " ".join("{:.1f}".format(v) for v in htc), row["seconds"]), flush=True)

    meta = dict(layers=args.layers, n_neurons=args.n_neurons, alpha=args.alpha,
                steps=args.steps, dt=0.1, batch_size=args.batch_size, epochs=args.epochs,
                lr=args.lr, seed=args.seed, total_seconds=time.time() - t_start,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              torch=torch.__version__, torchvision=torchvision.__version__),
                threads=torch.get_num_threads(), machine=platform.processor())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, epochs=np.arange(1, args.epochs + 1), meta=json.dumps(meta),
             **{k: np.array(v) for k, v in hist.items()})
    print("saved", out, "({:.1f} min)".format(meta["total_seconds"] / 60))


if __name__ == "__main__":
    main()
