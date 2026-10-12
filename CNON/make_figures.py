"""
Draw the CNON figures of the paper from the saved results.

    figures/fig4_cnon.(png|pdf)  CNON part of Fig. 4: test MSE versus HTC for the
                                 1L CNON on MNIST and fashion-MNIST, with the
                                 random-choice floor, the isotropic lower bound
                                 (Eq. 13) and the spectral description of each
                                 task (Eq. 14) with a fitted offset d.
    figures/fig5_cnon.(png|pdf)  Fig. 5: layer-resolved HTC of the backprop-trained
                                 3-layer CNON (a) versus epoch and (b) against
                                 the test MSE.

Usage:
    python make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import (AutoMinorLocator, FixedLocator, MultipleLocator,  # noqa: E402
                               NullFormatter, ScalarFormatter)

from htc import MSE_CHANCE, eta, fit_offset, mse_isotropic, mse_spectral  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES, name + ".png"), bbox_inches="tight", dpi=300)
    print("saved", os.path.join(FIGURES, name + ".(pdf|png)"))


def figure4():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 14, "axes.labelsize": 14})
    palette = plt.cm.viridis(np.linspace(0, 1, 6))
    H = np.logspace(0, 2, 1000)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(H, mse_isotropic(H), color=plt.cm.viridis(0.2), lw=2.5, ls=":", zorder=4)
    ax.axhline(MSE_CHANCE, color="0.3", ls=":", lw=2.5, zorder=4)
    ax.fill_between(H, mse_isotropic(H), MSE_CHANCE, color="0.95", zorder=1)
    ax.text(10, MSE_CHANCE * 0.8, r"random choice floor $(K-1)/K^2$", color="0.1", fontsize=14)
    ax.text(1.8, 0.048, "lower bound\n" r"$\lambda_i=\mathrm{HTC}/(K-1)$", fontsize=14,
            ha="center", va="bottom", color=plt.cm.viridis(0.2))

    legends = []
    for task, title, color, curve_color, loc in [
            ("mnist", "MNIST", palette[2], "black", (0.03, 0.03)),
            ("fashion", "fashion MNIST", palette[4], plt.cm.viridis(0.5), (0.36, 0.03))]:
        r = np.load(os.path.join(RESULTS, "alpha_sweep_{}.npz".format(task)))
        htc, mse = r["HTC"], r["test_mse"]
        d, d_err, rms, _ = fit_offset(r["lam"], htc, mse)
        ax.scatter(htc, mse, marker="^", s=50, zorder=9, edgecolor=color,
                   facecolor="none", linewidth=1.5)
        Hc = H[(H >= 0.8 * htc.min()) & (H <= 1.2 * htc.max())]
        ax.loglog(Hc, mse_spectral(r["lam"], Hc) + d, ls="--", color=curve_color, lw=2.5,
                  zorder=20)
        handles = [Line2D([], [], marker="^", ls="none", markersize=7, markeredgewidth=1.5,
                          markeredgecolor=color, markerfacecolor="none",
                          label="1L CNON (sim.)"),
                   Line2D([], [], ls="--", lw=2.5, color=curve_color,
                          label=r"spectral fit, $\eta={:.1f}$".format(eta(r["lam"])))]
        legends.append(ax.legend(handles=handles, title=title, loc="lower left",
                                 bbox_to_anchor=loc, fontsize=11, title_fontsize=12,
                                 frameon=False))
        print("Fig. 4 {:8s} eta = {:.2f}   d = {:.5f} +/- {:.5f}   rms = {:.4f} dex".format(
            task, eta(r["lam"]), d, d_err, rms))
    for leg in legends[:-1]:
        ax.add_artist(leg)

    ax.set_xlim(1, 100)
    ax.set_ylim(1e-2, 1e-1)
    ax.set_xlabel("HTC")
    ax.set_ylabel("MSE")
    save(fig, "fig4_cnon")
    plt.close(fig)


def figure5():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 22,
                         "axes.labelsize": 22, "xtick.labelsize": 22, "ytick.labelsize": 22,
                         "legend.fontsize": 18, "axes.unicode_minus": True})
    palette = plt.cm.viridis(np.linspace(0, 1, 5))
    colors = [palette[3], palette[2], palette[1]]          # h. layer 1, 2, 3

    r = np.load(os.path.join(RESULTS, "backprop_L3.npz"))
    epochs, htc, mse = r["epochs"], r["htc_test"].T, r["test_mse"]   # htc: (layers, epochs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for l in (2, 1, 0):
        ax1.plot(epochs, htc[l], marker="^", color=colors[l], ls=":", markersize=12,
                 markeredgewidth=1.5, markerfacecolor="none", label="h. layer {}".format(l + 1))
        ax2.scatter(htc[l], mse, marker="^", s=120, color=colors[l], facecolors="none",
                    linewidths=1.5, label="h. layer {}".format(l + 1))
        slope, intercept = np.polyfit(np.log10(htc[l]), np.log10(mse), 1)
        r_l = np.corrcoef(np.log10(htc[l]), np.log10(mse))[0, 1]
        if l > 0:                                         # dashed power-law fits, layers 2 and 3
            x = np.logspace(np.log10(htc[l].min()), np.log10(htc[l].max()), 200)
            ax2.plot(x, 10**intercept * x**slope, "--", lw=1.5, color=colors[l])
        print("Fig. 5 h. layer {}: MSE = {:.4g} HTC^({:+.3f})   r = {:+.3f}".format(
            l + 1, 10**intercept, slope, r_l))

    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlabel("Training epochs")
    ax1.set_ylabel("HTC")
    ax1.text(0.02, 0.98, "a", transform=ax1.transAxes, fontsize=24, va="top")
    ax1.legend(loc="center left", bbox_to_anchor=(0.45, 0.35), frameon=False, fontsize=18)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("HTC")
    ax2.set_ylabel("MSE")
    ax2.text(0.02, 0.98, "b", transform=ax2.transAxes, fontsize=24, va="top")
    ax2.legend(loc="center left", bbox_to_anchor=(0.45, 0.8), frameon=False, fontsize=18)
    ax1.minorticks_on()
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    for ax in (ax1, ax2):
        ax.tick_params(which="major", length=6, width=1)
        ax.tick_params(which="minor", length=3, width=0.8)

    # axis ranges of the published figure, widened if the data fall outside
    ax1.set_ylim(min(25, 0.9 * htc.min()), max(200, 1.1 * htc.max()))
    ax1.yaxis.set_major_locator(MultipleLocator(50))
    ax1.yaxis.set_minor_locator(MultipleLocator(25))
    ax2.set_xlim(min(30, 0.9 * htc.min()), max(200, 1.1 * htc.max()))
    ax2.set_ylim(min(10**-2.4, 0.9 * mse.min()), max(10**-1.6, 1.1 * mse.max()))
    ax2.xaxis.set_major_locator(FixedLocator([30, 50, 100, 200]))   # less than one decade:
    ax2.xaxis.set_major_formatter(ScalarFormatter())                # label plain numbers
    ax2.xaxis.set_minor_formatter(NullFormatter())
    plt.tight_layout()
    save(fig, "fig5_cnon")
    plt.close(fig)


if __name__ == "__main__":
    figure4()
    figure5()
    for layers in (2, 3):
        r = np.load(os.path.join(RESULTS, "backprop_L{}.npz".format(layers)))
        print("{} hidden layers: final test accuracy {:.2f}%".format(
            layers, 100 * r["test_accuracy"][-1]))
