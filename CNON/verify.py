"""
Compare the reproduced CNON results with the published values.

The published numbers (reference/published_cnon.json) come from independent,
unseeded runs, so a new run can only agree within the run-to-run spread. The
tolerances are therefore set at about the spread observed between the published
runs themselves (HTC: median 1.5-2.8 %, max 9-17 %; MSE: median 1.4-1.7 %,
max 6-7 %). Fig. 5 has a single published run, so its criteria are qualitative
features stated in the paper plus generous numerical bands.

Writes results/verification.md and figures/check_fig4.png, check_fig5.png.

Usage:
    python verify.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from htc import eta, fit_offset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REF = json.load(open(os.path.join(HERE, "reference", "published_cnon.json"), encoding="utf-8"))

rows = []


def check(name, published, reproduced, criterion, ok):
    rows.append((name, published, reproduced, criterion, "PASS" if ok else "DEVIATION"))


def loglog_fit(htc, mse):
    x, y = np.log10(htc), np.log10(mse)
    return np.polyfit(x, y, 1)[0], np.corrcoef(x, y)[0, 1]


def figure4_checks():
    alpha = np.array(REF["alpha"])
    settings = {"mnist": ("D_v36", (-0.70, -0.56), -0.97),
                "fashion": ("C_v36", (-0.44, -0.27), -0.95)}
    for task, (ref_run, slope_band, r_max) in settings.items():
        r = np.load(os.path.join(RESULTS, "alpha_sweep_{}.npz".format(task)))
        runs = {k: v for k, v in REF["fig4"][task].items() if "htc" in v}
        pub = runs[ref_run]
        htc_p, mse_p = np.array(pub["htc"]), np.array(pub["mse_test"])
        T = "Fig. 4 {} ".format("MNIST" if task == "mnist" else "fMNIST")

        dh, dm = np.abs(r["HTC"] / htc_p - 1), np.abs(r["test_mse"] / mse_p - 1)
        check(T + "HTC(alpha) median/max abs. rel. diff", "run " + ref_run,
              "{:.1f}% / {:.1f}%".format(100 * np.median(dh), 100 * dh.max()),
              "<= 4% / 20%", np.median(dh) <= 0.04 and dh.max() <= 0.20)
        check(T + "MSE(alpha) median/max abs. rel. diff", "run " + ref_run,
              "{:.1f}% / {:.1f}%".format(100 * np.median(dm), 100 * dm.max()),
              "<= 3% / 10%", np.median(dm) <= 0.03 and dm.max() <= 0.10)

        max_htc = [max(v["htc"]) for v in runs.values()]
        min_mse = [min(v["mse_test"]) for v in runs.values()]
        check(T + "max HTC", "{:.1f}-{:.1f}".format(min(max_htc), max(max_htc)),
              "{:.1f}".format(r["HTC"].max()), "range +/- 5%",
              0.95 * min(max_htc) <= r["HTC"].max() <= 1.05 * max(max_htc))
        check(T + "min test MSE", "{:.4f}-{:.4f}".format(min(min_mse), max(min_mse)),
              "{:.4f}".format(r["test_mse"].min()), "range +/- 5%",
              0.95 * min(min_mse) <= r["test_mse"].min() <= 1.05 * max(min_mse))
        a_opt = alpha[np.argmin(r["test_mse"])]
        check(T + "optimal alpha (min MSE)",
              ", ".join("{:.2f}".format(alpha[np.argmin(v["mse_test"])]) for v in runs.values()),
              "{:.2f}".format(a_opt), "0.5-0.9", 0.5 <= a_opt <= 0.9)

        slopes = [loglog_fit(np.array(v["htc"]), np.array(v["mse_test"])) for v in runs.values()]
        s, rr = loglog_fit(r["HTC"], r["test_mse"])
        check(T + "log-log slope / Pearson r",
              ", ".join("{:.3f}/{:.3f}".format(*sv) for sv in slopes),
              "{:.3f} / {:.3f}".format(s, rr),
              "[{}, {}] / <= {}".format(slope_band[0], slope_band[1], r_max),
              slope_band[0] <= s <= slope_band[1] and rr <= r_max)

        e = eta(r["lam"])
        d, _, rms, _ = fit_offset(r["lam"], r["HTC"], r["test_mse"])
        if task == "mnist":
            e_pub = eta(np.array(REF["fig4"]["mnist"]["spectra_B"]["p"]))
            check(T + "eta (Eq. 27), CNON spectra", "{:.2f} (run B)".format(e_pub),
                  "{:.2f}".format(e), "+/- 10%", abs(e / e_pub - 1) <= 0.10)
        else:
            sp = REF["fig4"]["fashion"]["spectra_unlabelled"]
            e0, e1 = eta(np.array(sp["p0"])), eta(np.array(sp["p1"]))
            closest = "p0" if abs(e - e0) < abs(e - e1) else "p1"
            check(T + "eta (Eq. 27), CNON spectra",
                  "{:.2f} (p0) / {:.2f} (p1), unlabelled".format(e0, e1),
                  "{:.2f} (closest: {})".format(e, closest), "+/- 10% of closest",
                  abs(e / (e0 if closest == "p0" else e1) - 1) <= 0.10)
        check(T + "spectral fit: offset d / rms", "-",
              "{:.5f} / {:.4f} dex".format(d, rms), "rms <= 0.03 dex", rms <= 0.03)


def figure5_checks():
    r3 = np.load(os.path.join(RESULTS, "backprop_L3.npz"))
    r2 = np.load(os.path.join(RESULTS, "backprop_L2.npz"))
    htc, mse = r3["htc_test"].T, r3["test_mse"]                 # (layers, epochs)
    htc_p, mse_p = np.array(REF["fig5"]["htc_test_per_layer"]), np.array(REF["fig5"]["mse_test"])

    peak = int(np.argmax(htc[0])) + 1
    drop = 1 - htc[0, -1] / htc[0].max()
    peak_p = int(np.argmax(htc_p[0])) + 1
    drop_p = 1 - htc_p[0, -1] / htc_p[0].max()
    check("Fig. 5 layer 1: peak epoch / drop to epoch 20",
          "{} / {:.0f}%".format(peak_p, 100 * drop_p), "{} / {:.0f}%".format(peak, 100 * drop),
          "<= 5 / >= 15%", peak <= 5 and drop >= 0.15)
    for l in (1, 2):
        growth = htc[l, -1] / htc[l, 0]
        plateau = abs(htc[l, -5:].mean() / htc[l, -10:-5].mean() - 1)
        check("Fig. 5 layer {}: HTC(20)/HTC(1), last-5 vs previous-5".format(l + 1),
              "{:.1f}x / {:.1f}%".format(htc_p[l, -1] / htc_p[l, 0],
                                        100 * abs(htc_p[l, -5:].mean() / htc_p[l, -10:-5].mean() - 1)),
              "{:.1f}x / {:.1f}%".format(growth, 100 * plateau), ">= 2x / <= 10%",
              growth >= 2 and plateau <= 0.10)
    ratio, ratio_p = htc[2, -1] / htc[0, -1], htc_p[2, -1] / htc_p[0, -1]
    check("Fig. 5 HTC layer 3 / layer 1 at epoch 20", "{:.2f}".format(ratio_p),
          "{:.2f}".format(ratio), "3-7", 3 <= ratio <= 7)
    check("Fig. 5 test MSE at epoch 20", "{:.4f}".format(mse_p[-1]), "{:.4f}".format(mse[-1]),
          "+/- 25%", abs(mse[-1] / mse_p[-1] - 1) <= 0.25)
    for l in range(3):
        _, rr = loglog_fit(htc[l], mse)
        _, rr_p = loglog_fit(htc_p[l], mse_p)
        if l == 0:
            check("Fig. 5 layer 1: log-log r (deviates from power law)", "{:+.3f}".format(rr_p),
                  "{:+.3f}".format(rr), "> -0.95", rr > -0.95)
        else:
            check("Fig. 5 layer {}: log-log r".format(l + 1), "{:+.3f}".format(rr_p),
                  "{:+.3f}".format(rr), "<= -0.95", rr <= -0.95)
    for layers, r, ref in [(2, r2, REF["text"]["test_accuracy_2_layers"]),
                           (3, r3, REF["text"]["test_accuracy_3_layers"])]:
        acc = r["test_accuracy"][-1]
        check("Text: test accuracy, {} hidden layers".format(layers), "{:.1f}%".format(100 * ref),
              "{:.2f}%".format(100 * acc), "+/- 0.5%", abs(acc - ref) <= 0.005)
    gain = r3["test_accuracy"][-1] - r2["test_accuracy"][-1]
    check("Text: accuracy gain 2 -> 3 layers", "+0.5%", "{:+.2f}%".format(100 * gain),
          "> 0", gain > 0)


def check_figures():
    plt.rcParams.update({"font.size": 11})
    alpha = np.array(REF["alpha"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for row, task, title in [(0, "mnist", "MNIST"), (1, "fashion", "fashion MNIST")]:
        r = np.load(os.path.join(RESULTS, "alpha_sweep_{}.npz".format(task)))
        for name, v in REF["fig4"][task].items():
            if "htc" not in v:
                continue
            kw = dict(marker="o", ms=4, ls="none", alpha=0.7, label="published " + name)
            axs[row, 0].semilogx(alpha, v["htc"], **kw)
            axs[row, 1].semilogx(alpha, v["mse_test"], **kw)
            axs[row, 2].loglog(v["htc"], v["mse_test"], **kw)
        kw = dict(color="k", marker="^", ms=6, mfc="none", label="reproduced (seed 0)")
        axs[row, 0].semilogx(alpha, r["HTC"], **kw)
        axs[row, 1].semilogx(alpha, r["test_mse"], **kw)
        axs[row, 2].loglog(r["HTC"], r["test_mse"], ls="none", **kw)
        axs[row, 0].set_ylabel(title + "\nHTC")
        axs[row, 1].set_ylabel("test MSE")
        axs[row, 2].set_ylabel("test MSE")
        axs[row, 0].set_xlabel(r"damping $\alpha$")
        axs[row, 1].set_xlabel(r"damping $\alpha$")
        axs[row, 2].set_xlabel("HTC")
        axs[row, 2].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "check_fig4.png"), dpi=150)
    plt.close(fig)

    r3 = np.load(os.path.join(RESULTS, "backprop_L3.npz"))
    r2 = np.load(os.path.join(RESULTS, "backprop_L2.npz"))
    htc_p = np.array(REF["fig5"]["htc_test_per_layer"])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
    for l, c in zip(range(3), ["C2", "C1", "C0"]):
        ax1.plot(r3["epochs"], htc_p[l], "o--", color=c, mfc="none",
                 label="published h. layer {}".format(l + 1))
        ax1.plot(r3["epochs"], r3["htc_test"][:, l], "^-", color=c,
                 label="reproduced h. layer {}".format(l + 1))
    ax2.plot(r3["epochs"], REF["fig5"]["mse_test"], "o--", color="k", mfc="none", label="published 3L")
    ax2.plot(r3["epochs"], r3["test_mse"], "^-", color="k", label="reproduced 3L")
    ax2.plot(r2["epochs"], r2["test_mse"], "s-", color="0.6", label="reproduced 2L")
    ax3.plot(r3["epochs"], 100 * r3["test_accuracy"], "^-", color="k", label="reproduced 3L")
    ax3.plot(r2["epochs"], 100 * r2["test_accuracy"], "s-", color="0.6", label="reproduced 2L")
    for y in (96.3, 96.8):
        ax3.axhline(y, ls=":", color="0.4")
    ax1.set_ylabel("HTC (test)")
    ax2.set_ylabel("test MSE")
    ax3.set_ylabel("test accuracy (%)")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", "check_fig5.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure4_checks()
    figure5_checks()
    check_figures()
    widths = [max(len(str(r[i])) for r in rows) for i in range(5)]
    header = ("check", "published", "reproduced", "criterion", "status")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    n_pass = sum(r[4] == "PASS" for r in rows)
    summary = "{} of {} checks pass".format(n_pass, len(rows))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(summary)
    with open(os.path.join(RESULTS, "verification.md"), "w", encoding="utf-8") as f:
        f.write("# Verification of the reproduced CNON results\n\n"
                "Generated by `verify.py`; published values from "
                "`reference/published_cnon.json`.\n\n" + "\n".join(lines) + "\n\n" + summary + "\n")
