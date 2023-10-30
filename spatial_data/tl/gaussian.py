from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def plot_mixture(gmm, X, rug=False, intersection=None, show_legend=False, xmin=-3, xmax=3.0, ax=None):
    if ax is None:
        ax = plt.gca()
    # Compute PDF of whole mixture
    x = np.linspace(xmin, xmax, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    # Plot data histogram
    ax.hist(X, 30, density=True, histtype="stepfilled", alpha=0.4, label="Data")
    # Plot PDF of whole model
    ax.plot(x, pdf, "-k", label="Mixture PDF")

    # Plot PDF of each component
    neg_idx = np.argmin(gmm.means_)
    pos_idx = np.argmax(gmm.means_)
    ax.plot(x, pdf_individual[:, neg_idx], "-", color="C3", label="Component PDF")
    ax.plot(x, pdf_individual[:, pos_idx], "-", color="C2", label="Component PDF")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")

    # if rug:
    # sns.rugplot(data=pd.DataFrame(X, columns=['samples']), x="samples", ax=ax, alpha=1, height=.07, palette='C0', lw=.05)
    # ax.set_ylim(bottom=-.1)

    if intersection is not None:
        ax.axvline(intersection, color="C0", lw=1.0, ls="--")
        ax.axvline(gmm.means_[neg_idx], color="C3", lw=1.0, ls="--")
        ax.axvline(gmm.means_[pos_idx], color="C2", lw=1.0, ls="--")

    if show_legend:
        ax.legend()

    return ax


def critical_pt(model):
    xc = intersect(
        model.means_.reshape(-1)[0],
        model.means_.reshape(-1)[1],
        model.covariances_[0],
        model.covariances_[1],
        model.weights_[0],
        model.weights_[1],
    )

    logprob = model.score_samples(xc.reshape(-1, 1))
    idx = np.where(logprob == np.max(logprob))[0]
    return xc[idx][0]


def intersect(μ1, μ2, σ1, σ2, ϕ1, ϕ2):
    a = σ2 - σ1
    b = 2 * (μ2 * σ1 - μ1 * σ2)
    c = μ1**2 * σ2 - μ2**2 * σ1 - 2 * σ1 * σ2 * (np.log(ϕ1 / np.sqrt(σ1)) - np.log((ϕ2) / np.sqrt(σ2)))
    sol = np.array([(-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a), (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)])

    return sol


def sigmoid(x, xc, temp=1.0):
    return 1 / (1.0 + np.exp(-temp * (x - xc)))


class TwoComponentGaussianMixture:
    def __init__(self, sdata, layer_key="_intensity", temp=1.0):
        self.adata = sdata
        self.layer_key = layer_key
        self.models = {}
        self.intersect = {}
        self.sigmoid = {}
        self.temp = temp

    def fit(self):
        """
        Fits a bimodal gaussian mixture to each channel and
        and computes the intersection
        """
        self.transformed = {}

        for marker in tqdm(self.adata.coords["channels"].values.tolist()):
            expression = self.adata[self.layer_key].sel({"channels": marker}).values.reshape(-1, 1)
            model = GaussianMixture(n_components=2, max_iter=200, covariance_type="spherical", n_init=1)
            model.fit(expression)
            self.models[marker] = model

            try:
                xc = critical_pt(model)
            except ValueError:
                xc = 0

            self.intersect[marker] = xc
            self.sigmoid[marker] = partial(sigmoid, xc=xc)

    def transform(self, temp=1.0, layer_key_added=None):
        transformed = {}
        for marker in tqdm(self.adata.coords["channels"].values.tolist()):
            expression = self.adata[self.layer_key].sel({"channels": marker}).values.squeeze()
            transformed[marker] = self.sigmoid[marker](expression, temp=temp).reshape(-1)  # transformed expression

        df = pd.DataFrame(transformed)

        if layer_key_added is not None:
            self.adata.layers[layer_key_added] = df.loc[:, self.adata.var.index.tolist()].values

        return df

    def transform_softmax(self, temp=1.0, layer_key_added=None):
        transformed = {}
        for marker in tqdm(self.adata.coords["channels"].values.tolist()):
            expression = self.adata[self.layer_key].sel({"channels": marker}).values.squeeze()
            transformed[marker] = self.sigmoid[marker](expression, temp=temp).reshape(-1)  # transformed expression

        df = pd.DataFrame(transformed)

        if layer_key_added is not None:
            self.adata.layers[layer_key_added] = softmax(df.loc[:, self.adata.var.index.tolist()].values, 1)

        return df

    def plot_results(
        self, markers=None, show_sigmoid=False, temp=None, ncols=6, width=4, height=3, xmin=-0.5, xmax=4.5
    ):
        if isinstance(markers, str):
            markers = [markers]

        if markers is not None:
            sub = self.adata[self.layer_key].sel({"channels": markers})
        else:
            sub = self.adata

        temp = temp if temp is not None else self.temp

        nrows, remainder = divmod(sub.shape[1], ncols)
        if remainder > 0:
            nrows += 1

        ncols = min([len(markers), ncols])

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))

        if ncols == 1 and nrows == 1:
            axes = np.array([axes])

        for marker, ax in zip(sub.coords["channels"].values.tolist(), axes.flatten()):
            expression = sub.sel({"channels": marker}).values.squeeze()

            plot_mixture(
                self.models[marker],
                expression.reshape(-1),
                self.intersect[marker],
                show_legend=False,
                xmin=xmin,
                xmax=xmax,
                ax=ax,
            )
            diff = np.abs(np.diff(self.models[marker].means_.squeeze()))[0]
            ax.set_title(f"{marker} ($x_c=${self.intersect[marker]:.2f}, $\\delta=${diff:.2f})")
            if show_sigmoid:
                ax2 = ax.twinx()
                x = np.linspace(xmin, xmax, 1000)
                ax2.axhline(0.5, color="k", lw=0.5, ls="--")
                ax2.plot(x, self.sigmoid[marker](x, temp=temp), color="C3")
                ax2.set_ylim([0, 1])
                ax2.tick_params(axis="y", which="major", colors="C3", labelcolor="C3")
                ax2.set_ylabel("σ(x)", color="C3")

        _ = [ax.axis("off") for ax in axes[sub.coords["channels"].values.shape[0] :]]
        plt.tight_layout()
