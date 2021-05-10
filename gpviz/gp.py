from typing import List
import gpjax.core as gpx

import jax.numpy as jnp
import matplotlib.pyplot as plt
from gpjax.gps import (
    Posterior,
    ConjugatePosterior,
    SpectralPosterior,
    NonConjugatePosterior,
)
from multipledispatch import dispatch
from .styles import get_colours
from .utils import tidy_legend, glow

Array = jnp.DeviceArray

####################
# Prior plotting
####################
@dispatch(Array, gpx.Prior, dict, gpx.Dataset)
def plot(
    key: Array,
    gp: gpx.Prior,
    params: dict,
    data: gpx.Dataset,
    n_samples: int = 10,
    title: str = None,
    ax=None,
):
    """
    Plot samples from the Gaussian process prior distribution.

    :param key: A Jax PRNGKey object to ensure reproducibility when sampling from the prior distribution.
    :param gp: A generic Gaussian process prior
    :param params: The Gaussian process priors's corresponding parameter set.
    :param data: The training dataset
    :param n_samples: The number of samples to be drawn from the predictive posterior's distribution. The default argument is 0 which corresponds to no samples being plotteed.
    :param title: What title, if any, should be added to the plot.
    :param ax: Optional matplotlib axes argument.
    :return:
    """
    samples = gpx.sample(key, gp, params, data, n_samples=n_samples)
    cols = get_colours()
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(data.X, samples.T, alpha=0.3, color=cols["base"])
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_xlim(jnp.min(data.X), jnp.max(data.X))
    ax.set_title(title, loc="left")


@dispatch(Array, gpx.Prior, dict, Array)
def plot(
    key: Array,
    gp: gpx.Prior,
    params: dict,
    data: Array,
    n_samples: int = 10,
    title: str = None,
):
    """
    Plot samples from the Gaussian process prior distribution.

    :param key: A Jax PRNGKey object to ensure reproducibility when sampling from the prior distribution.
    :param gp: A generic Gaussian process prior
    :param params: The Gaussian process priors's corresponding parameter set.
    :param data: The training dataset
    :param n_samples: The number of samples to be drawn from the predictive posterior's distribution. The default argument is 0 which corresponds to no samples being plotteed.
    :param title: What title, if any, should be added to the plot.
    :param ax: Optional matplotlib axes argument.
    :return:
    """
    D = gpx.Dataset(X=data)
    return plot(key, gp, params, D, n_samples=n_samples, title=title)


######################
# Posterior plotting
######################
@dispatch(
    Array,
    (Posterior, ConjugatePosterior, SpectralPosterior, NonConjugatePosterior),
    dict,
    gpx.Dataset,
    Array,
)
def plot(
    key: Array,
    gp: Posterior,
    params: dict,
    data: gpx.Dataset,
    testing: Array,
    n_samples: int = 0,
    mean: bool = True,
    glow_mean: bool = True,
    std_devs: List[int] = [1],
    title: str = None,
    legend:bool=False,
    ax=None
):
    """
    Create a plot of the Gaussian process' predictive posterior distribution.

    :param key: A Jax PRNGKey object to ensure reproducibility when sampling from the posterior distribution.
    :param gp: A generic Gaussian process posterior
    :param params: The Gaussian process posterior's corresponding parameter set.
    :param data: The training dataset
    :param testing: The testing dataset array.
    :param n_samples: The number of samples to be drawn from the predictive posterior's distribution. The default argument is 0 which corresponds to no samples being plotteed.
    :param mean: Boolean as to whether the predictive mean should be plotted.
    :param glow_mean: Boolean as to whether the predictive mean should line should be glowed.
    :param std_devs: The number of posterior standard deviation bands to be plotted.
    :param title: What title, if any, should be added to the plot.
    :param legend: Boolean as to whether the legend should be added to the plot.
    :param ax: Optional matplotlib axes argument.
    :return:
    """
    rv = gpx.random_variable(gp, params, data, jitter_amount=1e-6)(testing)
    if ax is None:
        fig, ax = plt.subplots()

    mu = rv.mean()
    sigma = rv.variance()
    one_stddev = jnp.sqrt(sigma)

    cols = get_colours()

    ax.plot(data.X, data.y, "o", color=cols["dark_gray"], label="Training data")
    if n_samples > 0:
        if not mean and not std_devs:
            col = cols['base']
            alph = 0.6
            width = 1.
        else:
            col = cols['base']
            alph = 0.4
            width = 0.5
        posterior_samples = gpx.sample(key, rv, n_samples=n_samples)
        ax.plot(
            testing,
            posterior_samples.T,
            color=col,
            alpha=alph,
            linewidth=width,
            label='Posterior samples'
        )
    for i in std_devs:
        ax.fill_between(
            testing.ravel(),
            mu.ravel() - i * one_stddev,
            mu.ravel() + i * one_stddev,
            alpha=0.4 / i,
            color=cols["cool_gray"],
            label=f"{i} standard deviation",
        )
    if std_devs == [1]:
        ax.plot(testing, mu.ravel() - one_stddev, linestyle="--", color=cols["base"])
        ax.plot(testing, mu.ravel() + one_stddev, linestyle="--", color=cols["base"])

    if mean:
        mu_line = ax.plot(testing, mu, color=cols["base"], label="Predictive mean", linewidth=5)
        if glow_mean:
            glow(mu_line, ax)

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    xmin = jnp.min(data.X)
    xmax = jnp.max(data.X)
    ax.set_xlim(xmin - 0.05*jnp.abs(xmin), xmax + 0.05*jnp.abs(xmax))
    ax.set_title(title, loc="left")

    if legend:
        # Remove duplicated labels
        ax = tidy_legend(ax)


@dispatch(
    Array,
    (Posterior, ConjugatePosterior, SpectralPosterior, NonConjugatePosterior),
    dict,
    gpx.Dataset,
    gpx.Dataset
)
def plot(
    key: Array,
    gp: Posterior,
    params: dict,
    data: gpx.Dataset,
    testing: gpx.Dataset,
    n_samples: int = 0,
    mean: bool = True,
    glow_mean: bool = True,
    std_devs: List[int] = [1],
    title: str = None,
    legend:bool=False,
    ax=None
):
    """
    Create a plot of the Gaussian process' predictive posterior distribution.

    :param key: A Jax PRNGKey object to ensure reproducibility when sampling from the posterior distribution.
    :param gp: A generic Gaussian process posterior
    :param params: The Gaussian process posterior's corresponding parameter set.
    :param data: The training dataset
    :param testing: The testing dataset.
    :param n_samples: The number of samples to be drawn from the predictive posterior's distribution. The default argument is 0 which corresponds to no samples being plotteed.
    :param mean: Boolean as to whether the predictive mean should be plotted.
    :param glow_mean: Boolean as to whether the predictive mean should line should be glowed.
    :param std_devs: The number of posterior standard deviation bands to be plotted.
    :param title: What title, if any, should be added to the plot.
    :param legend: Boolean as to whether the legend should be added to the plot.
    :param ax: Optional matplotlib axes argument.
    :return:
    """
    xstar = testing.X
    return plot(key, gp, params, data, xstar, n_samples=n_samples, mean=mean, glow_mean=glow_mean, std_devs=std_devs, title=title, legend=legend, ax=ax)