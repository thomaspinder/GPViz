import gpjax
import gpjax.core as gpx
from gpjax.parameters import initialise
import matplotlib.pyplot as plt
from multipledispatch import dispatch
from gpjax.kernels import Kernel
import jax.numpy as jnp
from .styles import get_colours, get_cmap
import mplcyberpunk
from typing import Tuple
Array = jnp.DeviceArray


@dispatch(Kernel, Array, dict)
def plot(kernel: Kernel, X: Array, params: dict, ax = None):
    if dict is None:
        params = initialise(kernel)

    cols = get_cmap()
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(False)

    K = gpjax.kernels.gram(kernel, X, params)
    ax.matshow(K, cmap = cols)


@dispatch(Kernel, Array, Array, dict)
def plot(kernel: Kernel, X: Array, Y: Array, params: dict = None, ax=None):
    if dict is None:
        params = initialise(kernel)

    cols = get_cmap()
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(False)

    K = gpjax.kernels.cross_covariance(kernel, X, Y, params)
    c = ax.matshow(K, cmap = cols)


@dispatch(Kernel)
def plot(kernel: Kernel, params: dict = None, ax=None, xrange: Tuple[float, float] = (-10, 10.)):
    if dict is None:
        params = initialise(kernel)

    cols = get_colours()
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(False)

    X = jnp.linspace(xrange[0], xrange[1], num=200).reshape(-1, 1)
    x1 = jnp.array([[0.0]])
    K = gpjax.kernels.cross_covariance(kernel, X, x1, params)
    ax.plot(X, K.T, color=cols['base'])
    mplcyberpunk.add_underglow(ax=ax)

