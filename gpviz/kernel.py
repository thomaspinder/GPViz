import gpjax
from gpjax.parameters import initialise
import matplotlib.pyplot as plt
from multipledispatch import dispatch
from gpjax.kernels import Kernel
import jax.numpy as jnp
from .styles import get_colours, get_cmap
import mplcyberpunk
from typing import Tuple
Array = jnp.DeviceArray


@dispatch(Kernel, Array)
def plot(kernel: Kernel, X: Array, params: dict = None, ax = None):
    """
    Plot the kernel's Gram matrix.

    :param kernel: The kernel function that generates the Gram matrix
    :param X: The data points for which the Gram matrix is computed on.
    :param params: A dictionary containing the kernel parameters
    :param ax: An optional matplotlib axes
    :return:
    """
    if params is None:
        params = initialise(kernel)

    cols = get_cmap()
    if ax is None:
        fig, ax = plt.subplots()

    K = gpjax.kernels.gram(kernel, X, params)
    ax.matshow(K, cmap = cols)


@dispatch(Kernel, Array, Array)
def plot(kernel: Kernel, X: Array, Y: Array, params: dict = None, ax=None):
    """
    Plot the kernel's cross-covariance matrix.

    :param kernel: The kernel function that generates the covariance matrix
    :param X: The first set of data points for which the covariance matrix is computed on.
    :param Y: The second set of data points for which the covariance matrix is computed on.
    :param params: A dictionary containing the kernel parameters
    :param ax: An optional matplotlib axes
    :return:
    """
    if params is None:
        params = initialise(kernel)

    cols = get_cmap()
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_tight_layout(False)

    K = gpjax.kernels.cross_covariance(kernel, X, Y, params)
    c = ax.matshow(K, cmap = cols)


@dispatch(Kernel)
def plot(kernel: Kernel, params: dict = None, ax=None, xrange: Tuple[float, float] = (-10, 10.)):
    """
    Plot the kernel's shape.

    :param kernel: The kernel function
    :param params: A dictionary containing the kernel parameters
    :param ax: An optional matplotlib axes
    :param xrange The tuple pair lower and upper values over which the kernel should be evaluated.
    :return:
    """
    if params is None:
        params = initialise(kernel)

    cols = get_colours()
    if ax is None:
        fig, ax = plt.subplots()

    X = jnp.linspace(xrange[0], xrange[1], num=200).reshape(-1, 1)
    x1 = jnp.array([[0.0]])
    K = gpjax.kernels.cross_covariance(kernel, X, x1, params)
    ax.plot(X, K.T, color=cols['base'])
    mplcyberpunk.add_underglow(ax=ax)
