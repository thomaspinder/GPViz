import gpjax
import gpviz as gpv
import gpjax.core as gpx
import jax.numpy as jnp
import jax.random as jr


def test_kernel():
    kern = gpx.RBF()
    params = gpx.initialise(kern)
    X = jnp.linspace(-1., 1., 10).reshape(-1, 1)
    Y = jnp.linspace(-1., 1., 10).reshape(-1, 1)
    gpv.plot(kern, params=params)
    gpv.plot(kern, X, params=params)
    gpv.plot(kern, X, Y, params=params)