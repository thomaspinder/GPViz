import gpjax
import gpviz as gpv
import gpjax.core as gpx
import jax.numpy as jnp
import jax.random as jr


def test_prior_plot():
    key = jr.PRNGKey(123)
    x = jnp.linspace(-2., 2., 100).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key, shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)
    f = gpx.Prior(kernel=gpx.RBF())
    params = gpx.initialise(f)
    fig, ax = gpv.plot(key, f, params, D, n_samples=10, title='Prior draws')
    assert fig is not None
    assert ax is not None

    fig, ax = gpv.plot(key, f, params, x, n_samples=10, title='Prior draws')
    assert fig is not None
    assert ax is not None


def test_posterior_plot():
    key = jr.PRNGKey(123)
    x = jnp.linspace(-2., 2., 100).reshape(-1, 1)
    y = jnp.sin(x) + jr.normal(key, shape=x.shape) * 0.1
    D = gpx.Dataset(X=x, y=y)
    f = gpx.Prior(kernel=gpx.RBF())
    fx = f * gpx.Gaussian()
    params = gpx.initialise(fx)
    testing = gpx.Dataset(X=jnp.linspace(-2.2, 2.2, 100).reshape(-1, 1))
    fig, ax = gpv.plot(key, fx, params, D, jnp.linspace(-2.2, 2.2, 100).reshape(-1, 1))
    assert fig is not None
    assert ax is not None
    fig, ax = gpv.plot(key, fx, params, D, testing)
    assert fig is not None
    assert ax is not None