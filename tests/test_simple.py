import numpy as np
from lamatrix import Polynomial, Constant, Sinusoid

def test_constant():
    g = Constant()
    g
    assert hasattr(g, 'arg_names')
    assert g.arg_names == {}
    dm = g.design_matrix(r=np.arange(10))
    assert dm.shape == (10, 1)
    assert (dm[:, 0] == np.ones(10)).all()
    assert g.width == 1

    x = np.arange(-1, 1, 0.01)
    w = np.random.normal()
    y = w + x
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g = Constant()
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w, atol=0.01)

    g = Constant(prior_distributions=[(w, 0.1)])
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w, atol=0.01)

def test_polynomial():
    g = Polynomial('r', polyorder=3)
    g
    assert hasattr(g, 'arg_names')
    assert g.arg_names == {'r'}
    dm = g.design_matrix(r=np.arange(10))
    assert dm.shape == (10, 3)
    assert (dm[:, 0] == np.arange(10)).all()
    assert g.width == 3

    x = np.arange(-1, 1, 0.01)
    w = np.random.normal()
    y = w * x
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g = Polynomial('x', polyorder=3)
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w, atol=0.01)
    assert np.allclose(g.fit_mean[1:], np.zeros(g.width - 1), atol=0.01)

    g = Polynomial('x', polyorder=3, prior_distributions=[(w, 0.1), (0, np.inf), (0, np.inf)])
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w, atol=0.01)
    assert np.allclose(g.fit_mean[1:], np.zeros(g.width - 1), atol=0.01)


def test_sinusoid():
    g = Sinusoid('phi', nterms=2)
    g
    assert hasattr(g, 'arg_names')
    assert g.arg_names == {'phi'}
    dm = g.design_matrix(phi=np.arange(10))
    assert dm.shape == (10, 4)
    assert (dm[:, 0] == np.sin(np.arange(10))).all()
    assert g.width == 4

    x = np.arange(-1, 1, 0.01)
    w = np.random.normal(size=2)
    y = w[0] * np.sin(x) + w[1] * np.cos(x)
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g = Sinusoid('x', nterms=2)
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w[0], atol=0.01)
    assert np.isclose(g.fit_distributions[1][0], w[1], atol=0.01)
    assert np.allclose(g.fit_mean[2:], np.zeros(g.width - 2), atol=0.01)

    g = Sinusoid('x', nterms=2, prior_distributions=[(w[0], 0.1), (w[1], 0.1), (0, np.inf), (0, np.inf)])
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.fit_distributions[0][0], w[0], atol=0.01)
    assert np.isclose(g.fit_distributions[1][0], w[1], atol=0.01)
    assert np.allclose(g.fit_mean[2:], np.zeros(g.width - 2), atol=0.01)

