import numpy as np
from lamatrix import Polynomial, Constant, Sinusoid, dPolynomial, dSinusoid, Spline, dSpline


def test_constant():
    g = Constant()
    g
    assert hasattr(g, "arg_names")
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
    assert np.isclose(g.best_fit.mean[0], w, atol=0.01)

    g = Constant(priors=[(w, 0.1)])
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.best_fit.mean[0], w, atol=0.01)


def test_polynomial():
    g = Polynomial("r", polyorder=3)
    g
    assert hasattr(g, "arg_names")
    assert g.arg_names == {"r"}
    dm = g.design_matrix(r=np.arange(10))
    assert dm.shape == (10, 3)
    assert (dm[:, 0] == np.arange(10)).all()
    assert g.width == 3

    x = np.arange(-1, 1, 0.01)
    w = np.random.normal()
    y = w * x
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g = Polynomial("x", polyorder=3)
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.best_fit.mean[0], w, atol=0.01)
    assert np.allclose(g.best_fit.mean[1:], np.zeros(g.width - 1), atol=0.01)

    g = Polynomial(
        "x", polyorder=3, priors=[(w, 0.1), (0, np.inf), (0, np.inf)]
    )
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.best_fit.mean[0], w, atol=0.01)
    assert np.allclose(g.best_fit.mean[1:], np.zeros(g.width - 1), atol=0.01)


def test_sinusoid():
    g = Sinusoid("phi", nterms=2)
    g
    assert hasattr(g, "arg_names")
    assert g.arg_names == {"phi"}
    dm = g.design_matrix(phi=np.arange(10))
    assert dm.shape == (10, 4)
    assert (dm[:, 0] == np.sin(np.arange(10))).all()
    assert g.width == 4

    x = np.arange(-1, 1, 0.01)
    w = np.random.normal(size=2)
    y = w[0] * np.sin(x) + w[1] * np.cos(x)
    y += np.random.normal(0, 0.001, size=x.shape[0])
    ye = np.ones_like(x) + 0.001

    g = Sinusoid("x", nterms=2)
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.best_fit.mean[0], w[0], atol=0.01)
    assert np.isclose(g.best_fit.mean[1], w[1], atol=0.01)
    assert np.allclose(g.best_fit.mean[2:], np.zeros(g.width - 2), atol=0.01)

    g = Sinusoid(
        "x",
        nterms=2,
        priors=[(w[0], 0.1), (w[1], 0.1), (0, np.inf), (0, np.inf)],
    )
    g.fit(data=y, errors=ye, x=x)
    assert np.isclose(g.best_fit.mean[0], w[0], atol=0.01)
    assert np.isclose(g.best_fit.mean[1], w[1], atol=0.01)
    assert np.allclose(g.best_fit.mean[2:], np.zeros(g.width - 2), atol=0.01)

def test_shape():
    """Test that we can pass in all sorts of weird shaped vectors"""
    for shape in [(53, ), (53, 5), (53, 5, 3), (53, 5, 3, 2)]:
        x = np.random.normal(size=shape)
        p1 = Polynomial(x_name='x', polyorder=4)

        X = p1.design_matrix(x=x)    
        assert X.shape == (*shape, 4)
        dp1 = p1.to_gradient()
        X = dp1.design_matrix(x=x)
        assert X.shape == (*shape, 1)
        X = dPolynomial(np.arange(4), x_name='x', polyorder=4).design_matrix(x=x)    
        assert X.shape == (*shape, 1)

        s1 = Sinusoid(x_name='x', nterms=4)
        X = s1.design_matrix(x=x)    
        assert X.shape == (*shape, 8)
        ds1 = s1.to_gradient()
        X = ds1.design_matrix(x=x)
        assert X.shape == (*shape, 1)
        X = dSinusoid(np.arange(8), x_name='x', nterms=4).design_matrix(x=x)    
        assert X.shape == (*shape, 1)

        X = Constant().design_matrix(x=x)    
        assert X.shape == (*shape, 1)
        
        sp1 = Spline(x_name='x', knots=np.arange(-2, 2, 0.4))
        X = sp1.design_matrix(x=x)    
        assert X.shape == (*shape, 6)
        dsp1 = sp1.to_gradient()
        X = dsp1.design_matrix(x=x)
        assert X.shape == (*shape, 1)
        X = dSpline(np.arange(6), x_name='x', knots=np.arange(-2, 2, 0.4)).design_matrix(x=x)    
        assert X.shape == (*shape, 1)
