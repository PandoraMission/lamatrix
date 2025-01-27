import numpy as np

from lamatrix import Constant, Polynomial, Sinusoid, Spline


def test_simple_fits():
    """Build some simple models for data, check that fitting works"""
    R, C, Z = np.mgrid[-10:10:50j, -30:20:100j, 0:15:10j]
    sp1 = Spline("R", knots=np.linspace(-10, 10, 10))
    ls1 = Sinusoid("C", nterms=1)
    p1 = Polynomial("Z", polyorder=2)
    model = ls1 + p1 + Constant()
    w = np.random.normal(size=model.width)
    data = model.design_matrix(Z=Z, C=C, R=R).dot(w)
    model.fit(data=data, Z=Z, C=C, R=R)
    assert np.allclose(w, model.posteriors.mean, rtol=1e-5)

    model = ls1 + p1 + sp1
    w = np.random.normal(size=model.width)
    data = model.design_matrix(Z=Z, C=C, R=R).dot(w)
    model.fit(data=data, Z=Z, C=C, R=R)
    assert np.allclose(w, model.posteriors.mean, rtol=1e-5)

    model = ls1 * p1
    w = np.random.normal(size=model.width)
    data = model.design_matrix(Z=Z, C=C, R=R).dot(w)
    model.fit(data=data, Z=Z, R=R, C=C)
    assert np.allclose(w, model.posteriors.mean, rtol=1e-5)

    model = ls1 * p1 + sp1
    w = np.random.normal(size=model.width)
    data = model.design_matrix(Z=Z, R=R, C=C).dot(w)
    model.fit(data=data, Z=Z, R=R, C=C)
    assert np.allclose(w, model.posteriors.mean, rtol=1e-5)
