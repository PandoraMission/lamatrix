import numpy as np
from scipy import sparse
from lamatrix import Polynomial


def test_sparse_fits():
    data = np.random.normal(size=200)
    row_coord = np.random.uniform(0, 120, size=(200))
    col_coord = np.zeros(200)
    x = sparse.csr_matrix((data, (row_coord, col_coord)), shape=(121, 1))

    model = Polynomial('x', polyorder=10)
    w = np.random.normal(size=10)
    data = model.design_matrix(x=x.toarray()[:, 0]).dot(w)
    errors = np.ones_like(data) * 0.001
    mask = np.ones(data.shape, bool)
    mask[99] = False

    # Dense design matrix
    model.fit(x=x.toarray()[:, 0], data=data)
    assert np.allclose(model.best_fit.mean, w)
    assert np.allclose(model.to_gradient().design_matrix(x=x.toarray()[:, 0]), model.to_gradient().design_matrix(x=x).toarray())
    model.fit(x=x.toarray()[:, 0], data=data, errors=errors)
    assert np.allclose(model.best_fit.mean, w)
    model.fit(x=x.toarray()[:, 0], data=data, errors=errors, mask=mask)
    assert np.allclose(model.best_fit.mean, w)

    # Sparse design matrix
    model.fit(x=x, data=data)
    assert np.allclose(model.best_fit.mean, w)
    assert np.allclose(model.to_gradient().design_matrix(x=x.toarray()[:, 0]), model.to_gradient().design_matrix(x=x).toarray())
    model.fit(x=x, data=data, errors=errors)
    assert np.allclose(model.best_fit.mean, w)
    model.fit(x=x, data=data, errors=errors, mask=mask)
    assert np.allclose(model.best_fit.mean, w)

    # Sparse data
    model.fit(x=x, data=sparse.csr_matrix(data).T)
    assert np.allclose(model.best_fit.mean, w)
    assert np.allclose(model.to_gradient().design_matrix(x=x.toarray()[:, 0]), model.to_gradient().design_matrix(x=x).toarray())
    model.fit(x=x, data=sparse.csr_matrix(data).T, errors=sparse.csr_matrix(errors).T)
    assert np.allclose(model.best_fit.mean, w)
    model.fit(x=x, data=sparse.csr_matrix(data).T, errors=sparse.csr_matrix(errors).T, mask=mask)
    assert np.allclose(model.best_fit.mean, w)

def test_sparse_combine():
    C, Z = np.mgrid[-30:20:11j, 0:15:9j]
    p1 = Polynomial('Z', polyorder=2)
    p2 = Polynomial('C', polyorder=3)
    model = p1 + p2
    dm = model.design_matrix(Z=Z, C=C)
    dm_sparse = model.design_matrix(Z=sparse.csr_matrix(Z.ravel()).T, C=sparse.csr_matrix(C.ravel()).T)
    assert np.allclose(dm.ravel(), dm_sparse.toarray().ravel())

    model = p1 * p2
    dm = model.design_matrix(Z=Z, C=C)
    dm_sparse = model.design_matrix(Z=sparse.csr_matrix(Z.ravel()).T, C=sparse.csr_matrix(C.ravel()).T)
    assert np.allclose(dm.ravel(), dm_sparse.toarray().ravel())