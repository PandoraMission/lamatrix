import pytest
import numpy as np
from lamatrix import lnGaussian2DGenerator, dlnGaussian2DGenerator, CombinedGenerator, Polynomial1DGenerator




def test_lngauss():
    column, row = np.meshgrid(
        np.arange(-10, 10, 0.2), np.arange(-10, 10, 0.2), indexing="ij"
    )
    A = 0
    rho = 0.2
    sigma_x = 0.8596
    sigma_y = 1.358
    offset = 0

    def Gauss2D(column, row, A, sigma_x, sigma_y, rho):
        return (
            A
            - (column**2) / (2 * (1 - rho**2) * sigma_x**2)
            - (row**2) / (2 * (1 - rho**2) * sigma_y**2)
            + ((2 * rho * column * row) / (2 * (1 - rho**2) * sigma_x * sigma_y))
        )

    data = Gauss2D(column + 0.01, row - 0.02, A, sigma_x, sigma_y, rho)
    data += offset
    data += np.random.normal(0, 0.1, size=row.shape)
    errors = np.ones_like(data) + 0.1

    g = lnGaussian2DGenerator(
        x_name="column",
        y_name="row",
        stddev_x_prior=(1, 0.5),
        stddev_y_prior=(1, 0.5),
        data_shape=data.shape,
    )
    g.fit(column=column, row=row, data=data, errors=errors)
    g.to_latex()
    assert np.isclose(sigma_x, g.stddev_x[0], atol=g.stddev_x[1] * 2)
    assert np.isclose(sigma_y, g.stddev_y[0], atol=g.stddev_y[1] * 2)
    assert np.isclose(rho, g.rho[0], atol=g.rho[1] * 2)

    dg = dlnGaussian2DGenerator(
        x_name="column",
        y_name="row",
        stddev_x=g.stddev_x[0],
        stddev_y=g.stddev_y[0],
        rho=g.rho[0],
        data_shape=data.shape,
    )
    assert np.isclose(-0.01, dg.shift_x[0], atol=dg.shift_x[1]*2)
    assert np.isclose(0.02, dg.shift_y[0], atol=dg.shift_y[1]*2)

    c = CombinedGenerator(g, dg)
    c.fit(column=column, row=row, data=data, errors=errors)
    assert np.isclose(sigma_x, c[0].stddev_x[0], atol=c[0].stddev_x[1] * 2)
    assert np.isclose(sigma_y, c[0].stddev_y[0], atol=c[0].stddev_y[1] * 2)
    assert np.isclose(rho, c[0].rho[0], atol=c[0].rho[1] * 2)
    assert np.isclose(-0.01, c[1].shift_x[0], atol=c[1].shift_x[1]*2)
    assert np.isclose(0.02, c[1].shift_y[0], atol=c[1].shift_y[1]*2)


def test_poly1d():
    x = (np.arange(200) - 200)/200
    x2 = (np.arange(-300, 300, 10) - 200)/200

    data = (3*x + 0.3) + np.random.normal(0, 0.1, size=200)
    s = np.random.choice(np.arange(200), size=5)
    data[s] += np.random.normal(0, 3, size=5)
    errors = np.ones_like(x) * 0.1

    g = Polynomial1DGenerator(polyorder=1)
    g.fit(x=x, data=data, errors=errors)

    outlier_mask = np.abs(data - g.evaluate(x=x))/errors < 3
    g.fit(x=x, data=data, errors=errors, mask=outlier_mask)
    assert np.allclose([0.3, 3], g.mu, atol=g.sigma*2)
    g.evaluate(x=x2)
