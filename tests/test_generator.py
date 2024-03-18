import numpy as np

from lamatrix import (BoundedGenerator, Polynomial1DGenerator,
                      SinusoidGenerator, Spline1DGenerator,
                      StackedIndependentGenerator, dlnGaussian2DGenerator,
                      lnGaussian2DGenerator, load)


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
    assert np.isclose(-0.01, dg.shift_x[0], atol=dg.shift_x[1] * 2)
    assert np.isclose(0.02, dg.shift_y[0], atol=dg.shift_y[1] * 2)

    c = StackedIndependentGenerator(g, dg)
    c.fit(column=column, row=row, data=data, errors=errors)
    assert np.isclose(sigma_x, c[0].stddev_x[0], atol=c[0].stddev_x[1] * 2)
    assert np.isclose(sigma_y, c[0].stddev_y[0], atol=c[0].stddev_y[1] * 2)
    assert np.isclose(rho, c[0].rho[0], atol=c[0].rho[1] * 2)
    assert np.isclose(-0.01, c[1].shift_x[0], atol=c[1].shift_x[1] * 2)
    assert np.isclose(0.02, c[1].shift_y[0], atol=c[1].shift_y[1] * 2)


def test_poly1d():
    x = (np.arange(200) - 200) / 200
    x2 = (np.arange(-300, 300, 10) - 200) / 200

    data = 4.658 * x**3 + 1.23 * x**2 + 3 * x + 0.3
    true_g = np.gradient(data, x)
    data += np.random.normal(0, 0.01, size=200)
    # s = np.random.choice(np.arange(200), size=5)
    # data[s] += np.random.normal(0, 3, size=5)
    errors = np.ones_like(x) * 0.01

    g = Polynomial1DGenerator(polyorder=3)
    g.fit(x=x, data=data, errors=errors)

    outlier_mask = np.abs(data - g.evaluate(x=x)) / errors < 3
    g.fit(x=x, data=data, errors=errors, mask=outlier_mask)
    # assert np.allclose([0.3, 3, 1.23, 4.658], g.mu, atol=g.sigma * 2)
    g.evaluate(x=x2)

    dg = g.gradient
    model_g = g.gradient.design_matrix(x=x).dot([1, 1, 1])
    analytical_grad = 3 * 4.658 * x**2 + 2 * 1.23 * x**1 + 3
    np.allclose(true_g[1:-1], analytical_grad[1:-1], model_g[1:-1], atol=0.001)


def test_sinusoid():
    true_w = np.random.normal(size=3)
    x = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
    p = SinusoidGenerator()
    data = p.design_matrix(x=x).dot(true_w)
    data += np.random.normal(0, 0.05, size=len(x))
    errors = np.zeros(len(x)) + 0.05
    p.fit(x=x, data=data, errors=errors)
    np.allclose(true_w, p.mu, atol=p.sigma * 3)

    true_g = np.gradient(p.design_matrix(x=x).dot(true_w), x)
    g = p.gradient.design_matrix(x=x)[:, 1:].dot([1, 1])

    assert np.allclose(true_g[1:-1], g[1:-1], atol=0.005)


def test_polycombine():
    c, r = np.meshgrid(np.arange(-10, 10, 0.2), np.arange(-10, 10, 0.2), indexing="ij")

    p1 = Polynomial1DGenerator("r", data_shape=c.shape)
    p2 = Polynomial1DGenerator("c", data_shape=c.shape)
    for p in [StackedIndependentGenerator(p1, p2), (p1 + p2)]:
        true_w = np.random.normal(0, 1, size=(p.width))
        data = p.design_matrix(c=c, r=r).dot(true_w).reshape(r.shape)
        errors = np.ones_like(r) + 10
        data += np.random.normal(0, 10, size=r.shape)
        p.fit(c=c, r=r, data=data, errors=errors)

        a = list(set(list(np.arange(8))) - set([0, 4]))
        np.allclose(true_w[a], p.mu[a], atol=p.sigma[a])
        np.isclose((true_w[0] + true_w[4]), p.mu[0], atol=p.sigma[0] * 2)


def test_spline():
    x = np.linspace(0, 10, 100)  # Independent variable
    true_g = np.gradient(np.sin(2 * x), x)
    y = np.random.normal(0, 0.1, 100) + np.sin(
        2 * (x)
    )  # Dependent variable, replace with your time-series data
    ye = np.ones(100) * 0.1
    k = 4
    model = Spline1DGenerator(
        knots=np.linspace(x.min(), x.max(), 20), offset_prior=(0, 0.01), splineorder=k
    )
    model.fit(x=x, data=y, errors=ye)
    model(x=np.linspace(-3, 13, 310))

    y2 = np.random.normal(0, 0.01, 100) + np.sin(2 * (x + 0.2))
    dmodel = model.gradient
    dmodel.fit(x=x, data=y2 - model.evaluate(x=x), errors=ye)

    assert np.abs((dmodel.shift_x[0] - 0.2)) / dmodel.shift_x[1] < 10
    assert np.allclose(
        true_g[10:-10], dmodel.design_matrix(x=x)[:, 1:].dot([1])[10:-10], atol=1
    )

    s1 = Spline1DGenerator(knots=np.arange(-10, 10, 3), x_name="x")
    s2 = Spline1DGenerator(knots=np.arange(-10, 10, 3), x_name="y")
    model = s1 * s2

    x, y = np.mgrid[-10:10, -10:10]

    true_w = np.random.normal(0, 1, size=model.width)
    data = model.design_matrix(x=x, y=y).dot(true_w).reshape(x.shape)
    model.fit(x=x, y=y, data=data)

    assert np.allclose(true_w, model.fit_mu)
    assert np.allclose(true_w, model.mu)


def test_save():
    p1 = Polynomial1DGenerator("c", polyorder=2)
    p2 = Polynomial1DGenerator("r")
    p = p1 + p2
    p.save("test.json")
    p = load("test.json")
    assert p[0].x_name == "c"
    assert p[1].x_name == "r"
    assert p[0].polyorder == 2


def test_bounded():
    p = Polynomial1DGenerator("x", polyorder=2, prior_sigma=[100, 100, 100])
    bg = BoundedGenerator(p, slice(-10, 10, 5))
    x = np.arange(-10, 10, 0.1)
    true_w = np.random.normal(0, 1, size=3)
    data = p.design_matrix(x=x).dot(true_w)
    bg.fit(x=x, data=data)
    assert np.allclose(
        bg.fit_mu.reshape((bg.nbounds, len(bg))),
        true_w,
        atol=bg.fit_sigma.reshape((bg.nbounds, len(bg))) * 2,
    )
