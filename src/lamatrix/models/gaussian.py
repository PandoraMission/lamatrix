"""Generator objects for Gaussian types of models"""

import numpy as np

from ..generator import Generator
from ..math import MathMixins

__all__ = [
    "lnGaussian1DGenerator",
    "dlnGaussian1DGenerator",
    "lnGaussian2DGenerator",
    "dlnGaussian2DGenerator",
]

class lnGaussian1DGenerator(MathMixins, Generator):
    def __init__(
        self,
        x_name: str = "x",
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        stddev_prior=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None
        self.stddev_prior = stddev_prior
        if self.stddev_prior is not None:
            if not hasattr(self.stddev_prior, "__iter__"):
                raise AttributeError("Pass stddev prior as a tuple with (mu, sigma)")
            if not len(self.stddev_prior) == 2:
                raise AttributeError("Pass stddev prior as a tuple with (mu, sigma)")

            self.prior_mu[1] = -1 / (2 * self.stddev_prior[0] ** 2)
            self.prior_sigma[1] = self.mu[1] - (
                -1 / 2 * (self.stddev_prior[0] + self.stddev_prior[1]) ** 2
            )

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _INIT_ATTRS(self):
        return [
            "x_name",
            "stddev_prior",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
        ]

    def design_matrix(self, *args, **kwargs):
        """

        Parameters
        ----------
        x : np.ndarray
            Vector to create ln Gaussian of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), 2)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        return np.vstack(
            [
                np.ones(np.prod(x.shape)),
                x.ravel() ** 2,
            ]
        ).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def A(self):
        return self.mu[0], self.sigma[0]

    @property
    def stddev(self):
        stddev = np.sqrt(-(1 / (2 * self.mu[1])))
        stddev_err = -(self.sigma[1])/(2 * np.sqrt(2) * self.mu[1] ** (3/2))
        return stddev, stddev_err

    @property
    def table_properties(self):
        return [
            *[
                (
                    "w_{idx}",
                    (self.mu[idx], self.sigma[idx]),
                    (self.prior_mu[idx], self.prior_sigma[idx]),
                )
                for idx in range(self.width)
            ],
            ("A", self.A, None),
            ("\\sigma", self.stddev, self.stddev_prior),
        ]

    @property
    def _equation(self):
        return [
            f"\\mathbf{{{self.x_name}}}^0",
            f"\mathbf{{{self.x_name}}}^{2}",
        ]

    def to_latex(self):
        eq0 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}})) = -\\frac{{1}}{{2}} \\ln(2\\pi\\sigma^2) + \\frac{{\\mathbf{{{self.x_name}}}^2}}{{2 \\sigma^2}}\\end{{equation}}"
        eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}})) = w_0 + w_1\\mathbf{{{self.x_name}}}^2\\end{{equation}}"
        eq2 = "\\[ w_0 = -\\frac{{1}}{{2}} \\ln(2\\pi\\sigma^2) \\]"
        eq3 = "\\[ w_1 = \\frac{1}{2\\sigma^2}\\]"
        eq4 = "\\[\\sigma = \\sqrt{-\\frac{1}{2w_1}}\\]"
        return "\n".join(
            [eq0, eq1, eq2, eq3, eq4, self._to_latex_table()]
        )

    @property
    def gradient(self):
        return dlnGaussian1DGenerator(
            stddev=self.stddev[0],
            x_name=self.x_name,
            data_shape=self.data_shape,
        )


class dlnGaussian1DGenerator(MathMixins, Generator):
    def __init__(
        self,
        stddev: float,
        x_name: str = "x",
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.stddev = stddev
        self.x_name = x_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _INIT_ATTRS(self):
        return [
            "x_name",
            "stddev",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
        ]

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        x : np.ndarray
            Vector to create ln Gaussian of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), 2)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)

        dfdx = (x / self.stddev**2)
        return np.vstack([np.ones(np.prod(dfdx.shape)), dfdx.ravel()]).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def shift(self):
        return self.mu[1], self.sigma[1]

    @property
    def table_properties(self):
        return [
            (
                "w_0",
                (self.mu[0], self.sigma[0]),
                (self.prior_mu[0], self.prior_sigma[0]),
            ),
            ("s", self.shift, (self.prior_mu[1], self.prior_sigma[1])),
        ]

    @property
    def _equation(self):
        dfdx = f"\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma^2}}"
        return [f"\\mathbf{{{self.x_name}}}^0", dfdx]

    @property
    def _mu_letter(self):
        return "v"

class lnGaussian2DGenerator(MathMixins, Generator):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        stddev_x_prior=None,
        stddev_y_prior=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None
        self.stddev_x_prior, self.stddev_y_prior = stddev_x_prior, stddev_y_prior
        if self.stddev_x_prior is not None:
            if not hasattr(self.stddev_x_prior, "__iter__"):
                raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")
            if not len(self.stddev_x_prior) == 2:
                raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")

            self.prior_mu[1] = -1 / (2 * self.stddev_x_prior[0] ** 2)
            self.prior_sigma[1] = self.mu[1] - (
                -1 / 2 * (self.stddev_x_prior[0] + self.stddev_x_prior[1]) ** 2
            )

        if self.stddev_y_prior is not None:
            if not hasattr(self.stddev_y_prior, "__iter__"):
                raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")
            if not len(self.stddev_y_prior) == 2:
                raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")

            self.prior_mu[2] = -1 / (2 * self.stddev_y_prior[0] ** 2)
            self.prior_sigma[2] = self.mu[2] - (
                -1 / 2 * (self.stddev_y_prior[0] + self.stddev_y_prior[1]) ** 2
            )

    @property
    def width(self):
        return 4

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    @property
    def _INIT_ATTRS(self):
        return [
            "x_name",
            "y_name",
            "stddev_x_prior",
            "stddev_y_prior",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
        ]

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        x : np.ndarray
            Vector to create ln Gaussian of
        y : np.ndarray
            Vector to create ln Gaussian of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), 4)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        y = kwargs.get(self.y_name)
        return np.vstack(
            [
                np.ones(np.prod(x.shape)),
                x.ravel() ** 2,
                y.ravel() ** 2,
                x.ravel() * y.ravel(),
            ]
        ).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def A(self):
        return self.mu[0], self.sigma[0]

    @property
    def rho(self):
        rho = np.sqrt(self.mu[3] ** 2 / np.abs(self.mu[1] * self.mu[2])) / 2
        err1 = np.abs((self.mu[3] ** 2) * 2 * self.sigma[3] / self.mu[3])
        rho_err = np.abs(rho) * np.sqrt(
            (err1 / self.mu[3]) ** 2
            + (self.sigma[1] / self.mu[1]) ** 2
            + (self.sigma[2] / self.mu[2]) ** 2
        )
        return rho, rho_err

    @property
    def stddev_x(self):
        stddev_x = np.sqrt(-(1 / (2 * self.mu[1] * (1 - self.rho[0] ** 2))))
        stddev_x_2_err = np.sqrt(
            (
                (np.abs((self.rho[0]) * 2 * self.rho[1] / self.rho[0])) ** 2
                + (self.sigma[1] / self.mu[1]) ** 2
            )
        )
        stddev_x_err = np.abs(stddev_x * -0.5 * (stddev_x_2_err / stddev_x**2))
        return stddev_x, stddev_x_err

    @property
    def stddev_y(self):
        stddev_y = np.sqrt(-(1 / (2 * self.mu[2] * (1 - self.rho[0] ** 2))))
        stddev_y_2_err = np.sqrt(
            (
                (np.abs((self.rho[0]) * 2 * self.rho[1] / self.rho[0])) ** 2
                + (self.sigma[2] / self.mu[2]) ** 2
            )
        )
        stddev_y_err = np.abs(stddev_y * -0.5 * (stddev_y_2_err / stddev_y**2))
        return stddev_y, stddev_y_err

    @property
    def table_properties(self):
        return [
            *[
                (
                    "w_{idx}",
                    (self.mu[idx], self.sigma[idx]),
                    (self.prior_mu[idx], self.prior_sigma[idx]),
                )
                for idx in range(self.width)
            ],
            ("A", self.A, None),
            (f"\\sigma_{{{self.x_name}}}", self.stddev_x, self.stddev_x_prior),
            (f"\\sigma_{{{self.y_name}}}", self.stddev_y, self.stddev_y_prior),
            ("\\rho", self.rho, None),
        ]

    @property
    def _equation(self):
        return [
            f"\\mathbf{{{self.x_name}}}^0",
            f"\mathbf{{{self.x_name}}}^{2}",
            f"\mathbf{{{self.y_name}}}^{2}",
            f"\mathbf{{{self.x_name}}}\mathbf{{{self.y_name}}}",
        ]

    def to_latex(self):
        eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}}, \\mathbf{{{self.y_name}}})) = a + b\\mathbf{{{self.x_name}}}^2 + c\\mathbf{{{self.y_name}}}^2 + 2d\\mathbf{{{self.x_name}}}\\mathbf{{{self.y_name}}}\\end{{equation}}"
        eq2 = f"\\[ a = -\\ln(2\\pi\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}\\sqrt{{1-\\rho^2}}) \\]"
        eq3 = f"\\[ b = \\frac{{1}}{{2(1-\\rho^2)\\sigma_{{{self.x_name}}}^2}}\\]"
        eq4 = f"\\[ c = \\frac{{1}}{{2(1-\\rho^2)\\sigma_{{{self.y_name}}}^2}}\\]"
        eq5 = f"\\[ d = \\frac{{\\rho}}{{2(1-\\rho^2)\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\]"
        eq6 = f"\\[\\sigma_{{{self.x_name}}} = \\sqrt{{-\\frac{{1}}{{2b(1-\\rho^2)}}}}\\]"
        eq7 = f"\\[\\sigma_{{{self.y_name}}} = \\sqrt{{-\\frac{{1}}{{2c(1-\\rho^2)}}}}\\]"
        eq8 = "\\[\\rho = \\sqrt{\\frac{d^2}{bc}}\\]"
        return "\n".join(
            [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, self._to_latex_table()]
        )

    @property
    def gradient(self):
        return dlnGaussian2DGenerator(
            stddev_x=self.stddev_x[0],
            stddev_y=self.stddev_y[0],
            rho=self.rho[0],
            x_name=self.x_name,
            y_name=self.y_name,
            data_shape=self.data_shape,
        )


class dlnGaussian2DGenerator(MathMixins, Generator):
    def __init__(
        self,
        stddev_x: float,
        stddev_y: float,
        rho: float = 0,
        x_name: str = "x",
        y_name: str = "y",
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.rho = rho
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return 3

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    @property
    def _INIT_ATTRS(self):
        return [
            "x_name",
            "y_name",
            "stddev_x",
            "stddev_y",
            "prior_mu",
            "rho",
            "prior_sigma",
            "offset_prior",
            "data_shape",
        ]

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        x : np.ndarray
            Vector to create ln Gaussian of
        y : np.ndarray
            Vector to create ln Gaussian of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), 4)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        y = kwargs.get(self.y_name)

        dfdx = (1 / 1 - self.rho**2) * (
            (x / self.stddev_x**2) - self.rho * (y / (self.stddev_x * self.stddev_y))
        )
        dfdy = (1 / 1 - self.rho**2) * (
            (y / self.stddev_y**2) - self.rho * (x / (self.stddev_x * self.stddev_y))
        )
        return np.vstack([np.ones(np.prod(dfdx.shape)), dfdx.ravel(), dfdy.ravel()]).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def shift_x(self):
        return self.mu[1], self.sigma[1]

    @property
    def shift_y(self):
        return self.mu[2], self.sigma[2]

    @property
    def table_properties(self):
        return [
            (
                "w_0",
                (self.mu[0], self.sigma[0]),
                (self.prior_mu[0], self.prior_sigma[0]),
            ),
            ("s_x", self.shift_x, (self.prior_mu[1], self.prior_sigma[1])),
            ("s_y", self.shift_y, (self.prior_mu[2], self.prior_sigma[2])),
        ]

    @property
    def _equation(self):
        dfdx = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left(\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}^2}} - \\rho\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\right)\\right)"
        dfdy = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left(\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}^2}} - \\rho\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\right)\\right)"
        return [f"\\mathbf{{{self.x_name}}}^0", dfdx, dfdy]

    @property
    def _mu_letter(self):
        return "v"