"""Makes objects to generate matrices"""

import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
import math


__all__ = [
    "Polynomial1DGenerator",
    "Polynomial2DGenerator",
    "lnGaussian2DGenerator",
    "dlnGaussian2DGenerator",
    "SinusoidGenerator",
    "CombinedGenerator",
]


class Generator(ABC):
    def _validate_arg_names(self):
        for arg in self.arg_names:
            if not isinstance(arg, str):
                raise ValueError("Argument names must be strings.")

    def _validate_priors(self, prior_mu, prior_sigma):
        if prior_mu is None:
            self.prior_mu = np.zeros(self.width)
        else:
            if isinstance(prior_mu, (float, int)):
                self.prior_mu = np.ones(self.width) * prior_mu
            elif isinstance(prior_mu, (list, np.ndarray, tuple)):
                if len(prior_mu) == self.width:
                    self.prior_mu = prior_mu
            else:
                raise ValueError("Can not parse `prior_mu`.")

        if prior_sigma is None:
            self.prior_sigma = np.ones(self.width) * np.inf
        else:
            if isinstance(prior_sigma, (float, int)):
                self.prior_sigma = np.ones(self.width) * prior_sigma
            elif isinstance(prior_sigma, (list, np.ndarray, tuple)):
                if len(prior_sigma) == self.width:
                    self.prior_sigma = prior_sigma
            else:
                raise ValueError("Can not parse `prior_sigma`.")

    # def update_priors(self):
    #     if self.fit_mu is None:
    #         raise ValueError("Can not update priors before fitting.")
    #     new = self.copy()
    #     new.prior_mu = new.fit_mu.copy()
    #     new.prior_sigma = new.fit_sigma.copy()
    #     return new

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str):
        raise NotImplementedError

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return f"{type(self).__name__}[n, {self.width}]"

    def __add__(self, other):
        if isinstance(other, Generator):
            return CombinedGenerator(self, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")

    @staticmethod
    def format_significant_figures(mean, error):
        # Check for inf, -inf, or NaN
        if (
            math.isinf(mean)
            or math.isinf(error)
            or math.isnan(mean)
            or math.isnan(error)
        ):
            # Handle these cases as you see fit, for example:
            return "\\varnothing", "\\varnothing"

        # Find the first significant digit of the error
        if error == 0:
            sig_figures = 0
        else:
            sig_figures = -int(math.floor(math.log10(abs(error))))

        # Format mean and error to have the same number of decimal places
        formatted_mean = f"{mean:.{sig_figures}f}"
        formatted_error = f"{error:.{sig_figures}f}"

        return formatted_mean, formatted_error

    def _get_table_matter(self):
        table_matter = ""
        for symbol, (mean, error) in self.table_properties:
            formatted_mean, formatted_error = self.format_significant_figures(
                mean, error
            )
            row = f"{symbol} & ${formatted_mean} \\pm {formatted_error}$ \\\\\\hline\n"
            table_matter += row
        return table_matter

    def to_latex(self):
        latex_table = "\\begin{table}[h!]\n\\centering\n"
        latex_table += "\\begin{tabular}{|c|c|}\n\\hline\n"
        latex_table += "Coefficient & Value \\\\\\hline\n"
        latex_table += self._get_table_matter()
        latex_table += "\\end{tabular}\n\\end{table}"
        return latex_table

    def _fit(self, data, errors=None, mask=None, *args, **kwargs):
        X = self.design_matrix(*args, **kwargs)
        if np.prod(data.shape) != X.shape[0]:
            raise ValueError(f"Data must have shape {X.shape[0]}")
        if errors is None:
            errors = np.ones_like(data)
        if mask is None:
            mask = np.ones(np.prod(data.shape), bool)
        self.data_shape = data.shape
        mask = mask.ravel()

        sigma_w_inv = X[mask].T.dot(
            X[mask] / errors.ravel()[mask, None] ** 2
        ) + np.diag(1 / self.prior_sigma**2)
        self.cov = np.linalg.inv(sigma_w_inv)
        B = (
            X[mask].T.dot(data.ravel()[mask] / errors.ravel()[mask] ** 2)
            + self.prior_mu / self.prior_sigma**2
        )
        fit_mu = np.linalg.solve(sigma_w_inv, B)
        fit_sigma = self.cov.diagonal() ** 0.5
        return fit_mu, fit_sigma

    @property
    def mu(self):
        return self.prior_mu if self.fit_mu is None else self.fit_mu

    @property
    def sigma(self):
        return self.prior_sigma if self.fit_sigma is None else self.fit_sigma

    def evaluate(self, *args, **kwargs):
        X = self.design_matrix(*args, **kwargs)
        if self.data_shape is not None:
            if np.prod(X.shape) == np.prod(self.data_shape):
                return X.dot(self.mu).reshape(self.data_shape)
        return X.dot(self.mu)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def sample(self, size=None, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def table_properties(self):
        """Returns a list of tuples, which contain the latex string for the parameter, and the value of the parameter in the format (mean, sigma)."""
        pass

    @property
    @abstractmethod
    def arg_names(self):
        """Returns a set of the user defined strings for all the arguments that the design matrix requires."""
        pass

    @property
    @abstractmethod
    def equation(self):
        """Returns a list of latex equations to describe the generation"""
        pass

    @abstractmethod
    def design_matrix(self):
        """Returns a design matrix, given inputs listed in self.arg_names."""
        pass

    @property
    @abstractmethod
    def nvectors(self):
        """Returns number of unique vectors required to build the design matrix."""
        pass

    @property
    @abstractmethod
    def width(self):
        """Returns the width of the design matrix once built."""
        pass


class Polynomial1DGenerator(Generator):
    def __init__(
        self,
        x_name: str = "x",
        polyorder: int = 3,
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.polyorder = polyorder
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma)
        self.fit_mu = None
        self.fit_sigma = None

        if offset_prior is not None:
            if not hasattr(offset_prior, "__iter__"):
                raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")
            if not len(offset_prior) == 2:
                raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")

            self.prior_mu[0] = offset_prior[0]
            self.prior_sigma[0] = offset_prior[1]

    @property
    def width(self):
        return self.polyorder + 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create polynomial of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name).ravel()
        return np.vstack([x**idx for idx in range(self.polyorder + 1)]).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def offset(self):
        return self.mu[0], self.sigma[0]

    @property
    def table_properties(self):
        return [
            ("w_$idx$", (self.mu[idx], self.sigma[idx])) for idx in range(self.width)
        ]

    @property
    def equation(self):
        return (
            "$f(x) = "
            + " + ".join(
                [
                    f"w_{coeff} {var}"
                    for coeff, var in enumerate(
                        [f"x^{idx}" for idx in range(self.polyorder + 1)]
                    )
                ]
            )
            + "$"
        )


class Polynomial2DGenerator(Generator):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        polyorder: int = 3,
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.polyorder = polyorder
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma)
        self.fit_mu = None
        self.fit_sigma = None

        if offset_prior is not None:
            if not hasattr(offset_prior, "__iter__"):
                raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")
            if not len(offset_prior) == 2:
                raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")

            self.prior_mu[0] = offset_prior[0]
            self.prior_sigma[0] = offset_prior[1]

    @property
    def width(self):
        return (self.polyorder + 1) ** 2

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    def design_matrix(self, *args, **kwargs):
        """Build a 2D polynomial in x and y

        Parameters
        ----------
        {} : np.ndarray
            Vector to create polynomial of
        {} : np.ndarray
            Vector to create polynomial of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name).ravel()
        y = kwargs.get(self.y_name).ravel()
        X = np.vstack([x**idx for idx in range(self.polyorder + 1)]).T
        Y = np.vstack([y**idx for idx in range(self.polyorder + 1)]).T
        return np.hstack([X * y[:, None] for y in Y.T])

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def offset(self):
        return self.mu[0], self.sigma[0]

    @property
    def table_properties(self):
        return [
            ("w_$idx$", (self.mu[idx], self.sigma[idx])) for idx in range(self.width)
        ]

    @property
    def equation(self):
        return (
            "$f(x, y) = "
            + " + ".join(
                [
                    f"w_{coeff} {var}"
                    for coeff, var in enumerate(
                        [
                            f"x^{idx} y^{jdx}"
                            for jdx in range(self.polyorder + 1)
                            for idx in range(self.polyorder + 1)
                        ]
                    )
                ]
            )
            + "$"
        )


class lnGaussian2DGenerator(Generator):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        prior_mu=None,
        prior_sigma=None,
        stddev_x_prior=None,
        stddev_y_prior=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma)
        self.fit_mu = None
        self.fit_sigma = None
        if stddev_x_prior is not None:
            if not hasattr(stddev_x_prior, "__iter__"):
                raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")
            if not len(stddev_x_prior) == 2:
                raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")

            self.prior_mu[1] = -1 / (2 * stddev_x_prior[0] ** 2)
            self.prior_sigma[1] = self.mu[1] - (
                -1 / 2 * (stddev_x_prior[0] + stddev_x_prior[1]) ** 2
            )

        if stddev_y_prior is not None:
            if not hasattr(stddev_y_prior, "__iter__"):
                raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")
            if not len(stddev_y_prior) == 2:
                raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")

            self.prior_mu[2] = -1 / (2 * stddev_y_prior[0] ** 2)
            self.prior_sigma[2] = self.mu[2] - (
                -1 / 2 * (stddev_y_prior[0] + stddev_y_prior[1]) ** 2
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
        rho = np.sqrt(self.mu[3] ** 2 / (self.mu[1] * self.mu[2])) / 2
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
            ("A", self.A),
            ("\\sigma_x", self.stddev_x),
            ("\\sigma_y", self.stddev_y),
            ("\\rho", self.rho),
        ]

    @property
    def equation(self):
        eq1 = "$\\ln(G(x, y)) = a + bx^2 + cy^2 + 2dxy$"
        eq2 = "$ a = -\\ln(2\\pi\\sigma_x\\sigma_y\\sqrt{1-\\rho^2}) $"
        eq3 = "$ b = \\frac{1}{2(1-\\rho^2)\\sigma_x^2}$"
        eq4 = "$ c = \\frac{1}{2(1-\\rho^2)\\sigma_y^2}$"
        eq5 = "$ d = \\frac{\\rho}{2(1-\\rho^2)\\sigma_x\\sigma_y}$"
        eq6 = "$\\sigma_x = \\sqrt{-\\frac{1}{2b(1-\\rho^2)}}$"
        eq7 = "$\\sigma_y = \\sqrt{-\\frac{1}{2c(1-\\rho^2)}}$"
        eq8 = "$\\rho = \\sqrt{\\frac{d^2}{bc}}$"
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]


class dlnGaussian2DGenerator(Generator):
    def __init__(
        self,
        stddev_x: float,
        stddev_y: float,
        rho:float=0,
        x_name: str = "x",
        y_name: str = "y",
        prior_mu=None,
        prior_sigma=None,
        data_shape=None,
    ):
        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.rho = rho
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

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
        return np.vstack([dfdx.ravel(), dfdy.ravel()]).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def shift_x(self):
        return self.mu[0], self.sigma[0]

    @property
    def shift_y(self):
        return self.mu[1], self.sigma[1]

    @property
    def table_properties(self):
        return [
            ("s_x", self.shift_x),
            ("s_y", self.shift_y),
        ]

    @property
    def equation(self):
        dgdx = "$\\frac{\\partial}{\\partial x} \\ln(G(x, y)) = -\\frac{1}{1-\\rho^2}\\left(\\frac{(x-\\mu_x)}{\\sigma_x^2} - \\rho\\frac{(y-\\mu_y)}{\\sigma_x\\sigma_y}\\right)$"
        dgdx = "$\\frac{\\partial}{\\partial y} \\ln(G(x, y)) = -\\frac{1}{1-\\rho^2}\\left(\\frac{(y-\\mu_y)}{\\sigma_y^2} - \\rho\\frac{(x-\\mu_x)}{\\sigma_x\\sigma_y}\\right)$"

        return [
            dgdx,
            dgdx,
            "$f(x, y) = s_x\\frac{\\partial}{\\partial x} \\ln(G(x, y)) + s_y\\frac{\\partial}{\\partial y} \\ln(G(x, y))$",
        ]


class SinusoidGenerator(Generator):
    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        prior_mu=None,
        prior_sigma=None,
        data_shape=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        self.width = (self.nterms * 2) + 1
        self._validate_priors(prior_mu, prior_sigma)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return (self.nterms * 2) + 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, *args, **kwargs):
        """Build a 1D polynomial in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create polynomial of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name).ravel()
        return np.vstack(
            [
                x**0,
                *[
                    [np.sin(x * (idx + 1)), np.cos(x * (idx + 1))]
                    for idx in range(self.nterms)
                ],
            ]
        ).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def table_properties(self):
        return [
            ("w_$idx$", (self.mu[idx], self.sigma[idx])) for idx in range(self.width)
        ]


class CombinedGenerator(Generator):
    def __init__(self, *args, **kwargs):
        self.generators = [a.copy() for a in args]

    def __getitem__(self, key):
        return self.generators[key]

    def __add__(self, other):
        if isinstance(other, Generator):
            return CombinedGenerator(*self.generators, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")

    def design_matrix(self, *args, **kwargs):
        return np.hstack([g.design_matrix(*args, **kwargs) for g in self])

    @property
    def width(self):
        return np.sum([g.width for g in self.generators])

    @property
    def nvectors(self):
        return len(np.unique(np.hstack([list(g.arg_names) for g in generators])))

    @property
    def prior_mu(self):
        return np.hstack([g.prior_mu for g in self])

    @property
    def prior_sigma(self):
        return np.hstack([g.prior_sigma for g in self])

    @property
    def mu(self):
        return np.hstack([g.mu for g in self])

    @property
    def sigma(self):
        return np.hstack([g.sigma for g in self])

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)
        lengths = [g.width for g in self]
        mu, sigma = (
            np.array_split(self.fit_mu, np.cumsum(lengths))[:-1],
            np.array_split(self.fit_sigma, np.cumsum(lengths))[:-1],
        )
        for idx, mu0, sigma0 in zip(np.arange(len(mu)), mu, sigma):
            self[idx].fit_mu = mu0
            self[idx].fit_sigma = sigma0

    def __len__(self):
        return len(self.generators)

    def to_latex(self):
        latex_table = "\\begin{table}[h!]\n\\centering\n"
        latex_table += "\\begin{tabular}{|c|c|}\n\\hline\n"
        latex_table += "Coefficient & Value \\\\\\hline\n"
        for g in self:
            latex_table += g._get_table_matter()
        latex_table += "\\end{tabular}\n\\end{table}"
        return latex_table

    @property
    def equation(self):
        raise NotImplementedError
    
    @property
    def table_properties(self):
        raise NotImplementedError

    @property
    def arg_names(self):
        raise NotImplementedError