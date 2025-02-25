"""Model objects for Gaussian types of models"""

from typing import List

import numpy as np

from ..distributions import Distribution, DistributionsContainer
from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Model

__all__ = [
    "Gaussian",
    "dGaussian",
    "lnGaussian",
    "dlnGaussian",
    "Gaussian2D",
    "dGaussian2D",
    "lnGaussian2D",
    "dlnGaussian2D",
]


class Gaussian(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        sigma: float = 1.0,
        mu: float = 0.0,
        priors=None,
        posteriors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.sigma = sigma
        self.mu = mu
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
            "sigma",
            "mu",
        ]

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `x_name`.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """

        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        normalization = 1 / (2 * np.pi * self.sigma**2) ** 0.5
        f = normalization * np.exp(-0.5 * (x - self.mu) ** 2 / (self.sigma**2))
        return np.asarray(f[None, :]).transpose(shape_a)

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def posterior_amplitude(self):
        return self.posteriors[0]

    @property
    def _equation(self):
        eqn = (
            f"\\frac{{1}}{{\\sqrt{{2\\pi\\sigma^2}}}} "
            f"e^{{-\\frac{{\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu^2}}{{2\\sigma^2}}}}"
        )
        return [eqn]

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        return dGaussian(
            weights=weights,
            x_name=self.x_name,
            sigma=self.sigma,
            mu=self.mu,
            priors=priors,
        )


class dGaussian(MathMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        sigma: float = 1.0,
        mu: float = 0.0,
        priors=None,
        posteriors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.sigma = sigma
        self.mu = mu
        self._weight_width = self.width
        self.weights = self._validate_weights(weights, self._weight_width)
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def _initialization_attributes(self):
        return [
            "weights",
            "x_name",
            "sigma",
            "mu",
        ]

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _mu_letter(self):
        return "v"

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `x_name`.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")

        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        normalization = 1 / (2 * np.pi * self.sigma**2) ** 0.5
        f = normalization * np.exp(-0.5 * (x - self.mu) ** 2 / (self.sigma**2))
        return np.asarray(
            (-((x - self.mu) / self.sigma**2) * f)[None, :] * self.weights
        ).transpose(shape_a)

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def posterior_amplitude(self):
        return self.posteriors[0]

    @property
    def _equation(self):
        eqn = (
            f"\\left( -w_0\\frac{{\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu^2}}{{2\\sigma^2}} \\right)"
            f"\\frac{{1}}{{\\sqrt{{2\\pi\\sigma^2}}}}"
            f"e^{{-\\frac{{\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu^2}}{{2\\sigma^2}}}}"
        )
        return [eqn]


class Gaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        priors=None,
        sigma_x=1,
        sigma_y=1,
        mu_x=0,
        mu_y=0,
        rho=0.0,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.mu_x, self.mu_y = mu_x, mu_y
        self.rho = rho
        super().__init__(priors=priors)

    @property
    def _initialization_attributes(self):
        return ["x_name", "y_name", "sigma_x", "mu_x", "sigma_y", "mu_y", "rho"]

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `x_name`.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name) - self.mu_x
        y = kwargs.get(self.y_name) - self.mu_y
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        normalization = 1 / (
            2 * np.pi * self.sigma_x * self.sigma_y * (1 - self.rho**2) ** 0.5
        )
        p = -1 / (2 * (1 - self.rho**2))
        exponent = p * (
            (x / self.sigma_x) ** 2
            + (y / self.sigma_y) ** 2
            - 2 * self.rho * (x / self.sigma_x) * (y / self.sigma_y)
        )
        return np.asarray(normalization * np.exp(exponent)[None, :]).transpose(shape_a)

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def fit_amplitude(self):
        return self.fit_distributions[0]

    @property
    def _equation(self):
        eqn = (
            f"\\frac{{1}}{{2\\pi\\sigma_x\\sigma_y\\sqrt{{1 - \\rho^2}}}} "
            f"e^{{- \\frac{{1}}{{2(1-\\rho^2)}} "
            f"\\left[\\frac{{(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)^2}}{{2\\sigma_x^2}} "
            f"+ \\frac{{(\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)^2}}{{2\\sigma_y^2}} "
            f"- \\frac{{2\\rho(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)"
            f"(\\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]}}"
        )
        return [eqn]

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                [self.posteriors.mean[0]] * 2
                if self.posteriors is not None
                else [self.priors.mean[0]] * 2
            )
        return dGaussian2D(
            weights=weights,
            x_name=self.x_name,
            y_name=self.y_name,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            mu_x=self.mu_x,
            mu_y=self.mu_y,
            rho=self.rho,
            priors=priors,
        )


class dGaussian2D(MathMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        y_name: str = "y",
        priors=None,
        sigma_x=1,
        sigma_y=1,
        mu_x=0,
        mu_y=0,
        rho=0.5,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.mu_x, self.mu_y = mu_x, mu_y
        self.rho = rho
        self._weight_width = self.width
        self.weights = self._validate_weights(weights, self._weight_width)
        super().__init__(priors=priors)

    @property
    def _initialization_attributes(self):
        return [
            "weights",
            "x_name",
            "y_name",
            "sigma_x",
            "mu_x",
            "sigma_y",
            "mu_y",
            "rho",
        ]

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    @property
    def _mu_letter(self):
        return "v"

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `x_name`.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name) - self.mu_x
        y = kwargs.get(self.y_name) - self.mu_y
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]

        normalization = 1 / (
            2 * np.pi * self.sigma_x * self.sigma_y * ((1 - self.rho**2) ** 0.5)
        )
        p = -1 / (2 * (1 - self.rho**2))
        exponent = p * (
            (x / self.sigma_x) ** 2
            + (y / self.sigma_y) ** 2
            - 2 * self.rho * (x / self.sigma_x) * (y / self.sigma_y)
        )
        f = normalization * np.exp(exponent)

        dfdx_0 = (
            2
            * p
            * ((x / self.sigma_x**2) - ((self.rho * y) / (self.sigma_x * self.sigma_y)))
        )
        dfdy_0 = (
            2
            * p
            * ((y / self.sigma_y**2) - ((self.rho * x) / (self.sigma_x * self.sigma_y)))
        )

        dfdx = self.weights[0] * f * dfdx_0
        dfdy = self.weights[1] * f * dfdy_0
        return np.asarray([dfdx, dfdy]).transpose(shape_a)

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def fit_amplitude(self):
        return self.fit_distributions[0]

    @property
    def _equation(self):
        eqn0 = (
            f"\\frac{{1}}{{2\\pi\\sigma_x\\sigma_y\\sqrt{{1 - \\rho^2}}}} "
            f"e^{{- \\frac{{1}}{{2(1-\\rho^2)}} "
            f"\\left[\\frac{{(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)^2}}{{2\\sigma_x^2}} "
            f"+ \\frac{{(\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)^2}}{{2\\sigma_y^2}} "
            f"- \\frac{{2\\rho(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)"
            f"(\\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]}}"
        )
        dfdx0 = (
            f"\\frac{{-w_0}}{{(1-\\rho^2)}} "
            f"\\left[ \\frac{{(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)}}{{\\sigma_x^2}} "
            f"- \\frac{{\\rho(\\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]"
        )
        dfdy0 = (
            f"\\frac{{-w_1}}{{(1-\\rho^2)}} "
            f"\\left[ \\frac{{(\\mathbf{{{self.latex_aliases[self.y_name]}}} - \\mu_y)}}{{\\sigma_y^2}} "
            f"- \\frac{{\\rho(\\mathbf{{{self.latex_aliases[self.x_name]}}} - \\mu_x)}}{{\\sigma_x\\sigma_y}}\\right]"
        )
        return [dfdx0 + "." + eqn0, dfdy0 + "." + eqn0]


class lnGaussian(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        # mu: float = 0,
        priors=None,
        prior_A=None,
        prior_mu=None,
        prior_sigma=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()

        # self.mu = mu
        super().__init__(priors=priors)
        if np.any([(p is not None) for p in [prior_A, prior_mu, prior_sigma]]):
            if priors is not None:
                raise ValueError(
                    "Specify either priors on sigma or priors on coefficients."
                )
            prior_sigma = self._validate_distributions(prior_sigma)[0]
            prior_A = self._validate_distributions(prior_A)[0]
            prior_mu = self._validate_distributions(prior_mu)[0]
            self.priors = self.gaussian_parameters_to_coefficients(
                DistributionsContainer([prior_A, prior_mu, prior_sigma])
            )

    def coefficients_to_gaussian_parameters(self, distributions):
        a = distributions[0]
        if isinstance(a, tuple):
            a, a_err = a
        b = distributions[1]
        if isinstance(b, tuple):
            b, b_err = b
        c = distributions[2]
        if isinstance(c, tuple):
            c, c_err = c

        if a >= 0:
            raise ValueError("Invalid input: 'a' must be negative for real sigma.")
        sigma = np.sqrt(-1 / (2 * a))
        mu = -b / (2 * a)
        A = np.exp(
            c + (1 / 2) * np.log(2 * np.pi * sigma**2) + (mu**2) / (2 * sigma**2)
        )
        if isinstance(distributions[0], (int, float)):
            return A, mu, sigma
        elif isinstance(distributions[0], tuple):
            sigma_err = (1 / (4 * sigma * a**2)) * a_err
            mu_err = np.sqrt(
                (b / (2 * a**2) * a_err) ** 2 + (-1 / (2 * a) * b_err) ** 2
            )

            dA_dc = A
            dA_dsigma = A * (1 / sigma + mu**2 / sigma**3)
            dA_dmu = A * (mu / sigma**2)

            A_err = np.sqrt(
                (dA_dc * c_err) ** 2
                + (dA_dsigma * sigma_err) ** 2
                + (dA_dmu * mu_err) ** 2
            )
            return DistributionsContainer(
                [(A, A_err), (mu, mu_err), (sigma, sigma_err)]
            )

    def gaussian_parameters_to_coefficients(self, distributions):
        A = distributions[0]
        if isinstance(A, tuple):
            A, A_err = A
        mu = distributions[1]
        if isinstance(mu, tuple):
            mu, mu_err = mu
        sigma = distributions[2]
        if isinstance(sigma, tuple):
            sigma, sigma_err = sigma
        if sigma <= 0:
            raise ValueError("Invalid input: 'sigma' must be positive.")
        a = -1 / (2 * sigma**2)
        b = mu / sigma**2
        c = (
            np.log(A)
            - (1 / 2) * np.log(2 * np.pi * sigma**2)
            - (mu**2) / (2 * sigma**2)
        )
        if isinstance(distributions[0], (int, float)):
            return a, b, c
        elif isinstance(distributions[0], tuple):
            a_err = (1 / sigma**3) * sigma_err
            b_err = np.sqrt(
                (1 / sigma**2 * mu_err) ** 2 + (-2 * mu / sigma**3 * sigma_err) ** 2
            )
            dc_dA = 1 / A
            dc_dsigma = -1 / sigma - mu**2 / sigma**3
            dc_dmu = -mu / sigma**2
            c_err = np.sqrt(
                (dc_dA * A_err) ** 2
                + (dc_dsigma * sigma_err) ** 2
                + (dc_dmu * mu_err) ** 2
            )
            return DistributionsContainer([(a, a_err), (b, b_err), (c, c_err)])

    @property
    def _initialization_attributes(self):
        return ["x_name", "mu", "sigma"]

    @property
    def width(self):
        return 3

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, **kwargs):
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
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        return np.asarray([x**2, x, x**0]).transpose(shape_a)

    #        return x[:, None] ** 2

    @property
    def _equation(self):
        return ["x^2", "x", ""]

    @property
    def sigma(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[2]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[2]
        )

    @property
    def mu(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[1]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[1]
        )

    @property
    def A(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[0]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[0]
        )

    def to_gradient(self, sigma=None, mu=None, priors=None):
        if sigma is None:
            sigma = self.sigma[0]
        if mu is None:
            mu = self.mu[0]

        return dlnGaussian(
            sigma=sigma,
            mu=mu,
            x_name=self.x_name,
            priors=priors,
        )

    def to_linear_space(self):
        return Gaussian(
            self.x_name, sigma=self.sigma[0], mu=self.mu[0], priors=[self.A]
        )


class dlnGaussian(MathMixins, Model):
    def __init__(
        self,
        mu: float,
        sigma: float,
        x_name: str = "x",
        priors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.mu = mu
        self.sigma = sigma
        super().__init__(priors=priors)

    @property
    def _initialization_attributes(self):
        return ["x_name", "mu", "sigma"]

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _mu_letter(self):
        return "v"

    def design_matrix(self, **kwargs):
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
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        a = -1 / (2 * self.sigma**2)
        b = self.mu / self.sigma**2
        return np.asarray((2 * a * x[None, :] + b)).transpose(shape_a)

    @property
    def _equation(self):
        return [
            f"\\left( \\frac{{-\\mathbf{{{self.latex_aliases[self.x_name]}}}}}{{\\sigma^2}} + \\frac{{\\mu}}{{\\sigma^2}} \\right)"
        ]


class lnGaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        # mu: float = 0,
        priors=None,
        prior_A=None,
        prior_mu_x=None,
        prior_sigma_x=None,
        prior_mu_y=None,
        prior_sigma_y=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()

        # self.mu = mu
        super().__init__(priors=priors)
        if np.any(
            [
                (p is not None)
                for p in [prior_A, prior_mu_x, prior_sigma_x, prior_mu_y, prior_sigma_y]
            ]
        ):
            if priors is not None:
                raise ValueError(
                    "Specify either priors on sigma or priors on coefficients."
                )
            prior_sigma_x = self._validate_distributions(prior_sigma_x)[0]
            prior_mu_x = self._validate_distributions(prior_mu_x)[0]
            prior_sigma_y = self._validate_distributions(prior_sigma_y)[0]
            prior_mu_y = self._validate_distributions(prior_mu_y)[0]
            prior_A = self._validate_distributions(prior_A)[0]
            self.priors = self.gaussian_parameters_to_coefficients(
                DistributionsContainer(
                    [prior_A, prior_mu_x, prior_sigma_x, prior_mu_y, prior_sigma_y]
                )
            )

    def gaussian_parameters_to_coefficients(self, distributions):
        A = distributions[0]
        if isinstance(A, tuple):
            A, A_err = A
        mu_x = distributions[1]
        if isinstance(mu_x, tuple):
            mu_x, mu_x_err = mu_x
        sigma_x = distributions[2]
        if isinstance(sigma_x, tuple):
            sigma_x, sigma_x_err = sigma_x
        mu_y = distributions[3]
        if isinstance(mu_y, tuple):
            mu_y, mu_y_err = mu_y
        sigma_y = distributions[4]
        if isinstance(sigma_y, tuple):
            sigma_y, sigma_y_err = sigma_y

        if (sigma_x <= 0) | (sigma_y <= 0):
            raise ValueError("Invalid input: 'sigma' must be positive.")
        a_x = -1 / (2 * sigma_x**2)
        b_x = mu_x / sigma_x**2
        a_y = -1 / (2 * sigma_y**2)
        b_y = mu_y / sigma_y**2
        c = (
            np.log(A)
            - np.log(2 * np.pi * sigma_x * sigma_y)
            - (mu_x**2) / (2 * sigma_x**2)
            - (mu_y**2) / (2 * sigma_y**2)
        )
        if isinstance(distributions[0], (int, float)):
            return a_x, b_x, a_y, b_y, c
        elif isinstance(distributions[0], tuple):
            a_x_err = (1 / sigma_x**3) * sigma_x_err
            b_x_err = np.sqrt(
                (1 / sigma_x**2 * mu_x_err) ** 2
                + (-2 * mu_x / sigma_x**3 * sigma_x_err) ** 2
            )
            a_y_err = (1 / sigma_y**3) * sigma_y_err
            b_y_err = np.sqrt(
                (1 / sigma_y**2 * mu_y_err) ** 2
                + (-2 * mu_y / sigma_y**3 * sigma_y_err) ** 2
            )

            dc_dA = 1 / A
            dc_dsigma_x = -1 / sigma_x - mu_x**2 / sigma_x**3
            dc_dsigma_y = -1 / sigma_y - mu_y**2 / sigma_y**3
            dc_dmu_x = -mu_x / sigma_x**2
            dc_dmu_y = -mu_y / sigma_y**2
            c_err = np.sqrt(
                (dc_dA * A_err) ** 2
                + (dc_dsigma_x * sigma_x_err) ** 2
                + (dc_dmu_x * mu_x_err) ** 2
                + (dc_dsigma_y * sigma_y_err) ** 2
                + (dc_dmu_y * mu_y_err) ** 2
            )
            return DistributionsContainer(
                [
                    (a_x, a_x_err),
                    (b_x, b_x_err),
                    (a_y, a_y_err),
                    (b_y, b_y_err),
                    (c, c_err),
                ]
            )

    def coefficients_to_gaussian_parameters(self, distributions):
        a_x = distributions[0]
        if isinstance(a_x, tuple):
            a_x, a_x_err = a_x
        b_x = distributions[1]
        if isinstance(b_x, tuple):
            b_x, b_x_err = b_x
        a_y = distributions[2]
        if isinstance(a_y, tuple):
            a_y, a_y_err = a_y
        b_y = distributions[3]
        if isinstance(b_y, tuple):
            b_y, b_y_err = b_y
        c = distributions[4]
        if isinstance(c, tuple):
            c, c_err = c

        if (a_x >= 0) | (a_y >= 0):
            raise ValueError("Invalid input: 'a' must be positive.")

        sigma_x = np.sqrt(-1 / (2 * a_x))
        mu_x = -b_x / (2 * a_x)
        sigma_y = np.sqrt(-1 / (2 * a_y))
        mu_y = -b_y / (2 * a_y)
        A = np.exp(
            c
            + np.log(2 * np.pi * sigma_x * sigma_y)
            + (mu_x**2) / (2 * sigma_x**2)
            + (mu_y**2) / (2 * sigma_y**2)
        )

        if isinstance(distributions[0], (int, float)):
            return A, mu_x, sigma_x, mu_y, sigma_y

        elif isinstance(distributions[0], tuple):
            sigma_x_err = (1 / (4 * sigma_x * a_x**2)) * a_x_err
            mu_x_err = np.sqrt(
                (b_x / (2 * a_x**2) * a_x_err) ** 2 + (-1 / (2 * a_x) * b_x_err) ** 2
            )

            sigma_y_err = (1 / (4 * sigma_y * a_y**2)) * a_y_err
            mu_y_err = np.sqrt(
                (b_y / (2 * a_y**2) * a_y_err) ** 2 + (-1 / (2 * a_y) * b_y_err) ** 2
            )

            dA_dc = A
            dA_dsigma_x = A * (1 / sigma_x + mu_x**2 / sigma_x**3)
            dA_dmu_x = A * (mu_x / sigma_x**2)
            dA_dsigma_y = A * (1 / sigma_y + mu_y**2 / sigma_y**3)
            dA_dmu_y = A * (mu_y / sigma_y**2)

            A_err = np.sqrt(
                (dA_dc * c_err) ** 2
                + (dA_dsigma_x * sigma_x_err) ** 2
                + (dA_dmu_x * mu_x_err) ** 2
                + (dA_dsigma_y * sigma_y_err) ** 2
                + (dA_dmu_y * mu_y_err) ** 2
            )
            return DistributionsContainer(
                [
                    (A, A_err),
                    (mu_x, mu_x_err),
                    (sigma_x, sigma_x_err),
                    (mu_y, mu_y_err),
                    (sigma_y, sigma_y_err),
                ]
            )

    @property
    def _initialization_attributes(self):
        return ["x_name", "y_name"]

    @property
    def width(self):
        return 5

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    def design_matrix(self, **kwargs):
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
        y = kwargs.get(self.y_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        return np.asarray([x**2, x, y**2, y, x**0]).transpose(shape_a)

    #        return x[:, None] ** 2

    @property
    def _equation(self):
        return ["x^2", "x", "y^2", "y", ""]

    @property
    def sigma_x(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[2]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[2]
        )

    @property
    def mu_x(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[1]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[1]
        )

    @property
    def sigma_y(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[4]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[4]
        )

    @property
    def mu_y(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[3]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[3]
        )

    @property
    def A(self):
        return Distribution(
            self.coefficients_to_gaussian_parameters(self.posteriors)[0]
            if self.posteriors is not None
            else self.coefficients_to_gaussian_parameters(self.priors)[0]
        )

    def to_gradient(
        self, sigma_x=None, mu_x=None, sigma_y=None, mu_y=None, priors=None
    ):
        if sigma_x is None:
            sigma_x = self.sigma_x[0]
        if mu_x is None:
            mu_x = self.mu_x[0]
        if sigma_y is None:
            sigma_y = self.sigma_y[0]
        if mu_y is None:
            mu_y = self.mu_y[0]

        return dlnGaussian2D(
            sigma_x=sigma_x,
            mu_x=mu_x,
            sigma_y=sigma_y,
            mu_y=mu_y,
            x_name=self.x_name,
            priors=priors,
        )

    def to_linear_space(self):
        return Gaussian2D(
            self.x_name,
            self.y_name,
            sigma_x=self.sigma_x[0],
            mu_x=self.mu_x[0],
            sigma_y=self.sigma_y[0],
            mu_y=self.mu_y[0],
            priors=self.A,
        )


class dlnGaussian2D(MathMixins, Model):
    def __init__(
        self,
        mu_x: float,
        sigma_x: float,
        mu_y: float,
        sigma_y: float,
        x_name: str = "x",
        y_name: str = "y",
        priors=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        super().__init__(priors=priors)

    @property
    def _initialization_attributes(self):
        return ["x_name", "y_name", "sigma_x", "mu_x", "sigma_y", "mu_y"]

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name, self.y_name}

    @property
    def _mu_letter(self):
        return "v"

    def design_matrix(self, **kwargs):
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
        y = kwargs.get(self.y_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        a_x = -1 / (2 * self.sigma_x**2)
        b_x = self.mu_x / self.sigma_x**2

        a_y = -1 / (2 * self.sigma_y**2)
        b_y = self.mu_y / self.sigma_y**2
        return np.asarray([(2 * a_x * x + b_x), (2 * a_y * y + b_y)]).transpose(shape_a)

    @property
    def _equation(self):
        return [
            f"\\left( \\frac{{-\\mathbf{{{self.latex_aliases[self.x_name]}}}}}{{\\sigma_x^2}} + \\frac{{\\mu_x}}{{\\sigma_x^2}} \\right)",
            f"\\left( \\frac{{-\\mathbf{{{self.latex_aliases[self.y_name]}}}}}{{\\sigma_y^2}} + \\frac{{\\mu_y}}{{\\sigma_y^2}} \\right)",
        ]
