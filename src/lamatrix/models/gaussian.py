"""Generator objects for Gaussian types of models"""

import numpy as np
from typing import List

from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Model
from ..distributions import DistributionsContainer, Distribution

__all__ = [
    "Gaussian",
    "dGaussian",
    "lnGaussian",
    "dlnGaussian",
    "Gaussian2D",
    "dGaussian2D",
    "lnGaussian2D",
    # "dlnGaussian2D",
]


# def _coeff_to_sigma1d(distributions):
#     a, aerr = distributions[0]
#     sigma = (-1 / (2 * a)) ** 0.5
#     sigma_err = (1 / (2 * np.sqrt(2))) * (1 / ((-a) ** (3 / 2))) * aerr
#     return (sigma, sigma_err)


# def _sigma1d_to_coeff(distribution):
#     sigma, sigma_err = distribution
#     coeff = -1 / (2 * sigma**2)
#     coeff_err = (1 / sigma**3) * sigma_err
#     return (coeff, coeff_err)


# def _coeff_to_mu1d(distributions):
#     a, aerr = distributions[0]
#     b, berr = distributions[1]
#     mu = -(b / (2 * a))
#     mu_err = mu * ((berr / b) ** 2 + (aerr / a) ** 2) ** 0.5
#     return (mu, mu_err)


# def _coeff_to_A1d(distributions):
#     sigma, sigma_err = _coeff_to_sigma1d(distributions)
#     mu, mu_err = _coeff_to_mu1d(distributions)
#     c, cerr = distributions[2]
#     A = np.exp(c + mu**2 / 2 * sigma**2 + 0.5 * (np.log(2 * np.pi * sigma**2)))
#     Aerr = A * np.sqrt(
#         cerr**2
#         + (mu / sigma**2 * mu_err) ** 2
#         + ((1 / (2 * sigma) - mu**2 / (2 * sigma**3)) * sigma_err) ** 2
#     )
#     return (A, Aerr)


def _coeff_to_sigma2d(distribution, rho_distribution):
    rho, rho_err = rho_distribution
    a, aerr = distribution
    sigma = (-1 / (2 * (1 - rho**2) * a)) ** 0.5
    sigma_err = (1 / (2 * np.sqrt(2))) * (1 / ((-a) ** (3 / 2))) * aerr
    return (sigma, sigma_err)


def _coeffs_to_rho(distributions):
    mean, std = np.asarray(distributions).T
    a, b, c = mean
    aerr, berr, cerr = std
    partial_a = 0.25 * c / (a**2 * np.sqrt(np.abs(b)))
    partial_b = 0.25 * c / (np.sqrt(np.abs(a)) * b**2)
    partial_c = 0.5 / np.sqrt(np.abs(a * b))

    # Error on rho
    rho = 0.5 * (c**2 / (a * b)) ** 0.5
    rho_err = np.sqrt(
        (partial_a * aerr) ** 2 + (partial_b * berr) ** 2 + (partial_c * cerr) ** 2
    )
    return (rho, rho_err)


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
        eqn = f"\\frac{{1}}{{\\sqrt{{2\\pi\\sigma^2}}}} e^{{-\\frac{{\mathbf{{{self.x_name}}} - \\mu^2}}{{2\\sigma^2}}}}"
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
        eqn = f"\\left( -w_0\\frac{{\mathbf{{{self.x_name}}} - \\mu^2}}{{2\\sigma^2}} \\right)\\frac{{1}}{{\\sqrt{{2\\pi\\sigma^2}}}} e^{{-\\frac{{\mathbf{{{self.x_name}}} - \\mu^2}}{{2\\sigma^2}}}}"
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
        eqn = f"\\frac{{1}}{{2\\pi\\sigma_x\\sigma_y\\sqrt{{1 - \\rho^2}}}} e^{{- \\frac{{1}}{{2(1-\\rho^2)}} \\left[\\frac{{(\\mathbf{{{self.x_name}}} - \\mu_x)^2}}{{2\\sigma_x^2}} + \\frac{{(\mathbf{{{self.y_name}}} - \\mu_y)^2}}{{2\\sigma_y^2}} - \\frac{{2\\rho(\\mathbf{{{self.x_name}}} - \\mu_x)(\\mathbf{{{self.y_name}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]}}"
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
        eqn0 = f"\\frac{{1}}{{2\\pi\\sigma_x\\sigma_y\\sqrt{{1 - \\rho^2}}}} e^{{- \\frac{{1}}{{2(1-\\rho^2)}} \\left[\\frac{{(\\mathbf{{{self.x_name}}} - \\mu_x)^2}}{{2\\sigma_x^2}} + \\frac{{(\mathbf{{{self.y_name}}} - \\mu_y)^2}}{{2\\sigma_y^2}} - \\frac{{2\\rho(\\mathbf{{{self.x_name}}} - \\mu_x)(\\mathbf{{{self.y_name}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]}}"
        dfdx0 = f"\\frac{{-w_0}}{{(1-\\rho^2)}} \\left[ \\frac{{(\\mathbf{{{self.x_name}}} - \\mu_x)}}{{\\sigma_x^2}} - \\frac{{\\rho(\\mathbf{{{self.y_name}}} - \\mu_y)}}{{\\sigma_x\\sigma_y}}\\right]"
        dfdy0 = f"\\frac{{-w_1}}{{(1-\\rho^2)}} \\left[ \\frac{{(\\mathbf{{{self.y_name}}} - \\mu_y)}}{{\\sigma_y^2}} - \\frac{{\\rho(\\mathbf{{{self.x_name}}} - \\mu_x)}}{{\\sigma_x\\sigma_y}}\\right]"
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
            prior_sigma = self._validate_priors(prior_sigma)[0]
            prior_A = self._validate_priors(prior_A)[0]
            prior_mu = self._validate_priors(prior_mu)[0]
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
            f"\\left( \\frac{{-\\mathbf{{{self.x_name}}}}}{{\\sigma^2}} + \\frac{{\\mu}}{{\\sigma^2}} \\right)"
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
            prior_sigma_x = self._validate_priors(prior_sigma_x)[0]
            prior_mu_x = self._validate_priors(prior_mu_x)[0]
            prior_sigma_y = self._validate_priors(prior_sigma_y)[0]
            prior_mu_y = self._validate_priors(prior_mu_y)[0]
            prior_A = self._validate_priors(prior_A)[0]
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
            f"\\left( \\frac{{-\\mathbf{{{self.x_name}}}}}{{\\sigma_x^2}} + \\frac{{\\mu_x}}{{\\sigma_x^2}} \\right)",
            f"\\left( \\frac{{-\\mathbf{{{self.y_name}}}}}{{\\sigma_y^2}} + \\frac{{\\mu_y}}{{\\sigma_y^2}} \\right)",
        ]


# class lnGaussian2D(MathMixins, Model):
#     def __init__(
#         self,
#         x_name: str = "x",
#         y_name: str = "y",
#         mu_x: float = 0,
#         mu_y: float = 0,
#         priors=None,
#         prior_sigma_x_distribution=None,
#         prior_sigma_y_distribution=None,
#     ):
#         self.x_name = x_name
#         self.y_name = y_name
#         self._validate_arg_names()
#         self.mu_x, self.mu_y = mu_x, mu_y
#         super().__init__(priors=priors)

#         self.prior_sigma_x_distribution, self.prior_sigma_y_distribution = (
#             prior_sigma_x_distribution,
#             prior_sigma_y_distribution,
#         )
#         if self.prior_sigma_x_distribution is not None:
#             if not hasattr(self.prior_sigma_x_distribution, "__iter__"):
#                 raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
#             if not len(self.prior_sigma_x_distribution) == 2:
#                 raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
#             self.priors[0] = _sigma1d_to_coeff(self.prior_sigma_x_distribution)

#         if self.prior_sigma_y_distribution is not None:
#             if not hasattr(self.prior_sigma_y_distribution, "__iter__"):
#                 raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
#             if not len(self.prior_sigma_y_distribution) == 2:
#                 raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
#             self.priors[1] = _sigma1d_to_coeff(self.prior_sigma_y_distribution)

#     @property
#     def width(self):
#         return 3

#     @property
#     def nvectors(self):
#         return 2

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     def design_matrix(self, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of
#         y : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 4)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name) - self.mu_x
#         y = kwargs.get(self.y_name) - self.mu_y
#         return np.vstack(
#             [
#                 x.ravel() ** 2,
#                 y.ravel() ** 2,
#                 x.ravel() * y.ravel(),
#             ]
#         ).T

#     @property
#     def prior_sigma_x(self):
#         return _coeff_to_sigma2d(self.priors[0], self.prior_rho)

#     @property
#     def fit_sigma_x(self):
#         return _coeff_to_sigma2d(self.fit_distributions[0], self.fit_rho)

#     @property
#     def prior_sigma_y(self):
#         return _coeff_to_sigma2d(self.priors[1], self.prior_rho)

#     @property
#     def fit_sigma_y(self):
#         return _coeff_to_sigma2d(self.fit_distributions[1], self.fit_rho)

#     @property
#     def prior_rho(self):
#         return _coeffs_to_rho(self.priors)

#     @property
#     def fit_rho(self):
#         return _coeffs_to_rho(self.fit_distributions)

#     def to_gradient(self, priors=None):
#         return dlnGaussian2D(
#             x_name=self.x_name,
#             y_name=self.y_name,
#             sigma_x=self.fit_sigma_x[0],
#             sigma_y=self.fit_sigma_y[0],
#             mu_x=self.mu_x,
#             mu_y=self.mu_y,
#             rho=self.fit_rho[0],
#             priors=priors,
#         )

#     @property
#     def _equation(self):
#         return [""]


# class dlnGaussian2D(MathMixins, Model):
#     def __init__(
#         self,
#         x_name: str = "x",
#         y_name: str = "y",
#         mu_x: float = 0,
#         mu_y: float = 0,
#         sigma_x: float = 0,
#         sigma_y: float = 0,
#         rho: float = 0,
#         priors=None,
#     ):
#         self.x_name = x_name
#         self.y_name = y_name
#         self._validate_arg_names()
#         self.mu_x, self.mu_y = mu_x, mu_y
#         self.sigma_x, self.sigma_y = sigma_x, sigma_y
#         self.rho = rho
#         super().__init__(priors=priors)

#     @property
#     def width(self):
#         return 2

#     @property
#     def nvectors(self):
#         return 2

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     def design_matrix(self, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of
#         y : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 4)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)
#         y = kwargs.get(self.y_name)
#         dfdx = (
#             -1
#             / (1 - self.rho**2)
#             * (
#                 (x - self.mu_x) / self.sigma_x**2
#                 - (self.rho * (y - self.mu_y) / (self.sigma_x * self.sigma_y))
#             )
#         )
#         dfdy = (
#             -1
#             / (1 - self.rho**2)
#             * (
#                 (y - self.mu_y) / self.sigma_y**2
#                 - (self.rho * (x - self.mu_x) / (self.sigma_x * self.sigma_y))
#             )
#         )
#         return np.vstack([dfdx, dfdy]).T


# class lnGaussian1D(MathMixins, Model):
#     def __init__(
#         self,
#         x_name: str = "x",
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         stddev_prior=None,
#         data_shape=None,
#     ):
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None
#         self.stddev_prior = stddev_prior
#         if self.stddev_prior is not None:
#             if not hasattr(self.stddev_prior, "__iter__"):
#                 raise AttributeError("Pass stddev prior as a tuple with (mu, sigma)")
#             if not len(self.stddev_prior) == 2:
#                 raise AttributeError("Pass stddev prior as a tuple with (mu, sigma)")

#             self.prior_mu[1] = -1 / (2 * self.stddev_prior[0] ** 2)
#             self.prior_sigma[1] = self.mu[1] - (
#                 -1 / 2 * (self.stddev_prior[0] + self.stddev_prior[1]) ** 2
#             )

#     @property
#     def width(self):
#         return 2

#     @property
#     def nvectors(self):
#         return 1

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "x_name",
#             "stddev_prior",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 2)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)
#         return np.vstack(
#             [
#                 np.ones(np.prod(x.shape)),
#                 x.ravel() ** 2,
#             ]
#         ).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def A(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def stddev(self):
#         stddev = np.sqrt(-(1 / (2 * self.mu[1])))
#         stddev_err = -(self.sigma[1])/(2 * np.sqrt(2) * self.mu[1] ** (3/2))
#         return stddev, stddev_err

#     @property
#     def table_properties(self):
#         return [
#             *[
#                 (
#                     "w_{idx}",
#                     (self.mu[idx], self.sigma[idx]),
#                     (self.prior_mu[idx], self.prior_sigma[idx]),
#                 )
#                 for idx in range(self.width)
#             ],
#             ("A", self.A, None),
#             ("\\sigma", self.stddev, self.stddev_prior),
#         ]

#     @property
#     def _equation(self):
#         return [
#             f"\\mathbf{{{self.x_name}}}^0",
#             f"\mathbf{{{self.x_name}}}^{2}",
#         ]

#     def to_latex(self):
#         eq0 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}}))
#            = -\\frac{{1}}{{2}} \\ln(2\\pi\\sigma^2) + \\frac{{\\mathbf{{{self.x_name}}}^2}}
#               {{2 \\sigma^2}}\\end{{equation}}"
#         eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}}))
#              = w_0 + w_1\\mathbf{{{self.x_name}}}^2\\end{{equation}}"
#         eq2 = "\\[ w_0 = -\\frac{{1}}{{2}} \\ln(2\\pi\\sigma^2) \\]"
#         eq3 = "\\[ w_1 = \\frac{1}{2\\sigma^2}\\]"
#         eq4 = "\\[\\sigma = \\sqrt{-\\frac{1}{2w_1}}\\]"
#         return "\n".join(
#             [eq0, eq1, eq2, eq3, eq4, self._to_latex_table()]
#         )

#     @property
#     def gradient(self):
#         return dlnGaussian1DGenerator(
#             stddev=self.stddev[0],
#             x_name=self.x_name,
#             data_shape=self.data_shape,
#         )


# class dlnGaussian1DGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         stddev: float,
#         x_name: str = "x",
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.stddev = stddev
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return 2

#     @property
#     def nvectors(self):
#         return 1

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "x_name",
#             "stddev",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 2)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)

#         dfdx = (x / self.stddev**2)
#         return np.vstack([np.ones(np.prod(dfdx.shape)), dfdx.ravel()]).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def shift(self):
#         return self.mu[1], self.sigma[1]

#     @property
#     def table_properties(self):
#         return [
#             (
#                 "w_0",
#                 (self.mu[0], self.sigma[0]),
#                 (self.prior_mu[0], self.prior_sigma[0]),
#             ),
#             ("s", self.shift, (self.prior_mu[1], self.prior_sigma[1])),
#         ]

#     @property
#     def _equation(self):
#         dfdx = f"\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma^2}}"
#         return [f"\\mathbf{{{self.x_name}}}^0", dfdx]

#     @property
#     def _mu_letter(self):
#         return "v"

# class lnGaussian2DGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         x_name: str = "x",
#         y_name: str = "y",
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         stddev_x_prior=None,
#         stddev_y_prior=None,
#         data_shape=None,
#     ):
#         self.x_name = x_name
#         self.y_name = y_name
#         self._validate_arg_names()
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None
#         self.stddev_x_prior, self.stddev_y_prior = stddev_x_prior, stddev_y_prior
#         if self.stddev_x_prior is not None:
#             if not hasattr(self.stddev_x_prior, "__iter__"):
#                 raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")
#             if not len(self.stddev_x_prior) == 2:
#                 raise AttributeError("Pass stddev_x prior as a tuple with (mu, sigma)")

#             self.prior_mu[1] = -1 / (2 * self.stddev_x_prior[0] ** 2)
#             self.prior_sigma[1] = self.mu[1] - (
#                 -1 / 2 * (self.stddev_x_prior[0] + self.stddev_x_prior[1]) ** 2
#             )

#         if self.stddev_y_prior is not None:
#             if not hasattr(self.stddev_y_prior, "__iter__"):
#                 raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")
#             if not len(self.stddev_y_prior) == 2:
#                 raise AttributeError("Pass stddev_y prior as a tuple with (mu, sigma)")

#             self.prior_mu[2] = -1 / (2 * self.stddev_y_prior[0] ** 2)
#             self.prior_sigma[2] = self.mu[2] - (
#                 -1 / 2 * (self.stddev_y_prior[0] + self.stddev_y_prior[1]) ** 2
#             )

#     @property
#     def width(self):
#         return 4

#     @property
#     def nvectors(self):
#         return 2

#     @property
#     def arg_names(self):
#         return {self.x_name, self.y_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "x_name",
#             "y_name",
#             "stddev_x_prior",
#             "stddev_y_prior",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of
#         y : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 4)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)
#         y = kwargs.get(self.y_name)
#         return np.vstack(
#             [
#                 np.ones(np.prod(x.shape)),
#                 x.ravel() ** 2,
#                 y.ravel() ** 2,
#                 x.ravel() * y.ravel(),
#             ]
#         ).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def A(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def rho(self):
#         rho = np.sqrt(self.mu[3] ** 2 / (self.mu[1] * self.mu[2])) / 2
#         err1 = np.abs((self.mu[3] ** 2) * 2 * self.sigma[3] / self.mu[3])
#         rho_err = np.abs(rho) * np.sqrt(
#             (err1 / self.mu[3]) ** 2
#             + (self.sigma[1] / self.mu[1]) ** 2
#             + (self.sigma[2] / self.mu[2]) ** 2
#         )
#         return rho, rho_err

#     @property
#     def stddev_x(self):
#         stddev_x = np.sqrt(-(1 / (2 * self.mu[1] * (1 - self.rho[0] ** 2))))
#         stddev_x_2_err = np.sqrt(
#             (
#                 (np.abs((self.rho[0]) * 2 * self.rho[1] / self.rho[0])) ** 2
#                 + (self.sigma[1] / self.mu[1]) ** 2
#             )
#         )
#         stddev_x_err = np.abs(stddev_x * -0.5 * (stddev_x_2_err / stddev_x**2))
#         return stddev_x, stddev_x_err

#     @property
#     def stddev_y(self):
#         stddev_y = np.sqrt(-(1 / (2 * self.mu[2] * (1 - self.rho[0] ** 2))))
#         stddev_y_2_err = np.sqrt(
#             (
#                 (np.abs((self.rho[0]) * 2 * self.rho[1] / self.rho[0])) ** 2
#                 + (self.sigma[2] / self.mu[2]) ** 2
#             )
#         )
#         stddev_y_err = np.abs(stddev_y * -0.5 * (stddev_y_2_err / stddev_y**2))
#         return stddev_y, stddev_y_err

#     @property
#     def table_properties(self):
#         return [
#             *[
#                 (
#                     "w_{idx}",
#                     (self.mu[idx], self.sigma[idx]),
#                     (self.prior_mu[idx], self.prior_sigma[idx]),
#                 )
#                 for idx in range(self.width)
#             ],
#             ("A", self.A, None),
#             (f"\\sigma_{{{self.x_name}}}", self.stddev_x, self.stddev_x_prior),
#             (f"\\sigma_{{{self.y_name}}}", self.stddev_y, self.stddev_y_prior),
#             ("\\rho", self.rho, None),
#         ]

#     @property
#     def _equation(self):
#         return [
#             f"\\mathbf{{{self.x_name}}}^0",
#             f"\mathbf{{{self.x_name}}}^{2}",
#             f"\mathbf{{{self.y_name}}}^{2}",
#             f"\mathbf{{{self.x_name}}}\mathbf{{{self.y_name}}}",
#         ]

#     def to_latex(self):
#         eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}
#                   \\ln(G(\\mathbf{{{self.x_name}}},
#                   \\mathbf{{{self.y_name}}})) =
#                    a + b\\mathbf{{{self.x_name}}}^2 + c\\mathbf{{{self.y_name}}}^2 +
#                   2d\\mathbf{{{self.x_name}}}\\mathbf{{{self.y_name}}}\\end{{equation}}"
#         eq2 = f"\\[ a = -\\ln(2\\pi\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}\\sqrt{{1-\\rho^2}}) \\]"
#         eq3 = f"\\[ b = \\frac{{1}}{{2(1-\\rho^2)\\sigma_{{{self.x_name}}}^2}}\\]"
#         eq4 = f"\\[ c = \\frac{{1}}{{2(1-\\rho^2)\\sigma_{{{self.y_name}}}^2}}\\]"
#         eq5 = f"\\[ d = \\frac{{\\rho}}{{2(1-\\rho^2)\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\]"
#         eq6 = f"\\[\\sigma_{{{self.x_name}}} = \\sqrt{{-\\frac{{1}}{{2b(1-\\rho^2)}}}}\\]"
#         eq7 = f"\\[\\sigma_{{{self.y_name}}} = \\sqrt{{-\\frac{{1}}{{2c(1-\\rho^2)}}}}\\]"
#         eq8 = "\\[\\rho = \\sqrt{\\frac{d^2}{bc}}\\]"
#         return "\n".join(
#             [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, self._to_latex_table()]
#         )

#     @property
#     def gradient(self):
#         return dlnGaussian2DGenerator(
#             stddev_x=self.stddev_x[0],
#             stddev_y=self.stddev_y[0],
#             rho=self.rho[0],
#             x_name=self.x_name,
#             y_name=self.y_name,
#             data_shape=self.data_shape,
#         )


# class dlnGaussian2DGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         stddev_x: float,
#         stddev_y: float,
#         rho: float = 0,
#         x_name: str = "x",
#         y_name: str = "y",
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.stddev_x = stddev_x
#         self.stddev_y = stddev_y
#         self.rho = rho
#         self.x_name = x_name
#         self.y_name = y_name
#         self._validate_arg_names()
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return 3

#     @property
#     def nvectors(self):
#         return 2

#     @property
#     def arg_names(self):
#         return {self.x_name, self.y_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "x_name",
#             "y_name",
#             "stddev_x",
#             "stddev_y",
#             "prior_mu",
#             "rho",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         x : np.ndarray
#             Vector to create ln Gaussian of
#         y : np.ndarray
#             Vector to create ln Gaussian of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), 4)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)
#         y = kwargs.get(self.y_name)

#         dfdx = (1 / 1 - self.rho**2) * (
#             (x / self.stddev_x**2) - self.rho * (y / (self.stddev_x * self.stddev_y))
#         )
#         dfdy = (1 / 1 - self.rho**2) * (
#             (y / self.stddev_y**2) - self.rho * (x / (self.stddev_x * self.stddev_y))
#         )
#         return np.vstack([np.ones(np.prod(dfdx.shape)), dfdx.ravel(), dfdy.ravel()]).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def shift_x(self):
#         return self.mu[1], self.sigma[1]

#     @property
#     def shift_y(self):
#         return self.mu[2], self.sigma[2]

#     @property
#     def table_properties(self):
#         return [
#             (
#                 "w_0",
#                 (self.mu[0], self.sigma[0]),
#                 (self.prior_mu[0], self.prior_sigma[0]),
#             ),
#             ("s_x", self.shift_x, (self.prior_mu[1], self.prior_sigma[1])),
#             ("s_y", self.shift_y, (self.prior_mu[2], self.prior_sigma[2])),
#         ]

#     @property
#     def _equation(self):
#         dfdx = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left
#           (\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}^2}}
#            - \\rho\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}
#           \\sigma_{{{self.y_name}}}}}\\right)\\right)"
#         dfdy = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left
#               (\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}^2}}
#            - \\rho\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}
#           \\sigma_{{{self.y_name}}}}}\\right)\\right)"
#         return [f"\\mathbf{{{self.x_name}}}^0", dfdx, dfdy]

#     @property
#     def _mu_letter(self):
#         return "v"
