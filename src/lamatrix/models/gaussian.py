"""Generator objects for Gaussian types of models"""

import numpy as np

from ..model import Model
from ..math import MathMixins
from ..io import LatexMixins, IOMixins

__all__ = [
    "Gaussian",
    "lnGaussian",
    "Gaussian2D",
    "lnGaussian2D",
]


def _coeff_to_sigma1d(distributions):
    a, aerr = distributions[0]
    sigma = (-1 / (2 * a)) ** 0.5
    sigma_err = (1 / (2 * np.sqrt(2))) * (1 / ((-a) ** (3 / 2))) * aerr
    return (sigma, sigma_err)

def _sigma1d_to_coeff(distribution):
    sigma, sigma_err = distribution
    coeff = -1 / (2 * sigma**2)
    coeff_err = (1 / sigma**3) * sigma_err
    return (coeff, coeff_err)

def _coeff_to_mu1d(distributions):
    a, aerr = distributions[0]
    b, berr = distributions[1]
    mu = - (b / (2 * a))
    mu_err = mu * ((berr/b)**2 + (aerr/a)**2)**0.5
    return (mu, mu_err)

def _coeff_to_A1d(distributions):
    sigma, sigma_err = _coeff_to_sigma1d(distributions)
    mu, mu_err = _coeff_to_mu1d(distributions)
    c, cerr = distributions[2]
    A = np.exp(c + mu**2/2*sigma**2 + 0.5*(np.log(2*np.pi*sigma**2)))
    Aerr = A * np.sqrt(
        cerr**2 + 
        (mu / sigma**2 * mu_err)**2 + 
        ((1 / (2 * sigma) - mu**2 / (2 * sigma**3)) * sigma_err)**2
    )
    return (A, Aerr)


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
        sigma:float=1.,
        mu:float=0.,
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
        x = kwargs.get(self.x_name).ravel()
        normalization = (1/(2 * np.pi * self.sigma**2)**0.5)
        return normalization * np.exp(-0.5 * (x - self.mu) ** 2 / (self.sigma**2))[:, None]

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def posterior_amplitude(self):
        return self.posteriors[0]

    # def to_gradient(self, prior_distributions=None):
    #     return dGaussian(
    #         x_name=self.x_name,
    #         sigma=self.sigma,
    #         mu=self.mu,
    #         prior_distributions=prior_distributions,
    #     )


class dGaussian(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        sigma:float=1.,
        mu:float=0.,
        priors=None,
        posteriors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.sigma = sigma
        self.mu = mu
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

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `x_name`.

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name).ravel()
        normalization = (1/(2 * np.pi * self.sigma**2)**0.5)
        f = normalization * np.exp(-0.5 * (x - self.mu) ** 2 / (self.sigma**2))
        return (-((x - self.mu) / self.sigma**2) * f)[:, None]

    @property
    def prior_amplitude(self):
        return self.priors[0]

    @property
    def posterior_amplitude(self):
        return self.posteriors[0]


class Gaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        prior_distributions=None,
        sigma_x=1,
        sigma_y=1,
        mu_x=0,
        mu_y=0,
        rho=0.,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.mu_x, self.mu_y = mu_x, mu_y
        self.rho = rho
        super().__init__(prior_distributions=prior_distributions)

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
        x = kwargs.get(self.x_name).ravel() - self.mu_x
        y = kwargs.get(self.y_name).ravel() - self.mu_y
        normalization = (1/(2 * np.pi * self.sigma_x * self.sigma_y * (1 - self.rho**2)**0.5))
        p = (- 1 / (2 * (1 - self.rho**2)))
        exponent = p * (
                (x / self.sigma_x) ** 2
                + (y / self.sigma_y) ** 2
                - 2
                * self.rho
                * (x / self.sigma_x)
                * (y / self.sigma_y)
            )
        return normalization * np.exp(exponent)[:, None]

    @property
    def prior_amplitude(self):
        return self.prior_distributions[0]

    @property
    def fit_amplitude(self):
        return self.fit_distributions[0]

    def to_gradient(self, prior_distributions=None):
        return dGaussian2D(
            x_name=self.x_name,
            y_name=self.y_name,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            mu_x=self.mu_x,
            mu_y=self.mu_y,
            rho=self.rho,
            prior_distributions=prior_distributions,
        )


class dGaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        prior_distributions=None,
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
        super().__init__(prior_distributions=prior_distributions)

    @property
    def width(self):
        return 2

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
        x = kwargs.get(self.x_name).ravel() - self.mu_x
        y = kwargs.get(self.y_name).ravel() - self.mu_y
        normalization = (1/(2 * np.pi * self.sigma_x * self.sigma_y * (1 - self.rho**2)**0.5))
        p = (- 1 / (2 * (1 - self.rho**2)))
        exponent = p * (
                (x / self.sigma_x) ** 2
                + (y / self.sigma_y) ** 2
                - 2
                * self.rho
                * (x / self.sigma_x)
                * (y / self.sigma_y)
            )
        f = np.exp(exponent)
        dfdx =  (
            normalization * f
            * (- 1 / (1 - self.rho**2))
            * (
                x / self.sigma_x**2
                - (self.rho * (y) / (self.sigma_x * self.sigma_y))
            )
        )
        dfdy =  (
            normalization * f
            * (- 1 / (1 - self.rho**2))
            * (
                (y) / self.sigma_y**2
                - (self.rho * x / (self.sigma_x * self.sigma_y))
            )
        )
        return np.vstack([dfdx, dfdy]).T

    @property
    def prior_amplitude(self):
        return self.prior_distributions[0]

    @property
    def fit_amplitude(self):
        return self.fit_distributions[0]


class lnGaussian(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        #mu: float = 0,
        prior_distributions=None,
        prior_sigma_distribution=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.prior_sigma_distribution = prior_sigma_distribution
        #self.mu = mu
        super().__init__(prior_distributions=prior_distributions)
        if self.prior_sigma_distribution is not None:
            if not hasattr(self.prior_sigma_distribution, "__iter__"):
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            if not len(self.prior_sigma_distribution) == 2:
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            self.prior_distributions[0] = _sigma1d_to_coeff(
                self.prior_sigma_distribution
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
        x = kwargs.get(self.x_name) #- self.mu
        return np.vstack([x**2, x]).T
#        return x[:, None] ** 2

    @property
    def prior_sigma(self):
        return _coeff_to_sigma1d(self.prior_distributions)

    @property
    def fit_sigma(self):
        return _coeff_to_sigma1d(self.fit_distributions)

    @property
    def fit_mu(self):
        return _coeff_to_mu1d(self.fit_distributions)

    # @property
    # def fit_amplitude(self):
    #     return _coeff_to_A1d(self.fit_distributions)


    def to_gradient(self, prior_distributions=None):
        return dlnGaussian(
            x_name=self.x_name,
            sigma=self.fit_sigma[0],
            mu=self.fit_mu[0],
            prior_distributions=prior_distributions,
        )


class dlnGaussian(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        mu: float = 0,
        sigma: float = 0,
        prior_distributions=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.mu = mu
        self.sigma = sigma
        super().__init__(prior_distributions=prior_distributions)

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
        x = kwargs.get(self.x_name) - self.mu
        return -x[:, None] / self.sigma**2


class lnGaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        mu_x: float = 0,
        mu_y: float = 0,
        prior_distributions=None,
        prior_sigma_x_distribution=None,
        prior_sigma_y_distribution=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.mu_x, self.mu_y = mu_x, mu_y
        super().__init__(prior_distributions=prior_distributions)

        self.prior_sigma_x_distribution, self.prior_sigma_y_distribution = (
            prior_sigma_x_distribution,
            prior_sigma_y_distribution,
        )
        if self.prior_sigma_x_distribution is not None:
            if not hasattr(self.prior_sigma_x_distribution, "__iter__"):
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            if not len(self.prior_sigma_x_distribution) == 2:
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            self.prior_distributions[0] = _sigma1d_to_coeff(
                self.prior_sigma_x_distribution
            )

        if self.prior_sigma_y_distribution is not None:
            if not hasattr(self.prior_sigma_y_distribution, "__iter__"):
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            if not len(self.prior_sigma_y_distribution) == 2:
                raise AttributeError("Pass sigma_x prior as a tuple with (mean, std)")
            self.prior_distributions[1] = _sigma1d_to_coeff(
                self.prior_sigma_y_distribution
            )

    @property
    def width(self):
        return 3

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, **kwargs):
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
        x = kwargs.get(self.x_name) - self.mu_x
        y = kwargs.get(self.y_name) - self.mu_y
        return np.vstack(
            [
                x.ravel() ** 2,
                y.ravel() ** 2,
                x.ravel() * y.ravel(),
            ]
        ).T

    @property
    def prior_sigma_x(self):
        return _coeff_to_sigma2d(self.prior_distributions[0], self.prior_rho)

    @property
    def fit_sigma_x(self):
        return _coeff_to_sigma2d(self.fit_distributions[0], self.fit_rho)

    @property
    def prior_sigma_y(self):
        return _coeff_to_sigma2d(self.prior_distributions[1], self.prior_rho)

    @property
    def fit_sigma_y(self):
        return _coeff_to_sigma2d(self.fit_distributions[1], self.fit_rho)

    @property
    def prior_rho(self):
        return _coeffs_to_rho(self.prior_distributions)

    @property
    def fit_rho(self):
        return _coeffs_to_rho(self.fit_distributions)

    def to_gradient(self, prior_distributions=None):
        return dlnGaussian2D(
            x_name=self.x_name,
            y_name=self.y_name,
            sigma_x=self.fit_sigma_x[0],
            sigma_y=self.fit_sigma_y[0],
            mu_x=self.mu_x,
            mu_y=self.mu_y,
            rho=self.fit_rho[0],
            prior_distributions=prior_distributions,
        )


class dlnGaussian2D(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        y_name: str = "y",
        mu_x: float = 0,
        mu_y: float = 0,
        sigma_x: float = 0,
        sigma_y: float = 0,
        rho: float = 0,
        prior_distributions=None,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self._validate_arg_names()
        self.mu_x, self.mu_y = mu_x, mu_y
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.rho = rho
        super().__init__(prior_distributions=prior_distributions)

    @property
    def width(self):
        return 2

    @property
    def nvectors(self):
        return 2

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, **kwargs):
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
        dfdx = - 1 / (1 - self.rho**2) * (
            (x - self.mu_x) / self.sigma_x**2
            - (self.rho * (y - self.mu_y) / (self.sigma_x * self.sigma_y))
        )
        dfdy = - 1 / (1 - self.rho**2) * (
            (y - self.mu_y) / self.sigma_y**2
            - (self.rho * (x - self.mu_x) / (self.sigma_x * self.sigma_y))
        )
        return np.vstack([dfdx, dfdy]).T


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
#         eq0 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}})) = -\\frac{{1}}{{2}} \\ln(2\\pi\\sigma^2) + \\frac{{\\mathbf{{{self.x_name}}}^2}}{{2 \\sigma^2}}\\end{{equation}}"
#         eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}})) = w_0 + w_1\\mathbf{{{self.x_name}}}^2\\end{{equation}}"
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
#         eq1 = f"\\begin{{equation}}\\label{{eq:lngauss}}\\ln(G(\\mathbf{{{self.x_name}}}, \\mathbf{{{self.y_name}}})) = a + b\\mathbf{{{self.x_name}}}^2 + c\\mathbf{{{self.y_name}}}^2 + 2d\\mathbf{{{self.x_name}}}\\mathbf{{{self.y_name}}}\\end{{equation}}"
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
#         dfdx = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left(\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}^2}} - \\rho\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\right)\\right)"
#         dfdy = f"\\left(-\\frac{{1}}{{1-\\rho^2}}\\left(\\frac{{\\mathbf{{{self.y_name}}}}}{{\\sigma_{{{self.x_name}}}^2}} - \\rho\\frac{{\\mathbf{{{self.x_name}}}}}{{\\sigma_{{{self.x_name}}}\\sigma_{{{self.y_name}}}}}\\right)\\right)"
#         return [f"\\mathbf{{{self.x_name}}}^0", dfdx, dfdy]

#     @property
#     def _mu_letter(self):
#         return "v"
