"""Generator objects for simple types of models"""

import numpy as np

from ..generator import Generator
from ..math import MathMixins

__all__ = [
    "Polynomial1DGenerator",
    "SinusoidGenerator",
    "dSinusoidGenerator",
]


class Polynomial1DGenerator(MathMixins, Generator):
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
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return self.polyorder + 1

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
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
            "polyorder",
        ]

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
    def _equation(self):
        eqn = [
            f"\mathbf{{{self.x_name}}}^{{{idx}}}" for idx in range(self.polyorder + 1)
        ]
        #        eqn[0] = ""
        return eqn

    @property
    def gradient(self):
        return dPolynomial1DGenerator(
            weights=self.mu,
            x_name=self.x_name,
            polyorder=self.polyorder,
            data_shape=self.data_shape,
            offset_prior=(self.mu[1], self.sigma[1]),
        )


class dPolynomial1DGenerator(MathMixins, Generator):
    def __init__(
        self,
        weights: list,
        x_name: str = "x",
        polyorder: int = 3,
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.weights = weights
        self.x_name = x_name
        self._validate_arg_names()
        self.polyorder = polyorder
        self.data_shape = data_shape
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
        self.fit_mu = None
        self.fit_sigma = None

    @property
    def width(self):
        return self.polyorder

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _INIT_ATTRS(self):
        return [
            "weights",
            "x_name",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
            "polyorder",
        ]

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
                (idx + 1) * w * x**idx
                for idx, w in zip(range(self.polyorder), self.weights[1:])
            ]
        ).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def offset(self):
        return self.mu[0], self.sigma[0]

    @property
    def _equation(self):
        eqn = [
            f"({idx + 1}v_{{\\textit{{poly}}, {idx + 1}}}\mathbf{{{self.x_name}}}^{{{idx}}})"
            for idx in range(self.polyorder)
        ]
        #        eqn[0] = ""
        return eqn


class SinusoidGenerator(MathMixins, Generator):
    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.nterms = nterms
        self.data_shape = data_shape
        self.x_name = x_name
        self._validate_arg_names()
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
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

    @property
    def _INIT_ATTRS(self):
        return [
            "x_name",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
            "nterms",
        ]

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
    def _equation(self):
        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                f"\\mathbf{{{self.x_name}}}^0",
                *[
                    [
                        f"\sin({frq(idx)}\\mathbf{{{self.x_name}}})",
                        f"\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
                    ]
                    for idx in range(self.nterms)
                ],
            ]
        )

    @property
    def gradient(self):
        return dSinusoidGenerator(
            nterms=self.nterms,
            weights=self.mu,
            x_name=self.x_name,
            data_shape=self.data_shape,
        )


class dSinusoidGenerator(MathMixins, Generator):
    def __init__(
        self,
        weights: list,
        x_name: str = "x",
        nterms: int = 1,
        prior_mu=None,
        prior_sigma=None,
        offset_prior=None,
        data_shape=None,
    ):
        self.nterms = nterms
        self.weights = weights
        self.data_shape = data_shape
        self.x_name = x_name
        self._validate_arg_names()
        self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
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

    @property
    def _INIT_ATTRS(self):
        return [
            "weights",
            "x_name",
            "prior_mu",
            "prior_sigma",
            "offset_prior",
            "data_shape",
            "nterms",
        ]

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
                    [w1 * np.cos(x * (idx + 1)), -w2 * np.sin(x * (idx + 1))]
                    for idx, w1, w2 in zip(
                        range(self.nterms),
                        self.weights[1:][::2],
                        self.weights[1:][1::2],
                    )
                ],
            ]
        ).T

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    @property
    def _equation(self):
        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                f"\\mathbf{{{self.x_name}}}^0",
                *[
                    [
                        f"v_{{\\textit{{sin}}, {idx * 2}}}\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
                        f"v_{{\\textit{{sin}}, {idx * 2 + 1}}}(-\sin({frq(idx)}\\mathbf{{{self.x_name}}}))",
                    ]
                    for idx in range(self.nterms)
                ],
            ]
        )
