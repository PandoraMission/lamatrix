"""Generator objects for simple types of models"""

import numpy as np

from ..generator import Generator
from ..math import MathMixins

__all__ = [
    "Polynomial1DGenerator",
    "SinusoidGenerator",
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
    def _equation(self):
        eqn = [
            f"\mathbf{{{self.x_name}}}^{{{idx}}}" for idx in range(self.polyorder + 1)
        ]
        eqn[0] = ""
        return eqn


class SinusoidGenerator(MathMixins, Generator):
    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        prior_mu=None,
        prior_sigma=None,
        data_shape=None,
    ):
        self.nterms = nterms
        self.data_shape = data_shape
        self.x_name = x_name
        self._validate_arg_names()
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
    def _equation(self):
        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                "",
                *[
                    [
                        f"\sin({frq(idx)}\\mathbf{{{self.x_name}}})",
                        f"\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
                    ]
                    for idx in range(self.nterms)
                ],
            ]
        )
