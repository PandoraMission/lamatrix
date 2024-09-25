"""Generator objects for simple types of models"""

import numpy as np
from typing import List
from ..model import Model
from ..math import MathMixins
from ..combine import CrosstermModel

__all__ = [
    "Polynomial",
    "Constant",
    "Sinusoid",
    "dPolynomial",
    "dSinusoid",
]


class Polynomial(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        polyorder: int = 3,
        priors=None,
    ):
        if polyorder < 1:
            raise ValueError("Must have polyorder >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.polyorder = polyorder
        super().__init__(priors=priors)

    @property
    def width(self):
        return self.polyorder

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
        return np.asarray([x**idx for idx in range(1, self.polyorder + 1)]).transpose(shape_a)

    def to_gradient(self, priors=None):
        weights = [
            fit if fit is not None else prior
            for fit, prior in zip(self.fit_mean, self.prior_mean)
        ]
        return dPolynomial(
            weights=weights[1:],
            x_name=self.x_name,
            polyorder=self.polyorder - 1,
            priors=priors,
        )

    def __pow__(self, other):
        if other != 2:
            raise ValueError("Can only square `Model` objects")
        model = CrosstermModel(self, self)
        prior_std_cube = (np.tril(np.ones(self.width)) * (1 - np.tril(np.ones(self.width), -2))).astype(bool)
        model.set_priors([model.prior_distributions[idx] if i else (0, 1e-10) for idx, i in enumerate(prior_std_cube.ravel())])
        return model

class dPolynomial(MathMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        polyorder: int = 3,
        priors=None,
    ):
        if polyorder < 1:
            raise ValueError("Must have polyorder >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.polyorder = polyorder
        self._weight_width = self.polyorder
        self.weights = self._validate_weights(weights, self._weight_width)
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

    def design_matrix(self, **kwargs):
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
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        return np.expand_dims(np.asarray(
            [(idx + 1) * x**idx for idx in range(1, self.polyorder + 1)]
        ).transpose(shape_a).dot(self.weights), axis=x.ndim)


class Sinusoid(MathMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
    ):

        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        super().__init__(priors=priors)

    @property
    def width(self):
        return self.nterms * 2

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    def design_matrix(self, **kwargs):
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        shape_b = [x.ndim, *np.arange(0, x.ndim)]
        sin = np.asarray([np.sin(x * (idx + 1)) for idx in np.arange(self.nterms)]).transpose(shape_a)
        cos = np.asarray([np.cos(x * (idx + 1)) for idx in np.arange(self.nterms)]).transpose(shape_a)
        X = np.vstack([sin.transpose(shape_b), cos.transpose(shape_b)])
        # # Reorder to be sin, cos, sin, cos...
        R, C = np.mgrid[:2, :self.nterms]
        X = X[((R * self.nterms + C).T).ravel()]
        return X.transpose(shape_a)

    def to_gradient(self, priors=None):
        weights = [
            fit if fit is not None else prior
            for fit, prior in zip(self.fit_mean, self.prior_mean)
        ]
        return dSinusoid(
            weights=weights,
            x_name=self.x_name,
            nterms=self.nterms,
            priors=priors,
        )


class dSinusoid(MathMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
    ):

        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        self._weight_width = self.nterms * 2
        self.weights = self._validate_weights(weights, self._weight_width)
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

    def design_matrix(self, **kwargs):
        """

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
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        shape_b = [x.ndim, *np.arange(0, x.ndim)]
        dsin = np.asarray([np.cos(x * (idx + 1)) for idx in np.arange(self.nterms)]).transpose(shape_a)
        dcos = np.asarray([-np.sin(x * (idx + 1)) for idx in np.arange(self.nterms)]).transpose(shape_a)
        X = np.vstack([dsin.transpose(shape_b), dcos.transpose(shape_b)])
        # Reorder to be sin, cos, sin, cos...
        R, C = np.mgrid[:2, :self.nterms]
        X = X[((R * self.nterms + C).T).ravel()]
        return np.expand_dims(X.transpose(shape_a).dot(self.weights), axis=x.ndim)

        # shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        # return np.asarray(
        #     [
        #         *[
        #             [w1 * np.cos(x * (idx + 1)), -w2 * np.sin(x * (idx + 1))]
        #             for idx, w1, w2 in zip(
        #                 range(self.nterms),
        #                 self.weights[::2],
        #                 self.weights[1::2],
        #             )
        #         ],
        #     ]
        # ).transpose(shape_a).sum(axis=1)[:, None]


class Constant(MathMixins, Model):
    """
    A generator which has no variable, and whos design matrix is entirely ones.
    """

    def __init__(self, priors=None):
        super().__init__(priors=priors)

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 0

    @property
    def arg_names(self):
        return {}

    def design_matrix(self, **kwargs):
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
        if len(kwargs) < 1:
            raise ValueError("Cannot create design matrix without regressors.")

        x = np.ones_like(next(iter(kwargs.values())))
        return np.expand_dims(x, axis=x.ndim)

    # @property
    # def gradient(self):
    #     return dPolynomial1DGenerator(
    #         weights=self._mu,
    #         x_name=self.x_name,
    #         polyorder=self.polyorder,
    #         data_shape=self.data_shape,
    #         offset_prior=(self._mu[1], self._sigma[1]),
    #     )


# class Polynomial1DGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         x_name: str = "x",
#         polyorder: int = 3,
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.polyorder = polyorder
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return self.polyorder + 1

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
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#             "polyorder",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         {} : np.ndarray
#             Vector to create polynomial of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), self.nvectors)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()
#         return np.vstack([x**idx for idx in range(self.polyorder + 1)]).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def offset(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def _equation(self):
#         eqn = [
#             f"\mathbf{{{self.x_name}}}^{{{idx}}}" for idx in range(self.polyorder + 1)
#         ]
#         #        eqn[0] = ""
#         return eqn

#     @property
#     def gradient(self):
#         return dPolynomial1DGenerator(
#             weights=self.mu,
#             x_name=self.x_name,
#             polyorder=self.polyorder,
#             data_shape=self.data_shape,
#             offset_prior=(self.mu[1], self.sigma[1]),
#         )


# class dPolynomial1DGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         weights: list,
#         x_name: str = "x",
#         polyorder: int = 3,
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.weights = weights
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.polyorder = polyorder
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return self.polyorder

#     @property
#     def nvectors(self):
#         return 1

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "weights",
#             "x_name",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#             "polyorder",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         {} : np.ndarray
#             Vector to create polynomial of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), self.nvectors)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()
#         return np.vstack(
#             [
#                 (idx + 1) * w * x**idx
#                 for idx, w in zip(range(self.polyorder), self.weights[1:])
#             ]
#         ).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def offset(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def _mu_letter(self):
#         return "v"

#     @property
#     def _equation(self):
#         eqn = [
#             f"({idx + 1}w_{{{idx + 1}}}\mathbf{{{self.x_name}}}^{{{idx}}})"
#             for idx in range(self.polyorder)
#         ]
#         #        eqn[0] = ""
#         return eqn


# class ConstantGenerator(Polynomial1DGenerator):
#     def __init__(
#             self,
#             x_name: str = "x",
#             prior_mu=None,
#             prior_sigma=None,
#             offset_prior=None,
#             data_shape=None,
#         ):

#        return super().__init__(x_name=x_name, polyorder=0, prior_mu=prior_mu, prior_sigma=prior_sigma, offset_prior=offset_prior, data_shape=data_shape)


# class SinusoidGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         x_name: str = "x",
#         nterms: int = 1,
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.nterms = nterms
#         self.data_shape = data_shape
#         self.x_name = x_name
#         self._validate_arg_names()
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return (self.nterms * 2) + 1

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
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#             "nterms",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         {} : np.ndarray
#             Vector to create polynomial of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), self.nvectors)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()
#         return np.vstack(
#             [
#                 x**0,
#                 *[
#                     [np.sin(x * (idx + 1)), np.cos(x * (idx + 1))]
#                     for idx in range(self.nterms)
#                 ],
#             ]
#         ).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def _equation(self):
#         def frq(term):
#             return f"{term + 1}" if term > 0 else ""

#         return np.hstack(
#             [
#                 f"\\mathbf{{{self.x_name}}}^0",
#                 *[
#                     [
#                         f"\sin({frq(idx)}\\mathbf{{{self.x_name}}})",
#                         f"\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
#                     ]
#                     for idx in range(self.nterms)
#                 ],
#             ]
#         )

#     @property
#     def gradient(self):
#         return dSinusoidGenerator(
#             nterms=self.nterms,
#             weights=self.mu,
#             x_name=self.x_name,
#             data_shape=self.data_shape,
#         )


# class dSinusoidGenerator(MathMixins, Generator):
#     def __init__(
#         self,
#         weights: list,
#         x_name: str = "x",
#         nterms: int = 1,
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         self.nterms = nterms
#         self.weights = weights
#         self.data_shape = data_shape
#         self.x_name = x_name
#         self._validate_arg_names()
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return (self.nterms * 2) + 1

#     @property
#     def nvectors(self):
#         return 1

#     @property
#     def arg_names(self):
#         return {self.x_name}

#     @property
#     def _INIT_ATTRS(self):
#         return [
#             "weights",
#             "x_name",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#             "nterms",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D polynomial in x

#         Parameters
#         ----------
#         {} : np.ndarray
#             Vector to create polynomial of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), self.nvectors)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()
#         return np.vstack(
#             [
#                 x**0,
#                 *[
#                     [w1 * np.cos(x * (idx + 1)), -w2 * np.sin(x * (idx + 1))]
#                     for idx, w1, w2 in zip(
#                         range(self.nterms),
#                         self.weights[1:][::2],
#                         self.weights[1:][1::2],
#                     )
#                 ],
#             ]
#         ).T

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def _equation(self):
#         def frq(term):
#             return f"{term + 1}" if term > 0 else ""

#         return np.hstack(
#             [
#                 f"\\mathbf{{{self.x_name}}}^0",
#                 *[
#                     [
#                         f"w_{{{idx * 2 + 1}}}\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
#                         f"w_{{{idx * 2 + 2}}}(-\sin({frq(idx)}\\mathbf{{{self.x_name}}}))",
#                     ]
#                     for idx in range(self.nterms)
#                 ],
#             ]
#         )

#     @property
#     def _mu_letter(self):
#         return "v"
