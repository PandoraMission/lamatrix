"""Generator objects for simple types of models"""

import warnings
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

from ..combine import CrosstermModel
from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Model

__all__ = [
    "Polynomial",
    "Constant",
    "Sinusoid",
    "dPolynomial",
    "dSinusoid",
    "Step",
    "Fixed",
]


class Polynomial(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        if order < 1:
            raise ValueError("Must have order >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return self.order

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
            "order",
        ]

    @property
    def _equation(self):
        eqn = [
            f"\mathbf{{{self.x_name}}}^{{{idx}}}" for idx in range(1, self.order + 1)
        ]
        return eqn

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
        if sparse.issparse(x):
            if not x.shape[1] == 1:
                raise ValueError(
                    f"Can only fit sparse matrices with shape (n, 1), {self.x_name} has shape {x.shape}."
                )
            return sparse.hstack(
                [x.power(idx) for idx in range(1, self.order + 1)], format="csr"
            )
        else:
            shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
            return np.asarray([x**idx for idx in range(1, self.order + 1)]).transpose(
                shape_a
            )

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        return dPolynomial(
            weights=weights[1:],
            x_name=self.x_name,
            order=self.order - 1,
            priors=priors,
        )

    def __pow__(self, other):
        if other != 2:
            raise ValueError("Can only square `Model` objects")
        model = CrosstermModel(self, self)
        prior_std_cube = (
            np.tril(np.ones(self.width)) * (1 - np.tril(np.ones(self.width), -2))
        ).astype(bool)
        model.set_priors(
            [
                model.prior_distributions[idx] if i else (0, 1e-10)
                for idx, i in enumerate(prior_std_cube.ravel())
            ]
        )
        return model


class dPolynomial(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        if order < 1:
            raise ValueError("Must have order >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        self._weight_width = self.order
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

    @property
    def _equation(self):
        eqn = [
            f"{idx + 1 if idx != 0 else ''}w_{{{idx}}}\mathbf{{{self.x_name}}}^{{{idx}}}"
            for idx in range(self.order + 1)
        ]
        return eqn

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
        if sparse.issparse(x):
            if not x.shape[1] == 1:
                raise ValueError(
                    f"Can only fit sparse matrices with shape (n, 1), {self.x_name} has shape {x.shape}."
                )
            return sparse.hstack(
                [x.power(idx).multiply(idx + 1) for idx in range(1, self.order + 1)],
                format="csr",
            ).dot(sparse.csr_matrix(self.weights).T)
        else:
            shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
            return np.expand_dims(
                np.asarray([(idx + 1) * x**idx for idx in range(1, self.order + 1)])
                .transpose(shape_a)
                .dot(self.weights),
                axis=x.ndim,
            )


class Sinusoid(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
        posteriors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return self.nterms * 2

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _equation(self):
        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                [
                    f"\sin({frq(idx)}\\mathbf{{{self.x_name}}})",
                    f"\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
                ]
                for idx in range(self.nterms)
            ]
        )

    def design_matrix(self, **kwargs):
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        if sparse.issparse(x):
            raise ValueError(
                "Can not make a sparse verison of this object. Zero valued inputs are not zero valued outputs."
            )
        else:
            shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
            shape_b = [x.ndim, *np.arange(0, x.ndim)]
            sin = np.asarray(
                [np.sin(x * (idx + 1)) for idx in np.arange(self.nterms)]
            ).transpose(shape_a)
            cos = np.asarray(
                [np.cos(x * (idx + 1)) for idx in np.arange(self.nterms)]
            ).transpose(shape_a)
            X = np.vstack([sin.transpose(shape_b), cos.transpose(shape_b)])
            # # Reorder to be sin, cos, sin, cos...
            R, C = np.mgrid[:2, : self.nterms]
            X = X[((R * self.nterms + C).T).ravel()]
            return X.transpose(shape_a)

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        return dSinusoid(
            weights=weights,
            x_name=self.x_name,
            nterms=self.nterms,
            priors=priors,
        )


class dSinusoid(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
        posteriors=None,
    ):
        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        self._weight_width = self.nterms * 2
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

    @property
    def _equation(self):
        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                [
                    f"w_{{{idx * 2}}}\cos({frq(idx)}\\mathbf{{{self.x_name}}})",
                    f"w_{{{idx * 2 + 1}}}(-\sin({frq(idx)}\\mathbf{{{self.x_name}}}))",
                ]
                for idx in range(self.nterms)
            ],
        )

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
        if sparse.issparse(x):
            raise ValueError(
                "Can not make a sparse verison of this object. Zero valued inputs are not zero valued outputs."
            )
        else:
            shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
            shape_b = [x.ndim, *np.arange(0, x.ndim)]
            dsin = np.asarray(
                [np.cos(x * (idx + 1)) for idx in np.arange(self.nterms)]
            ).transpose(shape_a)
            dcos = np.asarray(
                [-np.sin(x * (idx + 1)) for idx in np.arange(self.nterms)]
            ).transpose(shape_a)
            X = np.vstack([dsin.transpose(shape_b), dcos.transpose(shape_b)])
            # Reorder to be sin, cos, sin, cos...
            R, C = np.mgrid[:2, : self.nterms]
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


class Constant(MathMixins, LatexMixins, IOMixins, Model):
    """
    A generator which has no variable, and whos design matrix is entirely ones.
    """

    def __init__(self, priors=None, posteriors=None):
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return 1

    @property
    def nvectors(self):
        return 0

    @property
    def arg_names(self):
        return {}

    @property
    def _equation(self):
        return [""]

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

    @property
    def _initialization_attributes(self):
        return []

    # @property
    # def gradient(self):
    #     return dPolynomial1DGenerator(
    #         weights=self._mu,
    #         x_name=self.x_name,
    #         order=self.order,
    #         data_shape=self.data_shape,
    #         offset_prior=(self._mu[1], self._sigma[1]),
    #     )


class Step(MathMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        breakpoints: list[float] = [0],
        priors=None,
        posteriors=None,
    ):
        if len(breakpoints) == 0:
            raise ValueError("Must have at least one breakpoint")
        self.x_name = x_name
        self._validate_arg_names()
        self.breakpoints = np.sort(breakpoints)
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return len(self.breakpoints) + 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
            "breakpoints",
        ]

    @property
    def _equation(self):
        bounds = np.hstack(["-\infty", self.breakpoints, "\infty"])
        eqn = [
            f"\mathbb{{I}}_{{[{{{bounds[idx]}}}, {{{bounds[idx + 1]}}}]}}(\mathbf{{{self.x_name}}})"
            for idx in range(0, self.width)
        ]
        return eqn

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
        bounds = np.hstack([-np.inf, self.breakpoints, np.inf])
        if sparse.issparse(x):
            if 0 in bounds:
                raise ValueError(
                    "Can not create a Step function with a boundary at 0 if you supply a sparse matrix."
                )
            if not x.shape[1] == 1:
                raise ValueError(
                    f"Can only fit sparse matrices with shape (n, 1), {self.x_name} has shape {x.shape}."
                )
            vectors = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SparseEfficiencyWarning)
                for idx in range(0, self.width):
                    vectors.append((x >= bounds[idx]).multiply(x < bounds[idx + 1]))
            return sparse.hstack(vectors, format="csr")
        else:
            shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
            return (
                np.asarray(
                    [
                        (x >= bounds[idx]) & (x < bounds[idx + 1])
                        for idx in range(0, self.width)
                    ]
                )
                .transpose(shape_a)
                .astype(float)
            )


class Fixed(MathMixins, LatexMixins, IOMixins, Model):
    """A model that has fixed input vectors"""

    def __init__(
        self,
        x_name: str = "x",
        width: int = 1,
        priors=None,
        posteriors=None,
    ):
        """A model with fixed vector inputs"""
        self.x_name = x_name
        # We give this a hidden name so we can comply with the ABC
        self._width = width
        self._validate_arg_names()
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return self._width

    @property
    def nvectors(self):
        return self.width

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
            "width",
        ]

    @property
    def _equation(self):
        eqn = [f"\mathbf{{{self.x_name}}}_{{{idx}}}" for idx in range(0, self.width)]
        return eqn

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
        if not x.shape[-1] == self.width:
            raise ValueError(
                f"Must pass a vector for {self.x_name} that has width {self.width}."
            )
        return x
