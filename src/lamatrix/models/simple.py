"""Generator objects for simple types of models."""

import warnings
from typing import List

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import SparseEfficiencyWarning

from ..distributions import DistributionsContainer
from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Model
from ..combine import CrosstermModel

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
    """lamatrix.Model object to model polynomial trends."""

    def __init__(
        self,
        x_name: str = "x",
        order: int = 3,
        priors: DistributionsContainer = None,
        posteriors: DistributionsContainer = None,
    ) -> Model:
        """
        Initialize a Polynomial model.

        Note that this model does not include a constant term. You can add one using the "Constant" model.

        Parameters
        ----------
        x_name : str, optional
            The name of the independent variable (default is "x").
        order : int, optional
            The order of the polynomial. Must be at least 1 (default is 3).
        priors : optional
            Prior distributions for model weights (default is None, i.e. no priors).
        posteriors : optional
            Posterior distributions for model parameters (default is None, i.e. no posteriors).
            Posterior keyword is provided so that models can be loaded.

        Raises
        ------
        ValueError
            If `order` is less than 1.
        ValueError
            If `order` is not an integer.

        Examples
        --------
        Create a cubic polynomial model:

        >>> model = Polynomial(x_name="t", order=3)
        """
        if order < 1:
            raise ValueError("Must have order >= 1.")
        if not np.issubdtype(type(order), np.integer):
            raise ValueError("`order` must be an integer.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        """Width of the model. This is equivalent to the order."""
        return self.order

    @property
    def nvectors(self):
        """Number of vectors required to be input to create the model."""
        return 1

    @property
    def arg_names(self):
        """Argument names that must be input into the model."""
        return {self.x_name}

    @property
    def _initialization_attributes(self):
        """The parameters that are required to initialize the object. This is used by the `load` function."""
        return [
            "x_name",
            "order",
        ]

    @property
    def _equation(self):
        """Provides the list of equation components without the weights included, to be used by `self.equation`."""
        eqn = [
            f"\mathbf{{{self.latex_aliases[self.x_name]}}}^{{{idx}}}"
            for idx in range(1, self.order + 1)
        ]
        return eqn

    def design_matrix(self, **kwargs):
        """Build a 1D polynomial in `self.x_name`.

        You must pass the keyword argument `self.x_name` to use this function.

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

    def to_gradient(
        self, weights: npt.NDArray = None, priors: DistributionsContainer = None
    ):
        """Converts this model to the gradient of that model, assigning the posteriors of the fit as weights.

        If the posteriors are None, will assign the priors as weights.

        Parameters
        ----------
        weights: np.ndarray
            The weights applied to the Polynomial model. These update the commponents for the design matrix of the gradient.
        priors: DistributionsContainer
            Priors to apply to the gradient model.

        Returns
        -------
        model: lamatrix.models.simple.dPolynomial
            A model for the gradient of a polynomial

        Example
        -------

        To obtain the gradient of a fit polynomial model

        >>> model = Polynomial(x_name="t", order=3)
        >>> model.fit(t=t, data=data, errors=errors)
        >>> dmodel = model.to_gradient()
        """
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        else:
            raise ValueError(
                "To convert to a gradient, you must provide either posteriors from fitting data or priors."
            )
        return dPolynomial(
            weights=weights[1:],
            x_name=self.x_name,
            order=self.order - 1,
            priors=priors,
        )

    def __pow__(self, other):
        """Special case for Polynomial models when raised to powers.

        In the polynomial case raising the model to a given power creates
        duplicated components making the model degenerate.
        """
        if other != 2:
            raise ValueError("Can only square `Model` objects")
        model = CrosstermModel(self, self)
        prior_std_cube = (
            np.tril(np.ones(self.width)) * (1 - np.tril(np.ones(self.width), -2))
        ).astype(bool)
        model.priors = DistributionsContainer(
            [
                model.priors[idx] if i else (0, 1e-10)
                for idx, i in enumerate(prior_std_cube.ravel())
            ]
        )
        return model


class dPolynomial(MathMixins, LatexMixins, IOMixins, Model):
    """lamatrix.Model object for capturing derivatives of polynomial models.

    In this case, this is a special variant on a Polynomial model.
    """

    def __init__(
        self,
        weights: npt.NDArray,
        x_name: str = "x",
        order: int = 3,
        priors=None,
        posteriors=None,
    ) -> Model:
        """
        Initialize a dPolynomial model.

        Note that this model does not include a constant term. You can add one using the "Constant" model.

        Parameters
        ----------
        weights: npt.NDArray,
            The input weights to be applied to the input model.
        x_name : str, optional
            The name of the independent variable (default is "x").
        order : int, optional
            The order of the polynomial. Must be at least 1 (default is 3).
        priors : optional
            Prior distributions for model weights (default is None, i.e. no priors).
        posteriors : optional
            Posterior distributions for model parameters (default is None, i.e. no posteriors).
            Posterior keyword is provided so that models can be loaded.

        Raises
        ------
        ValueError
            If `order` is less than 1.
        ValueError
            If `order` is not an integer.
        """
        if order < 1:
            raise ValueError("Must have `order` >= 1.")
        if not np.issubdtype(type(order), np.integer):
            raise ValueError("`order` must be an integer.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        self._weight_width = self.order
        self.weights = self._validate_weights(weights, self._weight_width)
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        """Width of the model. This is 1 for 1D gradients."""
        return 1

    @property
    def nvectors(self):
        """Number of vectors required to be input to create the model."""
        return 1

    @property
    def arg_names(self):
        """Argument names that must be input into the model."""
        return {self.x_name}

    @property
    def _mu_letter(self):
        """Letter representing the weights of this gradient model."""
        return "v"

    @property
    def _equation(self):
        """Returns a list of latex equations for each vector to describe the generation of the design matrix."""
        eqn = [
            f"{idx + 1 if idx != 0 else ''}w_{{{idx}}}\mathbf{{{self.latex_aliases[self.x_name]}}}^{{{idx}}}"
            for idx in range(self.order + 1)
        ]
        return eqn

    def design_matrix(self, **kwargs):
        """Build the derivative of a 1D Polynomial in `self.x_name`.

        You must pass the keyword argument `self.x_name` to use this function.

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
    """lamatrix.Model object to model sinusoidal trends."""

    def __init__(
        self,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
        posteriors=None,
    ) -> Model:
        """
        Initialize a Sinusoid model.

        Parameters
        ----------
        x_name : str, optional
            The name of the independent variable (default is "x").
        nterms : int, optional
            The number of terms in the sinusoid.
        priors : optional
            Prior distributions for model weights (default is None, i.e. no priors).
        posteriors : optional
            Posterior distributions for model parameters (default is None, i.e. no posteriors).
            Posterior keyword is provided so that models can be loaded.

        Raises
        ------
        ValueError
            If `nterms` is less than 1.
        ValueError
            If `nterms` is not an integer.

        Examples
        --------
        Create a sinusoid model with nterms=3:

        >>> model = Sinusoid(x_name="phi", nterms=3)
        """
        if nterms < 1:
            raise ValueError("Must have nterms >= 1.")
        if not np.issubdtype(type(nterms), np.integer):
            raise ValueError("`nterms` must be an integer.")

        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        """Width of the model. This is equivalent to the `nterms` * 2 for Sinusoid models."""
        return self.nterms * 2

    @property
    def nvectors(self):
        """Number of vectors required to be input to create the model."""
        return 1

    @property
    def arg_names(self):
        """Argument names that must be input into the model."""
        return {self.x_name}

    @property
    def _equation(self):
        """Returns a list of latex equations for each vector to describe the generation of the design matrix."""

        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                [
                    f"\sin({frq(idx)}\\mathbf{{{self.latex_aliases[self.x_name]}}})",
                    f"\cos({frq(idx)}\\mathbf{{{self.latex_aliases[self.x_name]}}})",
                ]
                for idx in range(self.nterms)
            ]
        )

    def design_matrix(self, **kwargs):
        """Build a 1D Sinusoid in `self.x_name`.

        You must pass the keyword argument `self.x_name` to use this function.

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
        """Converts this model to the gradient of that model, assigning the posteriors of the fit as weights.

        If the posteriors are None, will assign the priors as weights.

        Parameters
        ----------
        weights: np.ndarray
            The weights applied to the Sinusoid model.
            These update the commponents for the design matrix of the gradient.
        priors: DistributionsContainer
            Priors to apply to the gradient model.

        Returns
        -------
        model: lamatrix.models.simple.dSinusoid
            A model for the gradient of a sinusoid

        Example
        -------

        To obtain the gradient of a fit sinusoid model

        >>> model = Sinusoid(x_name="phi", nterms=3)
        >>> model.fit(phi=phi, data=data, errors=errors)
        >>> dmodel = model.to_gradient()
        """
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        else:
            raise ValueError(
                "To convert to a gradient, you must provide either posteriors from fitting data or priors."
            )

        return dSinusoid(
            weights=weights,
            x_name=self.x_name,
            nterms=self.nterms,
            priors=priors,
        )


class dSinusoid(MathMixins, LatexMixins, IOMixins, Model):
    """lamatrix.Model object to model the gradient of sinusoid trends."""

    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        nterms: int = 1,
        priors=None,
        posteriors=None,
    ) -> Model:
        """
        Initialize a dSinusoid model.

        Parameters
        ----------
        weights: npt.NDArray,
            The input weights to be applied to the input model.
        x_name : str, optional
            The name of the independent variable (default is "x").
        nterms : int, optional
            The number of terms in the sinusoid.
        priors : optional
            Prior distributions for model weights (default is None, i.e. no priors).
        posteriors : optional
            Posterior distributions for model parameters (default is None, i.e. no posteriors).
            Posterior keyword is provided so that models can be loaded.

        Raises
        ------
        ValueError
            If `nterms` is less than 1.
        ValueError
            If `nterms` is not an integer.
        """
        if nterms < 1:
            raise ValueError("Must have `nterms` >= 1.")
        if not np.issubdtype(type(nterms), np.integer):
            raise ValueError("`nterms` must be an integer.")

        self.x_name = x_name
        self._validate_arg_names()
        self.nterms = nterms
        self._weight_width = self.nterms * 2
        self.weights = self._validate_weights(weights, self._weight_width)
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        """Width of the model. This is 1 for 1D gradients."""
        return 1

    @property
    def nvectors(self):
        """Number of vectors required to be input to create the model."""
        return 1

    @property
    def arg_names(self):
        """Argument names that must be input into the model."""
        return {self.x_name}

    @property
    def _mu_letter(self):
        """Letter representing the weights of this gradient model."""
        return "v"

    @property
    def _equation(self):
        """Provides the list of equation components without the weights included, to be used by `self.equation`."""

        def frq(term):
            return f"{term + 1}" if term > 0 else ""

        return np.hstack(
            [
                [
                    f"w_{{{idx * 2}}}\cos({frq(idx)}\\mathbf{{{self.latex_aliases[self.x_name]}}})",
                    f"w_{{{idx * 2 + 1}}}(-\sin({frq(idx)}\\mathbf{{{self.latex_aliases[self.x_name]}}}))",
                ]
                for idx in range(self.nterms)
            ],
        )

    def design_matrix(self, **kwargs):
        """Build a gradient of a 1D sinusoid in `self.x_name`.

        You must pass the keyword argument `self.x_name` to use this function.

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


class Constant(MathMixins, LatexMixins, IOMixins, Model):
    """lamatrix.Model object to model constants or offsets. This model has no input variable."""

    def __init__(self, priors=None, posteriors=None):
        """
        Initialize a Constant model.

        Note this model has no variable.

        Parameters
        ----------
        priors : optional
            Prior distributions for model weights (default is None, i.e. no priors).
        posteriors : optional
            Posterior distributions for model parameters (default is None, i.e. no posteriors).
            Posterior keyword is provided so that models can be loaded.

        Examples
        --------
        Create a constant model

        >>> model = Constant()
        """
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        """Width of the model. This is always one for a constant model."""
        return 1

    @property
    def nvectors(self):
        """Number of vectors required to be input to create the model.
        This is zero for a constant model."""
        return 0

    @property
    def arg_names(self):
        """Argument names that must be input into the model.
        This is an empty dictionary for a constant model."""
        return {}

    @property
    def _equation(self):
        """Returns a list of latex equations for each vector to describe the generation of the design matrix."""
        return [""]

    def design_matrix(self, **kwargs):
        """Build a design matrix consisting of a single column of ones.

        You must pass a keyword argument to this design matrix to detect the required shape.

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
            f"\mathbb{{I}}_{{[{{{bounds[idx]}}}, {{{bounds[idx + 1]}}}]}}(\mathbf{{{self.latex_aliases[self.x_name]}}})"
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
        eqn = [
            f"\mathbf{{{self.latex_aliases[self.x_name]}}}_{{{idx}}}"
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
        if not x.shape[-1] == self.width:
            raise ValueError(
                f"Must pass a vector for {self.x_name} that has width {self.width}."
            )
        return x
