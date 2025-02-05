# """Generator objects for different types of models"""
from typing import List

import numpy as np
from scipy import sparse

from ..io import IOMixins, LatexMixins
from ..math import MathMixins
from ..model import Equation, Model

__all__ = [
    "Spline",
    "dSpline",
    "SparseSpline",
    "dSparseSpline",
]


class SplineMixins:
    def bspline_basis(self, k, i, t, x):
        """
        Calculate B-spline basis function value of k-th order.

        k : order of the basis function (degree + 1)
        i : index of the basis function
        t : array of knot positions
        x : position where the basis function is evaluated
        """
        if k == 1:
            # return 1.0 if t[i] <= x < t[i + 1] else 0.0
            return ((x < t[i + 1]) & (t[i] <= x)).astype(float)
        else:
            div0 = t[i + k - 1] - t[i]
            div1 = t[i + k] - t[i + 1]
            term0 = (
                0
                if div0 == 0
                else ((x - t[i]) / div0) * self.bspline_basis(k - 1, i, t, x)
            )
            term1 = (
                0
                if div1 == 0
                else ((t[i + k] - x) / div1) * self.bspline_basis(k - 1, i + 1, t, x)
            )
            return term0 + term1

    def bspline_basis_derivative(self, k, i, t, x):
        """
        Calculate the derivative of B-spline basis function of k-th order.

        k : order of the basis function (degree + 1)
        i : index of the basis function
        t : array of knot positions
        x : position where the derivative is evaluated
        """
        if k > 1:
            term0 = (
                (k - 1) / (t[i + k - 1] - t[i]) * self.bspline_basis(k - 1, i, t, x)
                if t[i + k - 1] != t[i]
                else 0
            )
            term1 = (
                (k - 1) / (t[i + k] - t[i + 1]) * self.bspline_basis(k - 1, i + 1, t, x)
                if t[i + k] != t[i + 1]
                else 0
            )
            return term0 - term1
        else:
            return 0  # The derivative of a constant (k=1) function is 0


class Spline(MathMixins, SplineMixins, LatexMixins, IOMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        if order < 1:
            raise ValueError("Must have order >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        self.knots = knots
        super().__init__(priors=priors, posteriors=posteriors)

    @property
    def width(self):
        return len(self.knots) - self.order

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

    @property
    def _equation(self):
        return [
            f"N_{{{idx}, {{{self.order}}}}}(\\mathbf{{{self.latex_aliases[self.x_name]}}})"
            for idx in np.arange(1, self.width)
        ]

    def design_matrix(self, **kwargs):
        """Build a 1D spline in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create spline of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        X = np.zeros((self.width, *x.shape))
        for i in range(self.width):
            X[i] = self.bspline_basis(k=self.order, i=i, t=self.knots, x=x)
        return X.transpose(shape_a)

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        return dSpline(
            weights=weights,
            x_name=self.x_name,
            knots=self.knots,
            order=self.order,
            priors=priors,
        )

    @property
    def spline_equation(self):
        eqn1 = f"\\[f(\\mathbf{{{self.latex_aliases[self.x_name]}}}) = \sum_{{i=0}}^{{n-1}} w_i N_{{i,k}}(\\mathbf{{{self.latex_aliases[self.x_name]}}}) \\]"
        eqn2 = (
            f"\\[N_{{i,k}}(\\mathbf{{{self.latex_aliases[self.x_name]}}})"
            f"= \\frac{{\\mathbf{{{self.latex_aliases[self.x_name]}}} - t_i}}{{t_{{i+k-1}} - t_i}} N_{{i,k-1}}(\\mathbf{{{self.latex_aliases[self.x_name]}}}) "
            f"+ \\frac{{t_{{i+k}} - \\mathbf{{{self.latex_aliases[self.x_name]}}}}}{{t_{{i+k}} - t_{{i+1}}}}"
            f" N_{{i+1,k-1}}(\\mathbf{{{self.latex_aliases[self.x_name]}}})\\]"
        )
        eqn3 = f"""\\[N_{{i,1}}(\\mathbf{{{self.latex_aliases[self.x_name]}}}) =
        \\begin{{cases}}
        1 & \\text{{if }} t_i \leq \\mathbf{{{self.latex_aliases[self.x_name]}}} < t_{{i+1}} \\\\
        0 & \\text{{otherwise}} \\\\
        \\end{{cases}}
        \\]"""
        eqn4 = "\\[t = [" + " , ".join([f"{k}" for k in self.knots]) + "]\\]"
        eqn5 = f"\\[ k = {self.order} \\]"
        return Equation("\n".join([eqn1, eqn2, eqn3, eqn4, eqn5]))

    def to_latex(self):
        return "\n".join([self.spline_equation, self._to_latex_table()])


class SparseSpline(Spline):
    def __init__(
        self,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        super().__init__(
            x_name=x_name,
            order=order,
            knots=knots,
            priors=priors,
            posteriors=posteriors,
        )

    def design_matrix(self, **kwargs):
        """Build a 1D spline in x

        Parameters
        ----------
        {} : np.ndarray
            Vector to create spline of

        Returns
        -------
        X : np.ndarray
            Design matrix with shape (len(x), self.nvectors)
        """
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        if not x.ndim == 1:
            raise ValueError(
                f"Can only fit sparse matrices in one dimension, {self.x_name} has shape {x.shape}."
            )
        k = x.nonzero()[0]
        X = sparse.lil_matrix((self.width, x.shape[0]))
        for i in range(self.width):
            X[i, k] = self.bspline_basis(k=self.order, i=i, t=self.knots, x=x.data)
        return X.T.tocsr()

    def to_gradient(self, weights=None, priors=None):
        if weights is None:
            weights = (
                self.posteriors.mean
                if self.posteriors is not None
                else self.priors.mean
            )
        return dSparseSpline(
            weights=weights,
            x_name=self.x_name,
            knots=self.knots,
            order=self.order,
            priors=priors,
        )


class dSpline(MathMixins, SplineMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        if order < 1:
            raise ValueError("Must have order >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.order = order
        self.knots = knots
        self._weight_width = len(self.knots) - self.order
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
    def _equation(self):
        return [
            f"\\frac{{\\partial \\left( \\sum_{{i=0}}^{{{len(self.knots) - self.order - 2}}}"
            f" w_{{i}} N_{{i,{self.order}}}(\\mathbf{{{self.latex_aliases[self.x_name]}}})\\right)}}"
            f"{{\\partial \mathbf{{{self.latex_aliases[self.x_name]}}}}}",
        ]

    @property
    def _mu_letter(self):
        return "v"

    def design_matrix(self, **kwargs):
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        # Set up the least squares problem
        X = np.zeros((len(self.weights), *x.shape))
        for i in range(len(self.weights)):
            X[i] = self.bspline_basis_derivative(k=self.order, i=i, t=self.knots, x=x)
        return np.expand_dims(X.transpose(shape_a).dot(self.weights), x.ndim)


class dSparseSpline(dSpline):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        order: int = 3,
        priors=None,
        posteriors=None,
    ):
        super().__init__(
            weights=weights,
            x_name=x_name,
            order=order,
            knots=knots,
            priors=priors,
            posteriors=posteriors,
        )

    def design_matrix(self, **kwargs):
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        if not x.shape[1] == 1:
            raise ValueError(
                f"Can only fit sparse matrices with shape (n, 1), {self.x_name} has shape {x.shape}."
            )
        k = x.nonzero()[0]
        X = sparse.lil_matrix((len(self.weights), x.shape[0]))
        for i in range(len(self.weights)):
            X[i, k] = self.bspline_basis_derivative(
                k=self.order, i=i, t=self.knots, x=x.data
            )
        return sparse.csr_matrix(X.T.dot(self.weights)).T
