# """Generator objects for different types of models"""
from typing import List
import numpy as np
from ..model import Model
from ..math import MathMixins

__all__ = [
    "Spline",
    "dSpline",
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


class Spline(MathMixins, SplineMixins, Model):
    def __init__(
        self,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        splineorder: int = 3,
        priors=None,
    ):
        if splineorder < 1:
            raise ValueError("Must have splineorder >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.splineorder = splineorder
        self.knots = knots
        super().__init__(priors=priors)

    @property
    def width(self):
        return len(self.knots) - self.splineorder - 1

    @property
    def nvectors(self):
        return 1

    @property
    def arg_names(self):
        return {self.x_name}

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
            X[i] = self.bspline_basis(k=self.splineorder, i=i, t=self.knots, x=x)
        return X.transpose(shape_a)

    def to_gradient(self, priors=None):
        weights = [
            fit if fit is not None else prior
            for fit, prior in zip(self.best_fit.mean, self.best_fit.std)
        ]
        return dSpline(
            weights=weights,
            x_name=self.x_name,
            knots=self.knots,
            splineorder=self.splineorder,
            priors=priors,
        )


class dSpline(MathMixins, SplineMixins, Model):
    def __init__(
        self,
        weights: List,
        x_name: str = "x",
        knots: np.ndarray = np.arange(1, 1, 0.3),
        splineorder: int = 3,
        priors=None,
    ):
        if splineorder < 1:
            raise ValueError("Must have splineorder >= 1.")
        self.x_name = x_name
        self._validate_arg_names()
        self.splineorder = splineorder
        self.knots = knots
        self._weight_width = len(self.knots) - self.splineorder - 1
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
        if not self.arg_names.issubset(set(kwargs.keys())):
            raise ValueError(f"Expected {self.arg_names} to be passed.")
        x = kwargs.get(self.x_name)
        shape_a = [*np.arange(1, x.ndim + 1).astype(int), 0]
        # Set up the least squares problem
        X = np.zeros((len(self.weights), *x.shape))
        for i in range(len(self.weights)):
            X[i] = self.bspline_basis_derivative(
                k=self.splineorder, i=i, t=self.knots, x=x
            )
        return np.expand_dims(X.transpose(shape_a).dot(self.weights), x.ndim)


# class Spline1DGenerator(MathMixins, SplineMixins, Generator):
#     def __init__(
#         self,
#         knots: np.ndarray,
#         x_name: str = "x",
#         splineorder: int = 3,
#         prior_mu=None,
#         prior_sigma=None,
#         offset_prior=None,
#         data_shape=None,
#     ):
#         # Check if knots are padded
#         if not (len(np.unique(knots[:splineorder])) == 1) & (
#             len(np.unique(knots[-splineorder:])) == 1
#         ):
#             knots = np.concatenate(
#                 ([knots[0]] * (splineorder - 1), knots, [knots[-1]] * (splineorder - 1))
#             )
#         self.knots = knots
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.splineorder = splineorder
#         self.data_shape = data_shape
#         self._validate_priors(prior_mu, prior_sigma, offset_prior=offset_prior)
#         self.fit_mu = None
#         self.fit_sigma = None

#     @property
#     def width(self):
#         return len(self.knots) - self.splineorder - 1 + 1

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
#             "knots",
#             "splineorder",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         """Build a 1D spline in x

#         Parameters
#         ----------
#         {} : np.ndarray
#             Vector to create spline of

#         Returns
#         -------
#         X : np.ndarray
#             Design matrix with shape (len(x), self.nvectors)
#         """
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()

#         # Set up the least squares problem
#         X = np.zeros((len(x), self.width - 1))
#         for i in range(self.width - 1):
#             for j, xi in enumerate(x):
#                 X[j, i] = self.bspline_basis(
#                     k=self.splineorder, i=i, t=self.knots, x=xi
#                 )
#         return np.hstack([np.ones((len(x), 1)), X])

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def offset(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def _equation(self):
#         return [
#             f"\\mathbf{{{self.x_name}}}^0",
#             *[
#                 f"N_{{{idx},k}}(\\mathbf{{{self.x_name}}})"
#                 for idx in np.arange(1, self.width)
#             ],
#         ]

#     def to_latex(self):
#         eqn1 = f"\\begin{{equation}}\\label{{eq:spline}}f(\\mathbf{{{self.x_name}}}) = \sum_{{i=0}}^{{n-1}} w_i N_{{i,k}}(\\mathbf{{{self.x_name}}}) \\]\\end{{equation}}"
#         eqn2 = f"\\[N_{{i,k}}(\\mathbf{{{self.x_name}}}) = \\frac{{\\mathbf{{{self.x_name}}} - t_i}}{{t_{{i+k-1}} - t_i}} N_{{i,k-1}}(\\mathbf{{{self.x_name}}}) + \\frac{{t_{{i+k}} - \\mathbf{{{self.x_name}}}}}{{t_{{i+k}} - t_{{i+1}}}} N_{{i+1,k-1}}(\\mathbf{{{self.x_name}}})\\]"
#         eqn3 = f"""\\[N_{{i,1}}(\\mathbf{{{self.x_name}}}) =
#         \\begin{{cases}}
#         1 & \\text{{if }} t_i \leq \\mathbf{{{self.x_name}}} < t_{{i+1}} \\\\
#         0 & \\text{{otherwise}} \\\\
#         \\end{{cases}}
#         \\]"""
#         eqn4 = "$t = [" + " , ".join([f"{k}" for k in self.knots]) + "]$"

#         return "\n".join([eqn1, eqn2, eqn3, eqn4, self._to_latex_table()])

#     @property
#     def gradient(self):
#         return dSpline1DGenerator(
#             weights=self.mu,
#             knots=self.knots,
#             splineorder=self.splineorder,
#             data_shape=self.data_shape,
#             x_name=self.x_name,
#         )


# class dSpline1DGenerator(MathMixins, SplineMixins, Generator):
#     def __init__(
#         self,
#         weights: np.ndarray,
#         knots: np.ndarray,
#         x_name: str = "x",
#         splineorder: int = 3,
#         offset_prior=None,
#         prior_mu=None,
#         prior_sigma=None,
#         data_shape=None,
#     ):
#         # Check if knots are padded
#         if not (len(np.unique(knots[:splineorder])) == 1) & (
#             len(np.unique(knots[-splineorder:])) == 1
#         ):
#             knots = np.concatenate(
#                 ([knots[0]] * (splineorder - 1), knots, [knots[-1]] * (splineorder - 1))
#             )
#         self.knots = knots
#         self.weights = weights
#         self.x_name = x_name
#         self._validate_arg_names()
#         self.splineorder = splineorder
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
#             "weights",
#             "knots",
#             "splineorder",
#             "prior_mu",
#             "prior_sigma",
#             "offset_prior",
#             "data_shape",
#         ]

#     def design_matrix(self, *args, **kwargs):
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name).ravel()
#         n = len(self.weights) - 1
#         y_deriv = np.zeros_like(x)
#         for i in range(n):
#             for j, xi in enumerate(x):
#                 y_deriv[j] += self.weights[i + 1] * self.bspline_basis_derivative(
#                     k=self.splineorder, i=i, t=self.knots, x=xi
#                 )
#         return np.hstack([np.ones((len(x), 1)), y_deriv[:, None]])

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def offset(self):
#         return self.mu[0], self.sigma[0]

#     @property
#     def _equation(self):
#         raise NotImplementedError

#     @property
#     def shift_x(self):
#         return self.mu[1], self.sigma[1]

#     @property
#     def table_properties(self):
#         return [
#             (
#                 "w_0",
#                 (self.mu[0], self.sigma[0]),
#                 (self.prior_mu[0], self.prior_sigma[0]),
#             ),
#             ("s_x", self.shift_x, (self.prior_mu[1], self.prior_sigma[1])),
#         ]

#     @property
#     def _equation(self):
#         return [
#             f"\\mathbf{{{self.x_name}}}^0",
#             f"\\frac{{\\partial \\left( \\sum_{{i=1}}^{{{len(self.knots) - self.splineorder}}} w_{{i}} N_{{i,{self.splineorder}}}(\\mathbf{{{self.x_name}}})\\right)}}{{\\partial \mathbf{{{self.x_name}}}}}",
#         ]

#     @property
#     def _mu_letter(self):
#         return "v"
