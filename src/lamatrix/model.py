"""Abstract base class for a Model object"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy import sparse

from .distributions import Distribution, DistributionsContainer

__all__ = ["Model"]


class Equation(str):
    """Class for holding equations so we can have an HTML repr in Jupyter."""

    def __new__(cls, equation: str):
        """Ensure Equation behaves like a string while allowing custom behavior."""
        return super().__new__(cls, equation)

    def _repr_html_(self):
        """Jupyter-specific representation: renders equation using MathJax."""
        return f"<div>{self}</div>"

    def __repr__(self):
        """Fallback representation for debugging."""
        return super().__repr__()


def _sparse_ones_like(matrix):
    """From an input matrix, creates an object of the same type of matrix with all the non zero values as 1."""
    if sparse.isspmatrix_csr(matrix) or sparse.isspmatrix_csc(matrix):
        # For CSR and CSC, we can use the same data, indices, and indptr but replace data with ones
        return matrix.__class__(
            (np.ones_like(matrix.data), matrix.indices, matrix.indptr),
            shape=matrix.shape,
        )

    elif sparse.isspmatrix_lil(matrix):
        # For LIL, replace each row's data with ones
        ones_matrix = matrix.copy()
        ones_matrix.data = [[1] * len(row) for row in ones_matrix.data]
        return ones_matrix

    elif sparse.isspmatrix_coo(matrix):
        # For COO, replace the data with ones
        return sparse.coo_matrix(
            (np.ones_like(matrix.data), (matrix.row, matrix.col)), shape=matrix.shape
        )

    elif sparse.isspmatrix_dok(matrix):
        # For DOK, we can create a new matrix and fill the data dictionary with ones
        ones_matrix = matrix.copy()
        for key in ones_matrix.keys():
            ones_matrix[key] = 1
        return ones_matrix

    elif sparse.isspmatrix_dia(matrix):
        # For DIA (diagonal) matrices, replace the data with ones
        return sparse.dia_matrix(
            (np.ones_like(matrix.data), matrix.offsets), shape=matrix.shape
        )

    else:
        raise TypeError(f"Unsupported sparse matrix type: {type(matrix)}")


class Model(ABC):
    def _validate_priors(self, priors, width=None):
        if width is None:
            width = self.width
        if priors is None:
            return DistributionsContainer.from_number(width)
        elif isinstance(priors, (tuple, Distribution)):
            return DistributionsContainer([priors])
        elif isinstance(priors, (list, np.ndarray)):
            return DistributionsContainer(priors)
        elif isinstance(priors, DistributionsContainer):
            return priors
        else:
            raise ValueError("Could not parse input `priors`.")

    def __init__(
        self,
        priors: List[Tuple] = None,
        posteriors: List[Tuple] = None,
    ):
        # prior (always normal, and always specfied by mean and standard deviation)
        # fit_distributions (always normal, and always specfied by mean and standard deviation)
        self.priors = self._validate_priors(priors)
        if not len(self.priors) == self.width:
            raise ValueError(
                "distributions must have the number of elements as the design matrix."
            )

        self.posteriors = posteriors

    @property
    @abstractmethod
    def width(self):
        """Returns the width of the design matrix once built."""
        pass

    @property
    @abstractmethod
    def nvectors(self):
        """Returns the number of vectors required to build the object."""
        pass

    @property
    def _initialization_attributes(self):
        return [
            "x_name",
        ]

    @property
    def _mu_letter(self):
        """Letter to represent the weight in equation representation."""
        return "w"

    @property
    @abstractmethod
    def _equation(self):
        """Returns a list of latex equations for each vector to describe the generation of the design matrix."""
        pass

    @property
    def equation(self):
        func_signature = ", ".join([f"\mathbf{{{a}}}" for a in self.arg_names])
        eqn = (
            f"\[f({func_signature}) = "
            + " + ".join(
                [
                    f"{self._mu_letter}_{{{coeff}}} {e}"
                    for coeff, e in enumerate(self._equation)
                ]
            )
            + "\]"
        )
        return Equation(eqn)

    def __repr__(self):
        return (
            f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}]"
        )

    def copy(self):
        return deepcopy(self)

    def _validate_arg_names(self):
        for arg in self.arg_names:
            if not isinstance(arg, str):
                raise ValueError("Argument names must be strings.")

    # def _validate_distribution(self, distribution: Tuple):
    #     """Checks that a distribution is a valid input with format (mean, std)."""
    #     # Must be a tuple
    #     if not isinstance(distribution, tuple):
    #         raise ValueError("distribution must be a tuple of format (mean, std)")
    #     # Must be float or int
    #     for value in distribution:
    #         if not isinstance(value, (float, int, np.integer, np.floating)):
    #             raise ValueError("Values in distribution must be numeric.")
    #     # Standard deviation must be positive
    #     if np.sign(distribution[1]) == -1:
    #         raise ValueError("Standard deviation must be positive.")
    #     return

    # def _validate_distributions(self, distributions: List[Tuple]):
    #     """Checks that a list of distributions is a valid"""
    #     if not len(distributions) == self.width:
    #         raise ValueError(
    #             "distributions must have the number of elements as the design matrix."
    #         )
    #     return

    def _validate_weights(self, weights, weights_width):
        if not isinstance(weights, (list, np.ndarray)):
            raise ValueError(
                f"`weights` must be a list of numeric values with length {weights_width}."
            )
        if not isinstance(weights[0], (float, int, np.integer, np.number)):
            raise ValueError(
                f"`weights` must be a list of numeric values with length {weights_width}."
            )
        if not len(weights) == weights_width:
            raise ValueError(
                f"`weights` must be a list of numeric values with length {weights_width}."
            )
        return np.asarray(weights)

    # def set_prior(self, index: int, distribution: Tuple) -> None:
    #     """Sets a single prior."""
    #     self._validate_distribution(distribution=distribution)
    #     self.prior_distributions[index] = distribution
    #     return

    # def set_priors(self, distributions: List[Tuple]) -> None:
    #     """sets the full list of priors"""
    #     self._validate_distributions(distributions=distributions)
    #     self.prior_distributions = distributions
    #     return

    # def freeze_element(self, index: int):
    #     """Freezes an element of the design matrix by setting prior_sigma to zero."""
    #     self.set_prior(index, (self.prior_distributions[index][0], 1e-10))

    # @property
    # def prior_mean(self):
    #     return np.asarray(
    #         [distribution[0] for distribution in self.prior_distributions]
    #     )

    # @property
    # def prior_std(self):
    #     return np.asarray(
    #         [distribution[1] for distribution in self.prior_distributions]
    #     )

    # @property
    # def fit_mean(self):
    #     return np.asarray([distribution[0] for distribution in self.fit_distributions])

    # @property
    # def fit_std(self):
    #     return np.asarray([distribution[1] for distribution in self.fit_distributions])

    @abstractmethod
    def design_matrix(self):
        """Returns a design matrix, given inputs listed in self.arg_names."""
        pass

    @property
    @abstractmethod
    def arg_names(self):
        """Returns a set of the user defined strings for all the arguments that the design matrix requires."""
        pass

    def fit(  # noqa C901
        self,
        data: npt.NDArray,
        errors: npt.NDArray = None,
        mask: npt.NDArray = None,
        **kwargs,
    ):
        """Fit the design matrix.

        Parameters
        ----------
        data: np.ndarray
            Input data to fit
        errors: np.ndarray, optional
            Errors on the input data
        mask: np.ndarray, optional
            Mask to apply when fitting. Values where mask is False will not be used during the fit.

        Returns
        -------
        fit_distributions: List of Tuples
            The best fit distributions

        """
        if not isinstance(self.priors, DistributionsContainer):
            if isinstance(self.priors, tuple):
                self.priors = DistributionsContainer(self.priors)
            elif isinstance(self.priors, list):
                self.priors = DistributionsContainer(
                    [Distribution(d) for d in self.priors]
                )
            else:
                raise ValueError("Can not parse priors.")

        for attr in ["error", "err"]:
            if attr in kwargs.keys():
                raise ValueError(f"Pass `errors` not `{attr}`.")

        dense_data = not sparse.issparse(data)

        if mask is None:
            if dense_data:
                mask = np.ones(data.shape, bool)
            else:
                mask = np.ones(data.shape[0], bool)
        if dense_data:
            if not mask.shape == data.shape:
                raise ValueError(
                    f"Must pass vector for variable `mask` with shape {data.shape}."
                )
        else:
            if not mask.shape[0] == data.shape[0]:
                raise ValueError(
                    f"Must pass vector for variable `mask` with shape ({data.shape[0]},)."
                )

        if errors is None:
            if dense_data:
                errors = np.ones_like(data)
            else:
                errors = _sparse_ones_like(data)
        if not errors.shape == data.shape:
            raise ValueError(
                f"Must pass vector for variable `errors` with shape {data.shape}."
            )

        for key, item in kwargs.items():
            dense_vector = not sparse.issparse(item)
            if dense_vector:
                if dense_data:
                    if not item.shape == data.shape:
                        if not item.shape[:-1] == data.shape:
                            raise ValueError(
                                f"Must pass vector for variable `{key}` with shape {data.shape}."
                            )
                else:
                    if not (item.shape[0] == data.shape[0]) & (item.ndim == 1):
                        raise ValueError(
                            f"Must pass vector for variable `{key}` with shape ({data.shape[0]}, 1)."
                        )
            else:
                if dense_data:
                    if not (item.shape[0] == data.shape[0]) & (data.ndim == 1):
                        raise ValueError(
                            f"Must pass vector for variable `{key}` with shape ({data.shape[0]}, 1)."
                        )
                else:
                    if not item.shape == data.shape:
                        raise ValueError(
                            f"Must pass vector for variable `{key}` with shape {data.shape}."
                        )

        X = self.design_matrix(**kwargs)
        dense_designmatrix = not sparse.issparse(X)
        if (not dense_data) & dense_designmatrix:
            raise ValueError("Must fit sparse data with a sparse design matrix.")
        if not dense_designmatrix:
            if dense_data:
                sigma_w_inv = X[mask].T.dot(
                    X[mask].multiply(1 / errors[mask][:, None] ** 2)
                ) + sparse.diags(1 / self.priors.std**2)
                y = sparse.csr_matrix(data[mask] / errors[mask] ** 2).T
            else:
                sigma_w_inv = X[mask].T.dot(
                    X[mask].multiply(errors[mask].power(2).power(-1))
                ) + sparse.diags(1 / self.priors.std**2)
                y = data[mask].multiply(errors[mask].power(2).power(-1))

            self.cov = sparse.linalg.inv(sigma_w_inv)
            B = (
                X[mask].T.dot(y)
                + sparse.csr_matrix(
                    np.nan_to_num(self.priors.mean / self.priors.std**2)
                ).T
            )
            fit_mean = sparse.linalg.spsolve(sigma_w_inv, B)
            fit_std = self.cov.diagonal() ** 0.5

        else:
            if sparse.issparse(data):
                raise ValueError(
                    "Can not fit sparse data, if design matrix is not sparse."
                )
            sigma_w_inv = X[mask].T.dot(X[mask] / errors[mask][:, None] ** 2) + np.diag(
                1 / self.priors.std**2
            )
            self.cov = np.linalg.inv(sigma_w_inv)
            B = X[mask].T.dot(data[mask] / errors[mask] ** 2) + np.nan_to_num(
                self.priors.mean / self.priors.std**2
            )
            fit_mean = np.linalg.solve(sigma_w_inv, B)
            fit_std = self.cov.diagonal() ** 0.5
        self.posteriors = DistributionsContainer(
            [Distribution(m, s) for m, s in zip(fit_mean, fit_std)]
        )
        return

    def evaluate(self, **kwargs):
        X = self.design_matrix(**kwargs)
        return X.dot(self.posteriors.mean)

    def sample(self, **kwargs):
        X = self.design_matrix(**kwargs)
        return X.dot(self.posteriors.sample())

    def __call__(self, *args, **kwargs):
        return self.design_matrix(*args, **kwargs)
