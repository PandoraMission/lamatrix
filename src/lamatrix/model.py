"""Abstract base class for a Model object"""

import json
import math
from abc import ABC, abstractmethod
from copy import deepcopy

from typing import List, Tuple
import numpy as np
import numpy.typing as npt

from . import _META_DATA
from .distributions import DistributionsContainer, Distribution

__all__ = ["Model"]


class Model(ABC):
    def __init__(
        self,
        priors: List[Tuple] = None,
    ):
        # prior (always normal, and always specfied by mean and standard deviation)
        # fit_distributions (always normal, and always specfied by mean and standard deviation)
        if priors is None:
            self.priors = DistributionsContainer.from_number(self.width)
        elif isinstance(priors, (list, np.ndarray)):
            self.priors = DistributionsContainer(priors)
        elif isinstance(priors, DistributionsContainer):
            self.priors = priors
        else:
            raise ValueError("Could not parse input `priors`.")
        if not len(self.priors) == self.width:
            raise ValueError(
                "distributions must have the number of elements as the design matrix."
            )

        self.best_fit = DistributionsContainer.from_number(self.width)
        
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
            raise ValueError(f"`weights` must be a list of numeric values with length {weights_width}.")
        if not isinstance(weights[0], (float, int, np.integer, np.number)):
            raise ValueError(f"`weights` must be a list of numeric values with length {weights_width}.")
        if not len(weights) == weights_width:
            raise ValueError(f"`weights` must be a list of numeric values with length {weights_width}.")
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

    def fit(self, data: npt.NDArray, errors: npt.NDArray = None, mask:npt.NDArray = None, **kwargs):
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
        for attr in ["error", "err"]:
            if attr in kwargs.keys():
                raise ValueError(f"Pass `errors` not `{attr}`.")
        if mask is None:
            mask = np.ones(data.shape, bool)
        if errors is None:
            errors = np.ones_like(data)
        for key, item in kwargs.items():
            if not item.shape == data.shape:
                raise ValueError(f"Must pass vector for variable `{key}` with shape {data.shape}.")
        if not errors.shape == data.shape:
            raise ValueError(f"Must pass vector for variable `errors` with shape {data.shape}.")
        if not mask.shape == data.shape:
            raise ValueError(f"Must pass vector for variable `mask` with shape {data.shape}.")

        X = self.design_matrix(**kwargs)
        sigma_w_inv = X[mask].T.dot(X[mask] / errors[mask][:, None] ** 2) + np.diag(
            1 / self.priors.std**2
        )
        self.cov = np.linalg.inv(sigma_w_inv)
        B = X[mask].T.dot(data[mask] / errors[mask] ** 2) + np.nan_to_num(
            self.priors.mean / self.priors.std**2
        )
        fit_mean = np.linalg.solve(sigma_w_inv, B)
        fit_std = self.cov.diagonal() ** 0.5
        self.best_fit = DistributionsContainer([Distribution(m, s) for m, s in zip(fit_mean, fit_std)])
        return

    def evaluate(self, **kwargs):
        X = self.design_matrix(**kwargs)
        return X.dot(self.best_fit.mean)

    def __call__(self, *args, **kwargs):
        return self.design_matrix(*args, **kwargs)


# class Generator(ABC):
#     def _validate_arg_names(self):
#         for arg in self.arg_names:
#             if not isinstance(arg, str):
#                 raise ValueError("Argument names must be strings.")

#     def _validate_priors(self, prior_mu, prior_sigma, offset_prior=None):
#         if prior_mu is None:
#             self.prior_mu = np.zeros(self.width)
#         else:
#             if isinstance(prior_mu, (float, int)):
#                 self.prior_mu = np.ones(self.width) * prior_mu
#             elif isinstance(prior_mu, (list, np.ndarray, tuple)):
#                 if len(prior_mu) == self.width:
#                     self.prior_mu = prior_mu
#             else:
#                 raise ValueError("Can not parse `prior_mu`.")

#         if prior_sigma is None:
#             self.prior_sigma = np.ones(self.width) * np.inf
#         else:
#             if isinstance(prior_sigma, (float, int)):
#                 self.prior_sigma = np.ones(self.width) * prior_sigma
#             elif isinstance(prior_sigma, (list, np.ndarray, tuple)):
#                 if len(prior_sigma) == self.width:
#                     self.prior_sigma = prior_sigma
#             else:
#                 raise ValueError("Can not parse `prior_sigma`.")

#         self.offset_prior = offset_prior
#         if self.offset_prior is not None:
#             if not hasattr(self.offset_prior, "__iter__"):
#                 raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")
#             if not len(self.offset_prior) == 2:
#                 raise AttributeError("Pass offset prior as a tuple with (mu, sigma)")

#             self.prior_mu[0] = self.offset_prior[0]
#             self.prior_sigma[0] = self.offset_prior[1]

#     def update_priors(self):
#         if self.fit_mu is None:
#             raise ValueError("Can not update priors before fitting.")
#         self.prior_mu = self.fit_mu.copy()
#         self.prior_sigma = self.fit_sigma.copy()
#         return

#     def _create_save_data(self):
#         def process(arg):
#             if arg is None:
#                 return None
#             elif isinstance(arg, (str, int, float, list, tuple)):
#                 if arg is np.inf:
#                     return "Infinity"
#                 return arg
#             elif isinstance(arg, np.ndarray):
#                 arg = arg.tolist()
#                 arg = [a if a != np.inf else "Infinity" for a in arg]
#                 return arg

#         results = {
#             attr: process(getattr(self, attr)) for attr in ["fit_mu", "fit_sigma"]
#         }
#         kwargs = {attr: process(getattr(self, attr)) for attr in self._INIT_ATTRS}
#         type_name = type(self).__name__

#         data_to_store = {
#             "object_type": type_name,
#             "initializing_kwargs": kwargs,
#             "fit_results": results,
#             "equation": self.equation,
#             "latex": self.to_latex(),
#         }
#         return data_to_store

#     def save(self, filename: str):
#         data_to_store = self._create_save_data()
#         data_to_store["metadata"] = _META_DATA()
#         if not filename.endswith(".json"):
#             filename = filename + ".json"

#         # Write to a JSON file
#         with open(filename, "w") as json_file:
#             json.dump(data_to_store, json_file, indent=4)

#     def copy(self):
#         return deepcopy(self)

#     def __repr__(self):
#         fit = "fit" if self.fit_mu is not None else ""
#         return f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}] {fit}"

#     # def __add__(self, other):
#     #     if isinstance(other, Generator):
#     #         return StackedGenerator(self, other)
#     #     else:
#     #         raise ValueError("Can only combine `Generator` objects.")

#     @staticmethod
#     def format_significant_figures(mean, error):
#         # Check for inf, -inf, or NaN
#         if (
#             math.isinf(mean)
#             or math.isinf(error)
#             or math.isnan(mean)
#             or math.isnan(error)
#         ):
#             # Handle these cases as you see fit, for example:
#             return "0", "\\infty"

#         # Find the first significant digit of the error
#         if error == 0:
#             sig_figures = 0
#         else:
#             sig_figures = np.max([0, -int(math.floor(math.log10(abs(error))))])

#         # Format mean and error to have the same number of decimal places
#         formatted_mean = f"{mean:.{sig_figures}f}"
#         formatted_error = f"{error:.{sig_figures}f}"

#         return formatted_mean, formatted_error

#     def _get_table_matter(self):
#         table_matter = []
#         for symbol, fit, prior in self.table_properties:
#             formatted_fit_mean, formatted_fit_error = self.format_significant_figures(
#                 *fit
#             )
#             if prior is not None:
#                 formatted_prior_mean, formatted_prior_error = (
#                     self.format_significant_figures(*prior)
#                 )
#             else:
#                 formatted_prior_mean = ""
#                 formatted_prior_error = ""
#             row = f"{symbol} & ${formatted_fit_mean} \\pm {formatted_fit_error}$  & ${formatted_prior_mean} \\pm {formatted_prior_error}$ \\\\\\hline\n"
#             table_matter.append(row)
#         return table_matter

#     def _to_latex_table(self):
#         latex_table = "\\begin{table}[h!]\n\\centering\n"
#         latex_table += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
#         latex_table += "Coefficient & Best Fit & Prior \\\\\\hline\n"
#         idx = 0
#         for tm in self._get_table_matter():
#             latex_table += tm.format(idx=idx)
#             idx += 1
#         latex_table += "\\end{tabular}\n\\end{table}"
#         return latex_table

#     def to_latex(self):
#         return "\n".join([self.equation, self._to_latex_table()])

#     def _fit(self, data, errors=None, mask=None, *args, **kwargs):
#         X = self.design_matrix(*args, **kwargs)
#         if np.prod(data.shape) != X.shape[0]:
#             raise ValueError(f"Data must have shape {X.shape[0]}")
#         if errors is None:
#             errors = np.ones_like(data)
#         if mask is None:
#             mask = np.ones(np.prod(data.shape), bool)
#         self.data_shape = data.shape
#         mask = mask.ravel()
#         sigma_w_inv = X[mask].T.dot(
#             X[mask] / errors.ravel()[mask, None] ** 2
#         ) + np.diag(1 / self.prior_sigma**2)
#         self.cov = np.linalg.inv(sigma_w_inv)
#         B = X[mask].T.dot(
#             data.ravel()[mask] / errors.ravel()[mask] ** 2
#         ) + np.nan_to_num(self.prior_mu / self.prior_sigma**2)
#         fit_mu = np.linalg.solve(sigma_w_inv, B)
#         fit_sigma = self.cov.diagonal() ** 0.5
#         return fit_mu, fit_sigma

#     @property
#     def mu(self):
#         return self.prior_mu if self.fit_mu is None else self.fit_mu

#     @property
#     def sigma(self):
#         return self.prior_sigma if self.fit_sigma is None else self.fit_sigma

#     def evaluate(self, *args, **kwargs):
#         X = self.design_matrix(*args, **kwargs)
#         if self.data_shape is not None:
#             if X.shape[0] == np.prod(self.data_shape):
#                 return X.dot(self.mu).reshape(self.data_shape)
#         return X.dot(self.mu)

#     def __call__(self, *args, **kwargs):
#         return self.evaluate(*args, **kwargs)

#     def sample(self, size=None, *args, **kwargs):
#         X = self.design_matrix(*args, **kwargs)
#         if size is None:
#             return X.dot(np.random.multivariate_normal(self.mu, self.cov))
#         return X.dot(np.random.multivariate_normal(self.mu, self.cov, size=size).T)

#     @property
#     def table_properties(self):
#         return [
#             (
#                 "w_{idx}",
#                 (self.mu[idx], self.sigma[idx]),
#                 (self.prior_mu[idx], self.prior_sigma[idx]),
#             )
#             for idx in range(self.width)
#         ]

#     @property
#     @abstractmethod
#     def arg_names(self):
#         """Returns a set of the user defined strings for all the arguments that the design matrix requires."""
#         pass

#     @property
#     @abstractmethod
#     def _equation(self):
#         """Returns a list of latex equations to describe the generation"""
#         pass

#     @property
#     def equation(self):
#         func_signature = ", ".join([f"\\mathbf{{{a}}}" for a in self.arg_names])
#         return (
#             f"\\[f({func_signature}) = "
#             + " + ".join(
#                 [f"{self._mu_letter}_{{{coeff}}} {e}" for coeff, e in enumerate(self._equation)]
#             )
#             + "\\]"
#         )

#     @property
#     def _mu_letter(self):
#         return "w"

#     @abstractmethod
#     def design_matrix(self):
#         """Returns a design matrix, given inputs listed in self.arg_names."""
#         pass

#     @property
#     @abstractmethod
#     def nvectors(self):
#         """Returns number of unique vectors required to build the design matrix."""
#         pass

#     @property
#     @abstractmethod
#     def width(self):
#         """Returns the width of the design matrix once built."""
#         pass

#     @abstractmethod
#     def fit(self):
#         """Fits the design matrix, given input vectors and data"""
#         pass

#     @property
#     @abstractmethod
#     def _INIT_ATTRS(self):
#         """Defines the variables needed to initialize self, so that they can be stored when saved."""
#         pass
