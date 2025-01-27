"""Experimental Bounded class..."""

import functools

from .io import IOMixins, LatexMixins
from .math import MathMixins
from .model import Model


class Bounded(MathMixins, LatexMixins, IOMixins):
    """Class that applies bounds to models."""

    def __init__(self, model: Model, bounds: tuple):
        """Bounded version of a model. The vector that drives the model will have the bound applied whenever it is used.

        Parameters:
        -----------

        model: lamatrix.model.Model
            An input model to bound
        bounds: tuple
            The bounds to apply to the model as a tuple. The first element of the tuple will be the lower bound.
            The second element is the upper bound. This will be applied as lower_bound > x <= upper_bound.
        """
        self.model = model
        self.x_name = self.model.x_name
        self.bounds = bounds

    def __repr__(self):
        return "Bounded " + self.model.__repr__()

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped instance.
        """
        attr = getattr(self.model, name)
        if callable(attr):

            @functools.wraps(attr)
            def wrapped(*args, **kwargs):
                if self.x_name in kwargs:
                    old_x = kwargs.pop(self.x_name)
                    mask = (old_x > self.bounds[0]) & (old_x <= self.bounds[1])
                    new_x = old_x * mask
                    kwargs[self.x_name] = new_x
                return attr(*args, **kwargs)

            return wrapped
        else:
            return attr

    def __dir__(self):
        """
        Extend the list of attributes to include those of the wrapped instance.
        """
        return dir(self.model)


# import numpy as np

# from .combine import StackedIndependentGenerator
# from .math import MathMixins

# __all__ = ["BoundedGenerator"]


# class BoundedGenerator(MathMixins, StackedIndependentGenerator):
#     def __init__(self, generator, bounds: list, x_name=None, fill_value=0):
#         self.generator = generator
#         if x_name is None:
#             self.x_name = self.generator.x_name
#         else:
#             self.x_name = x_name
#         if isinstance(bounds, slice):
#             self._latex_bounds = self._slice_bounds_to_latex(
#                 bounds.start, bounds.stop, bounds.step
#             )
#             self.nbounds = (bounds.stop - bounds.start) // bounds.step

#         else:
#             self._latex_bounds = self._bounds_to_latex(bounds)
#             self.nbounds = len(bounds)

#         self.bounds = bounds
#         self.fill_value = fill_value
#         self.fit_mu = None
#         self.fit_sigma = None
#         self.data_shape = None

#     def _bounds_to_latex(self, bounds):
#         bound_latex = [
#             f"""\\[b_{{{idx}}}(\\mathbf{{{self.x_name}}}) =
#         \\begin{{cases}}
#         \\mathbf{{{self.x_name}}}, & \\text{{if }} {bound[0]} < \\mathbf{{{self.x_name}}} \\leq {bound[1]} \\\\
#         0, & \\text{{otherwise}}
#         \\end{{cases}}\\]"""
#             for idx, bound in enumerate(bounds)
#         ]
#         return "\n".join(bound_latex)

#     def _slice_bounds_to_latex(self, start, stop, step):
#         step_str = f"{step}{{\cdot}}" if step != 1 else ""
#         nbounds = (stop - start) // step
#         if nbounds < 4:
#             def_str1 = "0"
#         else:
#             def_str1 = ", ".join([f"{s}" for s in np.arange(0, 3)])
#         def_str = f"For $i = {def_str1}, \\ldots, {nbounds} $ define:"
#         bound_latex = [
#             f"""b_{{i}}(\\mathbf{{{self.x_name}}}) =
#         \\begin{{cases}}
#         \\mathbf{{{self.x_name}}}, & \\text{{if }} ({start} + {step_str}i)
#           < \\mathbf{{{self.x_name}}} \\leq ({start} + {step_str}(i + 1)) \\\\
#         0, & \\text{{otherwise}}
#         \\end{{cases}}"""
#         ]
#         return def_str + "\[" + "\n".join(bound_latex) + "\]"

#     def __repr__(self):
#         str1 = f"{type(self).__name__}({type(self.generator).__name__}({', '.join(list(self.arg_names))}))[n, {self.width}]"

#         return str1

#     @property
#     def mu(self):
#         return self.prior_mu if self.fit_mu is None else self.fit_mu

#     @property
#     def sigma(self):
#         return self.prior_sigma if self.fit_sigma is None else self.fit_sigma

#     @property
#     def prior_mu(self):
#         return np.hstack([self.generator.prior_mu] * self.nbounds)

#     @property
#     def prior_sigma(self):
#         return np.hstack([self.generator.prior_sigma] * self.nbounds)

#     @property
#     def arg_names(self):
#         return {*np.unique([self.x_name, *list(self.generator.arg_names)])}

#     @property
#     def width(self):
#         return self.generator.width * self.nbounds

#     @property
#     def _equation(self):
#         eqn = ["b_{i}(" + eqn + ")" for eqn in self.generator._equation]
#         return eqn

#     @property
#     def equation(self):
#         func_signature = ", ".join([f"\\mathbf{{{a}}}" for a in self.arg_names])
#         n = self.nbounds
#         return (
#             f"\\[f({func_signature}) = "
#             + " + ".join(
#                 [
#                     f"\\sum_{{i=0}}^{{{n}}} w_{{{{i}}, {coeff} }} {e}"
#                     for coeff, e in enumerate(self._equation)
#                 ]
#             )
#             + "\\]"
#         )

#     def to_latex(self):
#         return "\n".join([self._latex_bounds, self.equation, self._to_latex_table()])

#     def design_matrix(self, *args, **kwargs):
#         if not self.arg_names.issubset(set(kwargs.keys())):
#             raise ValueError(f"Expected {self.arg_names} to be passed.")
#         x = kwargs.get(self.x_name)
#         if isinstance(self.bounds, slice):
#             bounds_list = [
#                 (a, a + self.bounds.step)
#                 for a in np.arange(
#                     self.bounds.start, self.bounds.stop, self.bounds.step
#                 )
#             ]
#         else:
#             bounds_list = self.bounds
#         if self.fill_value == 0:
#             return np.hstack(
#                 [
#                     self.generator.design_matrix(*args, **kwargs) * ((x >= b[0]) & (x < b[1]))[:, None]
#                     for b in bounds_list
#                 ]
#             )
#         return np.hstack(
#             [
#                 self.generator.design_matrix(*args, **kwargs) * ((x >= b[0]) & (x < b[1]))[:, None]
#                + (~(((x >= b[0]) & (x < b[1]))[:, None])).astype(float) * self.fill_value
#                 for b in bounds_list
#             ]
#         )

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     @property
#     def table_properties(self):
#         return [
#             (
#                 "w_{{{i}, {idx}}}",
#                 (self.mu[idx], self.sigma[idx]),
#                 (self.prior_mu[idx], self.prior_sigma[idx]),
#             )
#             for idx in range(self.width)
#         ]

#     def _to_latex_table(self):
#         latex_table = "\\begin{table}[h!]\n\\centering\n"
#         latex_table += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
#         latex_table += "Coefficient & Best Fit & Prior \\\\\\hline\n"
#         idx = 0
#         for tm in self._get_table_matter():
#             latex_table += tm.format(
#                 idx=idx % self.nbounds, i=(idx - (idx % self.nbounds)) // self.nbounds
#             )
#             idx += 1
#         latex_table += "\\end{tabular}\n\\end{table}"
#         return latex_table

#     def __getitem__(self, key):
#         g = self.generator.copy()
#         attrs = ["fit_mu", "fit_sigma"]
#         for attr in attrs:
#             setattr(
#                 g, attr, getattr(self, attr).reshape((self.nbounds, self.width // self.nbounds))[key]
#             )
#         return g

#     def __len__(self):
#         return self.nbounds
