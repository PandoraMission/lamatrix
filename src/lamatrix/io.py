"""Mixins to deal with saving/loading models"""

import json
import math

import numpy as np

from . import _META_DATA


def format_significant_figures(mean, error):
    # Check for inf, -inf, or NaN
    if math.isinf(mean) or math.isinf(error) or math.isnan(mean) or math.isnan(error):
        # Handle these cases as you see fit, for example:
        return "0", "\\infty"

    # Find the first significant digit of the error
    if error == 0:
        sig_figures = 0
    else:
        sig_figures = np.max([0, -int(math.floor(math.log10(abs(error))))])

    # Format mean and error to have the same number of decimal places
    formatted_mean = f"{mean:.{sig_figures}f}"
    formatted_error = f"{error:.{sig_figures}f}"
    return formatted_mean, formatted_error


class LatexMixins:
    """Functions to show latex tables"""

    def _get_table_matter(self):
        table_matter = []
        for idx in range(self.width):
            if self.posteriors is not None:
                formatted_fit_mean, formatted_fit_error = format_significant_figures(
                    self.posteriors[idx].mean, self.posteriors[idx].std
                )
            else:
                formatted_fit_mean = ""
                formatted_fit_error = ""
            formatted_prior_mean, formatted_prior_error = format_significant_figures(
                self.priors[idx].mean, self.priors[idx].std
            )
            row = (
                f"{self._mu_letter} & ${formatted_fit_mean} \\pm {formatted_fit_error}$ "
                f" & ${formatted_prior_mean} \\pm {formatted_prior_error}$ \\\\\\hline\n"
            )
            table_matter.append(row)
        return table_matter

    def _to_latex_table(self):
        latex_table = "\\begin{table}[h!]\n\\centering\n"
        latex_table += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
        latex_table += "Coefficient & Posterior & Prior \\\\\\hline\n"
        idx = 0
        for tm in self._get_table_matter():
            latex_table += tm.format(idx=idx)
            idx += 1
        latex_table += "\\end{tabular}\n\\end{table}"
        return latex_table

    def to_latex(self):
        return "\n".join([self.equation, self._to_latex_table()])


class IOMixins:
    """Functions to save data"""

    def _create_save_data(self):
        def process(arg):
            if arg is None:
                return None
            elif isinstance(arg, (str, int, float, list, tuple)):
                if arg is np.inf:
                    return "Infinity"
                return arg
            elif isinstance(arg, np.ndarray):
                arg = arg.tolist()
                arg = [a if a != np.inf else "Infinity" for a in arg]
                return arg

        kwargs = {
            attr: process(getattr(self, attr))
            for attr in self._initialization_attributes
        }
        type_name = type(self).__name__

        data_to_store = {
            "object_type": type_name,
            "initializing_kwargs": kwargs,
            "priors": self.priors.to_dict(),
            "posteriors": (
                self.posteriors.to_dict() if self.posteriors is not None else None
            ),
            "equation": self.equation,
            "latex": self.to_latex(),
        }
        return data_to_store

    def save(self, filename: str):
        data_to_store = to_save_tree(self)
        data_to_store["metadata"] = _META_DATA()
        if not filename.endswith(".json"):
            filename = filename + ".json"

        # Write to a JSON file
        with open(filename, "w") as json_file:
            json.dump(data_to_store, json_file, indent=4)


def to_save_tree(obj, *, _seen=None):
    """Return a dict of obj._create_save_data() plus any nested models."""
    if _seen is None:
        _seen = set()

    oid = id(obj)
    if oid in _seen:
        # Prevent infinite loops on cyclic/shared graphs
        return {"$ref": oid}
    _seen.add(oid)

    data = obj._create_save_data()

    # If there are nested models, recurse.
    models = getattr(obj, "models", None)
    if models:
        # Support list/tuple, or dict of models
        if isinstance(models, dict):
            data["models"] = {
                k: to_save_tree(v, _seen=_seen) for k, v in models.items()
            }
        else:
            # keep your existing naming: model1, model2, ...
            data["models"] = {
                f"model{i + 1}": to_save_tree(m, _seen=_seen)
                for i, m in enumerate(models)
            }

    return data
