import json

import numpy as np
import re

from . import _META_DATA
from .generator import Generator

__all__ = ["StackedIndependentGenerator", "StackedDependentGenerator"]


def combine_equations(*equations):
    # Base case: if there's only one equation, just return it
    if len(equations) == 1:
        return equations[0]

    # Step case: combine the first two equations and recursively call the function with the result
    combined = [f + e for f in equations[1] for e in equations[0]]

    # If there are more equations left, combine further
    if len(equations) > 2:
        return combine_equations(combined, *equations[2:])
    else:
        return np.asarray(combined)


def combine_sigmas(*sigmas):
    # Base case: if there's only one equation, just return it
    if len(sigmas) == 1:
        return sigmas[0]

    if (np.isfinite(sigmas[0])).any():
        if sigmas[1][0] == np.inf:
            sigmas[1][0] = 0
    if (np.isfinite(sigmas[1])).any():
        if sigmas[0][0] == np.inf:
            sigmas[0][0] = 0

    # Step case: combine the first two equations and recursively call the function with the result
    combined = [(f**2 + e**2) ** 0.5 for f in sigmas[1] for e in sigmas[0]]

    # If there are more equations left, combine further
    if len(sigmas) > 2:
        return combine_sigmas(combined, *sigmas[2:])
    else:
        return np.asarray(combined)

def combine_masks(*masks):
    # Base case: if there's only one equation, just return it
    if len(masks) == 1:
        return masks[0]
    
    # Step case: combine the first two equations and recursively call the function with the result
    combined = [(f | e) for f in masks[1] for e in masks[0]]

    # If there are more equations left, combine further
    if len(masks) > 2:
        return combine_sigmas(combined, *masks[2:])
    else:
        return np.asarray(combined)

def combine_mus(*mus):
    return combine_equations(*mus)


def combine_matrices(*matrices):
    # Base case: if there's only one equation, just return it
    if len(matrices) == 1:
        return matrices[0]
    # Step case: combine the first two equations and recursively call the function with the result
    combined = [matrices[0] * f[:, None] for f in matrices[1].T]

    # If there are more equations left, combine further
    if len(matrices) > 2:
        return np.hstack(combine_matrices(combined, *matrices[2:]))
    else:
        return np.hstack(combined)


class StackedIndependentGenerator(Generator):
    def __init__(self, *args, **kwargs):
        if (
            not len(np.unique([a.data_shape for a in args if a.data_shape is not None]))
            <= 1
        ):
            raise ValueError("Can not have different `data_shape`.")
        self.generators = [a.copy() for a in args]
        self.data_shape = self.generators[0].data_shape
        self.fit_mu = None
        self.fit_sigma = None

        self.sigma_mask = []
        for idx, g in enumerate(self.generators):
            self.sigma_mask.append(np.copy(g.sigma_mask))
        self.sigma_mask = np.hstack(self.sigma_mask)

    def __repr__(self):
        str1 = (
            f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}]"
        )
        def add_tab_to_runs_of_tabs(repr_str):
            pattern = r'\t+'
            replacement = lambda match: match.group(0) + '\t'
            result_string = re.sub(pattern, replacement, repr_str)
            return result_string

        str2 = [f"\t{add_tab_to_runs_of_tabs(g.__repr__())}" for g in self.generators]
        return "\n".join([str1, *str2])

    def __add__(self, other):
        if isinstance(other, Generator):
            return StackedIndependentGenerator(self, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")

    def __mul__(self, other):
        if isinstance(other, Generator):
            return StackedDependentGenerator(self, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")

    def __getitem__(self, key):
        return self.generators[key]

    # def __add__(self, other):
    #     if isinstance(other, Generator):
    #         return VStackedGenerator(*self.generators, other)
    #     else:
    #         raise ValueError("Can only combine `Generator` objects.")

    def design_matrix(self, *args, **kwargs):
        return np.hstack([g.design_matrix(*args, **kwargs) for g in self.generators])

    @property
    def gradient(self):
        return StackedIndependentGenerator(*[g.gradient for g in self.generators])

    @property
    def width(self):
        return np.sum([g.width for g in self.generators])

    @property
    def nvectors(self):
        return len(self.arg_names)

    @property
    def prior_mu(self):
        prior_mu = []
        for idx, g in enumerate(self.generators):
            pm = np.copy(g.prior_mu)
            if idx != 0:
                pm[0] = 0
            else:
                pm[0] = np.sum([g.prior_mu[0] for g in self.generators])
            prior_mu.append(pm)
        return np.hstack(prior_mu)

    @property
    def prior_sigma_raw(self):
        prior_sigma_raw = []
        for idx, g in enumerate(self.generators):
            pm = np.copy(g.prior_sigma_raw)
            if idx != 0:
                pm[0] = 0
            else:
                pm[0] = (
                    np.nansum(np.asarray([g.prior_sigma_raw[0] if g.prior_sigma_raw[0] != np.inf else 0 for g in self.generators], dtype=float) ** 2)
                    ** 0.5
                )
            prior_sigma_raw.append(pm)
        if (np.asarray([pm[0] for pm in prior_sigma_raw]) == 0).all():
            prior_sigma_raw[0][0] = np.inf
        return np.hstack(prior_sigma_raw)
    
    # This may not be necessary, since it is identical to the one in Generator
    # @property
    # def prior_sigma(self):
    #     ps = self.prior_sigma_raw.copy()
    #     ps[self.sigma_mask] = 0.
    #     return ps

    # @property
    # def mu(self):
    #     mu = []
    #     for idx, g in enumerate(self.generators):
    #         pm = np.copy(g.mu)
    #         if idx != 0:
    #             pm[0] = 0
    #         mu.append(pm)
    #     return np.hstack(mu)

    # @property
    # def sigma(self):
    #     sigma = []
    #     for idx, g in enumerate(self.generators):
    #         pm = np.copy(g.sigma)
    #         if idx != 0:
    #             pm[0] = 0
    #         sigma.append(pm)
    #     return np.hstack(sigma)

    def update_priors(self):
        if self.fit_mu is None:
            raise ValueError("Can not update priors before fitting.")
        new = self.copy()
        for idx in range(len(new)):
            new[idx].prior_mu = new[idx].fit_mu.copy()
            new[idx].prior_sigma = new[idx].fit_sigma.copy()
        return new

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)
        lengths = [g.width for g in self.generators]
        mu, sigma = (
            np.array_split(self.fit_mu, np.cumsum(lengths))[:-1],
            np.array_split(self.fit_sigma, np.cumsum(lengths))[:-1],
        )
        for idx, mu0, sigma0 in zip(np.arange(len(mu)), mu, sigma):
            self[idx].fit_mu = mu0
            self[idx].fit_sigma = sigma0

        indices = np.cumsum([0, *[g.width for g in self.generators]])
        for idx, a, b in zip(range(len(indices) - 1), indices[:-1], indices[1:]):
            self[idx].cov = self.cov[a:b, a:b]

    def __len__(self):
        return len(self.generators)

    @property
    def _equation(self):
        return np.hstack([g._equation for g in self.generators])

    @property
    def arg_names(self):
        return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))

    @property
    def _INIT_ATTRS(self):
        return []

    def save(self, filename: str):
        if not filename.endswith(".json"):
            filename = filename + ".json"

        # Write to a JSON file
        with open(filename, "w") as json_file:
            data_to_store = self._create_save_data()
            generators_to_store = {
                f"generator{idx+1}": g._create_save_data()
                for idx, g in enumerate(self.generators)
            }
            data_to_store["generators"] = generators_to_store
            data_to_store["metadata"] = _META_DATA()
            json.dump(data_to_store, json_file, indent=4)


class StackedDependentGenerator(StackedIndependentGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prior_mu = None
        self._prior_sigma = None
        self.sigma_mask = combine_masks(*[g.sigma_mask for g in self.generators])

    @property
    def width(self):
        return np.prod([g.width for g in self.generators])

    def design_matrix(self, *args, **kwargs):
        return combine_matrices(
            *[g.design_matrix(*args, **kwargs) for g in self.generators]
        )

    @property
    def _equation(self):
        return combine_equations(*[g._equation for g in self.generators])

    @property
    def arg_names(self):
        return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))

    @property
    def nvectors(self):
        return len(self.arg_names)

    @property
    def width(self):
        return np.prod([g.width for g in self.generators])

    @property
    def prior_sigma(self):
        if self._prior_sigma is None:
            ps = combine_sigmas(*[g.prior_sigma_raw for g in self.generators])
        else:
            ps = self._prior_sigma
        ps[self.sigma_mask] = 0.
        return ps

    @property
    def prior_mu(self):
        if self._prior_mu is None:
            return combine_mus(*[g.prior_mu for g in self.generators])
        else:
            return self._prior_mu

    def fit(self, *args, **kwargs):
        self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

    # @property
    # def mu(self):
    #     return self.prior_mu if self.fit_mu is None else self.fit_mu

    # @property
    # def sigma(self):
    #     return self.prior_sigma if self.fit_sigma is None else self.fit_sigma

    def __getitem__(self, key):
        raise AttributeError(
            "Can not extract individual generators from a dependent stacked generator."
        )

    @property
    def gradient(self):
        raise AttributeError(
            "Can not create a gradient for a dependent stacked generator."
        )

    def update_priors(self):
        if self.fit_mu is None:
            raise ValueError("Can not update priors before fitting.")
        new = self.copy()
        new._prior_mu = new.fit_mu.copy()
        new._prior_sigma = new.fit_sigma.copy()
        return new
