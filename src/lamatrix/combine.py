"""Classes and methods to combine models"""

import itertools

import numpy as np
from scipy import sparse

from .distributions import Distribution, DistributionsContainer
from .io import IOMixins, LatexMixins
from .model import Model

__all__ = ["JointModel", "CrosstermModel"]


# def combine_matrices(*matrices):
#     if len(matrices) == 1:
#         return matrices[0]
#     # Step case: combine the first two equations and recursively call the function with the result
#     combined = [matrices[0] * f[:, None] for f in matrices[1].T]

#     # If there are more equations left, combine further
#     if len(matrices) > 2:
#         return np.hstack(combine_matrices(combined, *matrices[2:]))
#     else:
#         return np.hstack(combined)


def _combine_equations(*equations):
    # Base case: if there's only one equation, just return it
    if len(equations) == 1:
        return equations[0]

    # Step case: combine the first two equations and recursively call the function with the result
    combined = [f + e for f in equations[1] for e in equations[0]]

    # If there are more equations left, combine further
    if len(equations) > 2:
        return _combine_equations(combined, *equations[2:])
    else:
        return np.asarray(combined)


class JointModel(Model, IOMixins, LatexMixins):
    def __init__(self, *args):
        # Check that every arg is a generator
        if not np.all([isinstance(a, Model) for a in args]):
            raise ValueError("Can only combine `Model` objects.")
        self.models = [a.copy() for a in args]
        self.widths = [g.width for g in self.models]
        self.posteriors = DistributionsContainer.from_number(np.sum(self.widths))

    def __getitem__(self, key):
        if isinstance(key, slice):
            new = self.__class__(*self.models[key])
            new.posteriors = DistributionsContainer(
                [l for m in new.models for l in m.posteriors]
            )
            return new
        return self.models[key]

    def __repr__(self):
        return f"{type(self).__name__}\n\t" + "\n\t".join(
            [g.__repr__() for g in self.models]
        )

    # def set_prior(self, index, distribution):
    #     cs = np.cumsum(self.widths)
    #     generator_index = np.where(cs >= index)[0][0]
    #     vector_index = index - cs[generator_index - 1] if generator_index > 0 else 0
    #     return self.models[generator_index].set_prior(vector_index, distribution)

    # def set_priors(self, distributions):
    #     cs = [0, *np.cumsum(self.widths)]
    #     for idx, g in enumerate(self.models):
    #         g.set_priors(
    #             [distributions[jdx] for jdx in np.arange(cs[idx], cs[idx + 1])]
    #         )

    @property
    def _initialization_attributes(self):
        return []

    @property
    def _equation(self):
        return np.hstack([g._equation for g in self.generators])

    @property
    def arg_names(self):
        return {*np.unique(np.hstack([list(g.arg_names) for g in self.models]))}

    @property
    def _equation(self):
        return [*np.hstack([g._equation for g in self.models])]

    @property
    def priors(self):
        return DistributionsContainer([p for g in self.models for p in g.priors])

    # @property
    # def prior_mean(self):
    #     return np.asarray(
    #         [
    #             distribution.mean
    #             for g in self.models
    #             for distribution in g.priors
    #         ]
    #     )

    # @property
    # def prior_std(self):
    #     return np.asarray(
    #         [
    #             distribution.std
    #             for g in self.models
    #             for distribution in g.prior_distributions
    #         ]
    #     )

    # @property
    # def fit_mean(self):
    #     return np.asarray([distribution[0] for distribution in self.fit_distributions])

    # @property
    # def fit_std(self):
    #     return np.asarray([distribution[1] for distribution in self.fit_distributions])

    def design_matrix(self, *args, **kwargs):
        Xs = [g.design_matrix(*args, **kwargs) for g in self.models]
        if np.all([sparse.issparse(matrix) for matrix in Xs]):
            return sparse.hstack(Xs, format="csr")
        elif np.all([not sparse.issparse(matrix) for matrix in Xs]):
            ndim = Xs[0].ndim - 1
            shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
            shape_b = [ndim, *np.arange(0, ndim)]
            return np.vstack([X.transpose(shape_b) for X in Xs]).transpose(shape_a)
        else:
            raise ValueError("Can not combine sparse and dense matrices.")

    @property
    def width(self):
        return np.sum(self.widths)

    @property
    def nvectors(self):
        return np.sum([g.nvectors for g in self.models])

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        means = np.array_split(self.posteriors.mean, np.cumsum(self.widths)[:-1])
        stds = np.array_split(self.posteriors.std, np.cumsum(self.widths)[:-1])

        for idx, mean, std in zip(range(len(self.models)), means, stds):
            self.models[idx].posteriors = DistributionsContainer(
                [Distribution((m, s)) for m, s in zip(mean, std)]
            )

    def __add__(self, other):
        has_constant = np.any([g.arg_names == {} for g in self.models])
        if (other.arg_names == {}) & (has_constant):
            return self
        if isinstance(other, JointModel):
            if has_constant:
                return JointModel(
                    *self.models, *[g for g in other.models if not g.arg_names == {}]
                )
            else:
                return JointModel(*self.models, *other.models)
        elif isinstance(other, Model):
            if (has_constant) & (other.arg_names == {}):
                raise ValueError("Can not combine multiple `ConstantModel`s")
            else:
                return JointModel(*self.models, other)
        else:
            raise ValueError(f"Can only combine `Model` objects, not {type(other)}.")

    def __mul__(self, other):
        if other.arg_names == {}:
            return self
        if isinstance(other, CrosstermModel):
            return CrosstermModel(self, *other.models)
        if isinstance(other, JointModel):
            return JointModel(*[g * p for g in self.models for p in other.models])
        elif isinstance(other, Model):
            return JointModel(*[g * other for g in self.models])


class CrosstermModel(Model, IOMixins, LatexMixins):
    def __init__(self, *args):
        # Check that every arg is a generator
        if not np.all([isinstance(a, Model) for a in args]):
            raise ValueError("Can only combine `Model` objects.")
        self.models = [a.copy() for a in args]
        self.widths = [g.width for g in self.models]
        self.posteriors = DistributionsContainer.from_number(np.prod(self.widths))
        prior_mean = np.asarray(
            [
                means[0] * means[1]
                for means in itertools.product(
                    *[
                        [distribution[0] for distribution in g.priors]
                        for g in self.models
                    ]
                )
            ]
        )
        prior_std = np.sqrt(
            np.asarray(
                [
                    means[0] ** 2 * stds[0] ** 2
                    + means[1] ** 2 * stds[1] ** 2
                    + stds[0] ** 2 * stds[1] ** 2
                    for means, stds in zip(
                        itertools.product(
                            *[
                                [distribution[1] for distribution in g.priors]
                                for g in self.models
                            ]
                        ),
                        itertools.product(
                            *[
                                [distribution[1] for distribution in g.priors]
                                for g in self.models
                            ]
                        ),
                    )
                ]
            )
        )
        self.priors = DistributionsContainer(
            [Distribution(m, s) for m, s in zip(prior_mean, prior_std)]
        )
        # self._validate_distributions(prior_distributions)
        # self.prior_distributions = prior_distributions

    @property
    def _initialization_attributes(self):
        return []

    @property
    def arg_names(self):
        return {*np.unique(np.hstack([list(g.arg_names) for g in self.models]))}

    @property
    def width(self):
        return np.prod(self.widths)

    @property
    def nvectors(self):
        return np.sum([g.nvectors for g in self.models])

    @property
    def _equation(self):
        return np.hstack(
            [
                f"{eqns[0]}{eqns[1]}"
                for eqns in itertools.product(*[g._equation for g in self.models])
            ]
        )

    # @property
    # def prior_distributions(self):
    #     return [(m, s) for m, s in zip(self.prior_mean, self.prior_std)]

    # @property
    # def prior_mean(self):
    #     return np.asarray(
    #         [
    #             np.sum(i)
    #             for i in itertools.product(
    #                 *[
    #                     [distribution[0] for distribution in g.prior_distributions]
    #                     for g in self.models
    #                 ]
    #             )
    #         ]
    #     )

    # @property
    # def prior_std(self):
    #     return np.asarray(
    #         [
    #             np.prod(i)
    #             for i in itertools.product(
    #                 *[
    #                     [distribution[1] for distribution in g.prior_distributions]
    #                     for g in self.models
    #                 ]
    #             )
    #         ]
    #     )

    # @property
    # def fit_mean(self):
    #     return np.asarray([distribution[0] for distribution in self.fit_distributions])

    # @property
    # def fit_std(self):
    #     return np.asarray([distribution[1] for distribution in self.fit_distributions])

    def design_matrix(self, *args, **kwargs):
        Xs = [g.design_matrix(*args, **kwargs) for g in self.models]
        if np.all([sparse.issparse(matrix) for matrix in Xs]):
            X = sparse.hstack(
                [i[0].multiply(i[1]).T for i in itertools.product(*[x.T for x in Xs])],
                format="csr",
            )
            return X
        elif np.all([not sparse.issparse(matrix) for matrix in Xs]):
            ndim = Xs[0].ndim - 1
            shape_a = [*np.arange(1, ndim + 1).astype(int), 0]
            shape_b = [ndim, *np.arange(0, ndim)]
            Xs = [X.transpose(shape_b) for X in Xs]
            X = np.vstack(
                [
                    np.expand_dims(np.prod(i, axis=0), ndim).transpose(shape_b)
                    for i in itertools.product(*Xs)
                ]
            ).transpose(shape_a)
            return X
        else:
            raise ValueError("Can not combine sparse and dense matrices.")

        print(itertools.product())

        return np.vstack(
            [
                np.expand_dims(np.prod(i, axis=0), axis=ndim)
                for i in itertools.product(*Xs)
            ]
        ).transpose(shape_a)
        return np.vstack(
            [np.prod(i, axis=0) for i in itertools.product(*Xs)]
        ).transpose(shape_a)

        return np.vstack([X.transpose(shape_b) for X in Xs]).transpose(shape_a)

        return np.asarray(
            [
                np.prod(i, axis=0)
                for i in itertools.product(
                    *[g.design_matrix(**kwargs).T for g in self.models]
                )
            ]
        ).T

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, CrosstermModel):
            return JointModel(self, other)
        if isinstance(other, JointModel):
            return JointModel(self, *other.models)
        elif isinstance(other, Model):
            return JointModel(self, other)
        else:
            raise ValueError("Can only combine `Model` objects.")

    def __mul__(self, other):
        if isinstance(other, CrosstermModel):
            raise ValueError
        if isinstance(other, JointModel):
            raise ValueError
        elif isinstance(other, Model):
            return CrosstermModel(*self.models, other)
        else:
            raise ValueError("Can only combine `Model` objects.")


# class CrosstermGenerator(Generator):
#     def __init__(self, generator1, generator2):
#         # Check that every arg is a generator
#         if not (isinstance(generator1, Generator) & isinstance(generator2, Generator)):
#             raise ValueError("Can only combine `Generator` objects.")
#         self.generator1, self.generator2 = generator1.copy(), generator2.copy()
#         self.widths = [self.generator1.width, self.generator2.width]
#         self.fit_distributions = [[(None, None)] * self.generator1.width] * self.generator2.width

#     @property
#     def arg_names(self):
#         return {*np.unique(np.hstack([list(self.generator1.arg_names), list(self.generator2.arg_names)]))}

#     @property
#     def prior_distributions(self):
#         return [[(d1[0] + d2[0], (d1[1]**2 + d2[1]**2)**0.5) for d1 in self.generator1.prior_distributions] for d2 in self.generator2.prior_distributions]

#     @property
#     def prior_mean(self):
#         return np.asarray([[d1[0] + d2[0] for d1 in self.generator1.prior_distributions] for d2 in self.generator2.prior_distributions])

#     @property
#     def prior_std(self):
#         return np.asarray([[(d1[1]**2 * d2[1]**2)**0.5 for d1 in self.generator1.prior_distributions] for d2 in self.generator2.prior_distributions])

#     # @property
#     # def fit_mean(self):
#     #     return np.asarray([distribution[0] for g in self.generators for distribution in g.fit_distributions])

#     # @property
#     # def fit_std(self):
#     #     return np.asarray([distribution[1] for g in self.generators for distribution in g.fit_distributions])

#     # def design_matrix(self, *args, **kwargs):
#     #     return np.hstack([g.design_matrix(*args, **kwargs) for g in self.generators])

#     @property
#     def width(self):
#         return np.prod(self.widths)

#     @property
#     def nvectors(self):
#         return np.sum([g.nvectors for g in self.generators])

#     def design_matrix(self, **kwargs):
#         return combine_matrices(self.generator1.design_matrix(**kwargs), self.generator2.design_matrix(**kwargs))


# import json

# import numpy as np
# import re

# from . import _META_DATA
# from .generator import Generator

# __all__ = ["StackedIndependentGenerator", "StackedDependentGenerator"]


# def combine_equations(*equations):
#     # Base case: if there's only one equation, just return it
#     if len(equations) == 1:
#         return equations[0]

#     # Step case: combine the first two equations and recursively call the function with the result
#     combined = [f + e for f in equations[1] for e in equations[0]]

#     # If there are more equations left, combine further
#     if len(equations) > 2:
#         return combine_equations(combined, *equations[2:])
#     else:
#         return np.asarray(combined)


# def combine_sigmas(*sigmas):
#     # Base case: if there's only one equation, just return it
#     if len(sigmas) == 1:
#         return sigmas[0]

#     if (np.isfinite(sigmas[0])).any():
#         if sigmas[1][0] == np.inf:
#             sigmas[1][0] = 0
#     if (np.isfinite(sigmas[1])).any():
#         if sigmas[0][0] == np.inf:
#             sigmas[0][0] = 0

#     # Step case: combine the first two equations and recursively call the function with the result
#     combined = [(f**2 + e**2) ** 0.5 for f in sigmas[1] for e in sigmas[0]]

#     # If there are more equations left, combine further
#     if len(sigmas) > 2:
#         return combine_sigmas(combined, *sigmas[2:])
#     else:
#         return np.asarray(combined)


# def combine_mus(*mus):
#     return combine_equations(*mus)


# def combine_matrices(*matrices):
#     # Base case: if there's only one equation, just return it
#     if len(matrices) == 1:
#         return matrices[0]
#     # Step case: combine the first two equations and recursively call the function with the result
#     combined = [matrices[0] * f[:, None] for f in matrices[1].T]

#     # If there are more equations left, combine further
#     if len(matrices) > 2:
#         return np.hstack(combine_matrices(combined, *matrices[2:]))
#     else:
#         return np.hstack(combined)


# class StackedIndependentGenerator(Generator):
#     def __init__(self, *args, **kwargs):
#         if (
#             not len(np.unique([a.data_shape for a in args if a.data_shape is not None]))
#             <= 1
#         ):
#             raise ValueError("Can not have different `data_shape`.")
#         self.generators = [a.copy() for a in args]
#         self.data_shape = self.generators[0].data_shape
#         self.fit_mu = None
#         self.fit_sigma = None

#     def __repr__(self):
#         str1 = (
#             f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}]"
#         )
#         def add_tab_to_runs_of_tabs(repr_str):
#             pattern = r'\t+'
#             replacement = lambda match: match.group(0) + '\t'
#             result_string = re.sub(pattern, replacement, repr_str)
#             return result_string

#         str2 = [f"\t{add_tab_to_runs_of_tabs(g.__repr__())}" for g in self.generators]
#         return "\n".join([str1, *str2])

#     def __add__(self, other):
#         if isinstance(other, Generator):
#             return StackedIndependentGenerator(self, other)
#         else:
#             raise ValueError("Can only combine `Generator` objects.")

#     def __mul__(self, other):
#         if isinstance(other, Generator):
#             return StackedDependentGenerator(self, other)
#         else:
#             raise ValueError("Can only combine `Generator` objects.")

#     def __getitem__(self, key):
#         return self.generators[key]

#     # def __add__(self, other):
#     #     if isinstance(other, Generator):
#     #         return VStackedGenerator(*self.generators, other)
#     #     else:
#     #         raise ValueError("Can only combine `Generator` objects.")

#     def design_matrix(self, *args, **kwargs):
#         return np.hstack([g.design_matrix(*args, **kwargs) for g in self.generators])

#     @property
#     def gradient(self):
#         return StackedIndependentGenerator(*[g.gradient for g in self.generators])

#     @property
#     def width(self):
#         return np.sum([g.width for g in self.generators])

#     @property
#     def nvectors(self):
#         return len(self.arg_names)

#     @property
#     def prior_mu(self):
#         prior_mu = []
#         for idx, g in enumerate(self.generators):
#             pm = np.copy(g.prior_mu)
#             if idx != 0:
#                 pm[0] = 0
#             else:
#                 pm[0] = np.sum([g.prior_mu[0] for g in self.generators])
#             prior_mu.append(pm)
#         return np.hstack(prior_mu)

#     @property
#     def prior_sigma(self):
#         prior_sigma = []
#         for idx, g in enumerate(self.generators):
#             pm = np.copy(g.prior_sigma)
#             if idx != 0:
#                 pm[0] = 0
#             else:
#                 pm[0] = (
#                     np.nansum(np.asarray([g.prior_sigma[0] if g.prior_sigma[0] != np.inf else 0 for g in self.generators], dtype=float) ** 2)
#                     ** 0.5
#                 )
#             prior_sigma.append(pm)
#         if (np.asarray([pm[0] for pm in prior_sigma]) == 0).all():
#             prior_sigma[0][0] = np.inf
#         return np.hstack(prior_sigma)

#     # @property
#     # def mu(self):
#     #     mu = []
#     #     for idx, g in enumerate(self.generators):
#     #         pm = np.copy(g.mu)
#     #         if idx != 0:
#     #             pm[0] = 0
#     #         mu.append(pm)
#     #     return np.hstack(mu)

#     # @property
#     # def sigma(self):
#     #     sigma = []
#     #     for idx, g in enumerate(self.generators):
#     #         pm = np.copy(g.sigma)
#     #         if idx != 0:
#     #             pm[0] = 0
#     #         sigma.append(pm)
#     #     return np.hstack(sigma)

#     def update_priors(self):
#         if self.fit_mu is None:
#             raise ValueError("Can not update priors before fitting.")
#         new = self.copy()
#         for idx in range(len(new)):
#             new[idx].prior_mu = new[idx].fit_mu.copy()
#             new[idx].prior_sigma = new[idx].fit_sigma.copy()
#         return new

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)
#         lengths = [g.width for g in self.generators]
#         mu, sigma = (
#             np.array_split(self.fit_mu, np.cumsum(lengths))[:-1],
#             np.array_split(self.fit_sigma, np.cumsum(lengths))[:-1],
#         )
#         for idx, mu0, sigma0 in zip(np.arange(len(mu)), mu, sigma):
#             self[idx].fit_mu = mu0
#             self[idx].fit_sigma = sigma0

#         indices = np.cumsum([0, *[g.width for g in self.generators]])
#         for idx, a, b in zip(range(len(indices) - 1), indices[:-1], indices[1:]):
#             self[idx].cov = self.cov[a:b, a:b]

#     def __len__(self):
#         return len(self.generators)

#     @property
#     def _equation(self):
#         return np.hstack([g._equation for g in self.generators])

#     @property
#     def arg_names(self):
#         return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))

#     @property
#     def _INIT_ATTRS(self):
#         return []

#     def save(self, filename: str):
#         if not filename.endswith(".json"):
#             filename = filename + ".json"

#         # Write to a JSON file
#         with open(filename, "w") as json_file:
#             data_to_store = self._create_save_data()
#             generators_to_store = {
#                 f"generator{idx+1}": g._create_save_data()
#                 for idx, g in enumerate(self.generators)
#             }
#             data_to_store["generators"] = generators_to_store
#             data_to_store["metadata"] = _META_DATA()
#             json.dump(data_to_store, json_file, indent=4)


# class StackedDependentGenerator(StackedIndependentGenerator):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._prior_mu = None
#         self._prior_sigma = None

#     @property
#     def width(self):
#         return np.prod([g.width for g in self.generators])

#     def design_matrix(self, *args, **kwargs):
#         return combine_matrices(
#             *[g.design_matrix(*args, **kwargs) for g in self.generators]
#         )

#     @property
#     def _equation(self):
#         return combine_equations(*[g._equation for g in self.generators])

#     @property
#     def arg_names(self):
#         return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))

#     @property
#     def nvectors(self):
#         return len(self.arg_names)

#     @property
#     def width(self):
#         return np.prod([g.width for g in self.generators])

#     @property
#     def prior_sigma(self):
#         if self._prior_sigma is None:
#             return combine_sigmas(*[g.prior_sigma for g in self.generators])
#         else:
#             return self._prior_sigma

#     @property
#     def prior_mu(self):
#         if self._prior_mu is None:
#             return combine_mus(*[g.prior_mu for g in self.generators])
#         else:
#             return self._prior_mu

#     def fit(self, *args, **kwargs):
#         self.fit_mu, self.fit_sigma = self._fit(*args, **kwargs)

#     # @property
#     # def mu(self):
#     #     return self.prior_mu if self.fit_mu is None else self.fit_mu

#     # @property
#     # def sigma(self):
#     #     return self.prior_sigma if self.fit_sigma is None else self.fit_sigma

#     def __getitem__(self, key):
#         raise AttributeError(
#             "Can not extract individual generators from a dependent stacked generator."
#         )

#     @property
#     def gradient(self):
#         raise AttributeError(
#             "Can not create a gradient for a dependent stacked generator."
#         )

#     def update_priors(self):
#         if self.fit_mu is None:
#             raise ValueError("Can not update priors before fitting.")
#         new = self.copy()
#         new._prior_mu = new.fit_mu.copy()
#         new._prior_sigma = new.fit_sigma.copy()
#         return new
