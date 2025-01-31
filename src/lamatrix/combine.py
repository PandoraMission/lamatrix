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
        # self.priors = DistributionsContainer([p for g in self.models for p in g.priors])
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
