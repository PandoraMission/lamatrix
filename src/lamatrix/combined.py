import numpy as np

from .generator import Generator

__all__ = ["VStackedGenerator"]


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
        return combined


class VStackedGenerator(Generator):
    def __init__(self, *args, **kwargs):
        if (
            not len(np.unique([a.data_shape for a in args if a.data_shape is not None]))
            <= 1
        ):
            raise ValueError("Can not have different `data_shape`.")
        self.generators = [a.copy() for a in args]
        self.data_shape = self.generators[0].data_shape

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
    def prior_sigma(self):
        prior_sigma = []
        for idx, g in enumerate(self.generators):
            pm = np.copy(g.prior_sigma)
            if idx != 0:
                pm[0] = 0
            else:
                pm[0] = np.sum(np.asarray([g.prior_sigma[0] for g in self.generators])**2)**0.5
            prior_sigma.append(pm)
        return np.hstack(prior_sigma)

    @property
    def mu(self):
        mu = []
        for idx, g in enumerate(self.generators):
            pm = np.copy(g.mu)
            if idx != 0:
                pm[0] = 0
            mu.append(pm)
        return np.hstack(mu)

    @property
    def sigma(self):
        sigma = []
        for idx, g in enumerate(self.generators):
            pm = np.copy(g.sigma)
            if idx != 0:
                pm[0] = 0
            sigma.append(pm)
        return np.hstack(sigma)

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

    def __len__(self):
        return len(self.generators)

    # @property
    # def _equation(self):
    #     return combine_equations(*[g._equation for g in self])

    @property
    def _equation(self):
        return np.hstack([g._equation for g in self.generators])

    @property
    def arg_names(self):
        return np.unique(np.hstack([list(g.arg_names) for g in self.generators]))


class CombinedGenerator(VStackedGenerator):

    @property
    def width(self):
        return np.prod([g.width for g in self.generators])

    # def design_matrix(self, *args, **kwargs):
    #     return np.hstack([g.design_matrix(*args, **kwargs) for g in self.generators])

    @property
    def _equation(self):
        return combine_equations(*[g._equation for g in self])
