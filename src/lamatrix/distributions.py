"""Classes to hold distributions, either of priors or of fit values."""

from typing import List, Tuple

import numpy as np


class Distribution(tuple):
    """Special tuple with the ability to "freeze", i.e. set standard deviation to 0."""

    def __new__(cls, mean, std=None):
        # Ensure the tuple is initialized with mean and std
        if isinstance(mean, tuple):
            if len(mean) == 2:
                return super().__new__(cls, mean)
            else:
                raise ValueError("Must pass mean and standard deviation.")
        else:
            if std is None:
                raise ValueError("Must pass mean and standard deviation.")
            return super().__new__(cls, (mean, std))

    def __init__(self, mean, std=None):
        self.frozen = False  # Initialize with 'frozen' set to False

    def freeze(self):
        """Freeze the distribution by setting the standard deviation to 0."""
        self.frozen = True

    def thaw(self):
        """Unfreeze the distribution to restore the original standard deviation."""
        self.frozen = False

    @property
    def mean(self):
        return self[0]

    @property
    def std(self):
        return 1e-10 if self.frozen else self[1]

    def __repr__(self):
        """Custom string representation"""
        if self.frozen:
            return f"({self.mean}, {self.std}) [Frozen]"
        return f"({self.mean}, {self.std})"

    def as_tuple(self):
        """Return the distribution as a standard tuple."""
        return (self.mean, self.std)

    def sample(self):
        if self.frozen:
            return self.mean
        return np.random.normal(self.mean, self.std)


class DistributionsContainer:
    """Holds distributions"""

    def __init__(self, distributions: List[Tuple]):
        if not isinstance(distributions, list) or not all(
            isinstance(d, tuple) and len(d) == 2 for d in distributions
        ):
            raise ValueError(
                "Distributions should be a list of tuples, each containing (mean, std)"
            )

        # Validate each distribution
        for mean, std in distributions:
            if not (
                isinstance(mean, (int, float, np.integer, np.number))
                and isinstance(std, (int, float, np.integer, np.number))
            ):
                raise ValueError("Distribution's mean and std should be numbers")
            if std < 0:
                raise ValueError("Standard deviation must be non-negative")

        self._distributions = [Distribution(*dist) for dist in distributions]
        self._length = len(distributions)

    @staticmethod
    def from_number(number_of_distributions: int):
        """Make a set of empty distributions with a given length"""
        return DistributionsContainer([(0, np.inf)] * number_of_distributions)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._distributions[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            # Case where slice is all elements
            if (index.start is None) & (index.stop is None) & (index.step is None):
                if not len(value) == self._length:
                    raise ValueError(
                        "Replacement value must be a list with the same length as the slice"
                    )
                for idx in range(self._length):
                    self._distributions[idx] = Distribution(value[idx])
                return

            # Check there are enough values
            if not isinstance(value, list) or len(value) != len(
                range(*index.indices(self._length))
            ):
                raise ValueError(
                    "Replacement value must be a list with the same length as the slice"
                )

            idxs = np.arange(
                index.start if index.start is not None else 0,
                index.stop if index.stop is not None else self._length,
                index.stp if index.step is not None else 1,
            )
            for jdx, idx in enumerate(idxs):
                self._distributions[idx] = Distribution(value[jdx])

        else:
            self._distributions[index] = Distribution(*value)

    @property
    def mean(self):
        return np.asarray([distribution[0] for distribution in self._distributions])

    @property
    def std(self):
        return np.asarray([distribution[1] for distribution in self._distributions])

    def __repr__(self):
        return f"DistributionContainer\n\t{self._distributions.__repr__()}"

    def get_distributions(self):
        """Returns a copy of the distributions."""
        return self._distributions.copy()

    def freeze(self):
        """Freeze all the distributions by setting its standard deviation to 0."""
        idxs = np.arange(0, len(self))
        for idx in idxs:
            self._distributions[idx].freeze()

    def thaw(self):
        """Thaw all the distributions by setting its standard deviation back to a value."""
        idxs = np.arange(0, len(self))
        for idx in idxs:
            self._distributions[idx].thaw()

    @property
    def mean(self):
        return np.asarray([dist.mean for dist in self])

    @property
    def std(self):
        return np.asarray([dist.std for dist in self])

    def sample(self):
        return np.asarray([dist.sample() for dist in self._distributions])

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(dict):
        return DistributionsContainer(
            [Distribution(m, s) for m, s in zip(dict["mean"], dict["std"])]
        )
