from .combine import StackedIndependentGenerator, StackedDependentGenerator
from .generator import Generator

__all__ = ["MathMixins"]


class MathMixins:
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
