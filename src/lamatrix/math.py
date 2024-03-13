from .combined import VStackedGenerator, CombinedGenerator
from .generator import Generator

__all__ = ["MathMixins"]


class MathMixins:
    def __add__(self, other):
        if isinstance(other, Generator):
            return VStackedGenerator(self, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")
        
    def __mul__(self, other):
        if isinstance(other, Generator):
            return CombinedGenerator(self, other)
        else:
            raise ValueError("Can only combine `Generator` objects.")
