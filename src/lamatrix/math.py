from .combine import JointModel, CrosstermModel
from .model import Model

__all__ = ["MathMixins"]


class MathMixins:
    def __add__(self, other):
        if (self.arg_names == {}) & (other.arg_names == {}):
            raise ValueError("Can not add multiple offset terms.")
        if isinstance(other, JointModel):
            return JointModel(self, *other.models)
        if isinstance(other, Model):
            return JointModel(self, other)
        else:
            raise ValueError("Can only combine `Model` objects.")

    def __mul__(self, other):
        if (self.arg_names == {}) & (other.arg_names == {}):
            return other
        if (self.arg_names == {}) ^ (other.arg_names == {}):
            if self.arg_names == {}:
                return other
            elif other.arg_names == {}:
                return self
        if isinstance(other, CrosstermModel):
            return CrosstermModel(self, *other.models)
        elif isinstance(other, JointModel):
            return JointModel(*[CrosstermModel(self, g) for g in other.models])
        elif isinstance(other, Model):
            return CrosstermModel(self, other)
        else:
            raise ValueError("Can only combine `Model` objects.")
