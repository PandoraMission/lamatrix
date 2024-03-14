# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

version = "0.1.0"

# Standard library
import logging  # noqa: E402

# This library lets us have log messages with syntax highlighting
from rich.logging import RichHandler  # noqa: E402

log = logging.getLogger("tesswcs")
log.addHandler(RichHandler(markup=True))

from .combined import *  # noqa: E402, F401
from .models.astrophysical import *  # noqa: E402, F401
from .models.gaussian import *  # noqa: E402, F401
from .models.simple import *  # noqa: E402, F401
from .models.spline import *  # noqa: E402, F401

import json
import numpy as np

def load(filename):
    def process(arg):
        if isinstance(arg, dict):
            return {key:process(item) for key, item in arg.items()}
        if arg is None:
            return None
        elif isinstance(arg, str):
            if arg == "Infinity":
                return np.nan
            return arg
        elif isinstance(arg, (int, float, tuple)):
            return arg
        elif isinstance(arg, list):
            return np.asarray(arg)

    """Load a Generator object"""
    with open(filename, 'r') as json_file:
        data_loaded = json.load(json_file)
    data_loaded = {key:process(item) for key, item in data_loaded.items()}
    new = globals()[data_loaded['object_type']](**data_loaded['initializing_kwargs'])
    _ = [setattr(new, key, value) for key, value in data_loaded['fit_results'].items()]
    return new