# Standard library
import os  # noqa
import platform
from datetime import datetime

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

__version__ = "0.1.0"


def _META_DATA():
    """
    Returns metadata information to a dictionary.
    """
    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tool_name": "lamatrix",
        "tool_version": f"{__version__}",
        "operating_system": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
    }
    return metadata


# Standard library
import logging  # noqa: E402

# This library lets us have log messages with syntax highlighting
from rich.logging import RichHandler  # noqa: E402

log = logging.getLogger("tesswcs")
log.addHandler(RichHandler(markup=True))

import json

import numpy as np

from .bound import *
from .combine import *  # noqa: E402, F401
from .models.astrophysical import *  # noqa: E402, F401
from .models.gaussian import *  # noqa: E402, F401
from .models.simple import *  # noqa: E402, F401
from .models.spline import *  # noqa: E402, F401


def _load_from_dict(dict):
    new = globals()[dict["object_type"]](**dict["initializing_kwargs"])
    _ = [setattr(new, key, value) for key, value in dict["fit_results"].items()]
    return new


def load(filename):
    def process(arg):
        if isinstance(arg, dict):
            return {key: process(item) for key, item in arg.items()}
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
    with open(filename, "r") as json_file:
        data_loaded = json.load(json_file)
    data_loaded = {key: process(item) for key, item in data_loaded.items()}
    if "generators" in data_loaded.keys():
        generators = [
            _load_from_dict(item) for _, item in data_loaded["generators"].items()
        ]
        new = globals()[data_loaded["object_type"]](*generators)
        _ = [
            setattr(new, key, value)
            for key, value in data_loaded["fit_results"].items()
        ]
        return new
    return _load_from_dict(data_loaded)
