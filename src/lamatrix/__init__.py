# Standard library
import os  # noqa
import platform
from datetime import datetime

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from importlib.metadata import PackageNotFoundError, version  # noqa


def get_version():
    try:
        return version("lamatrix")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


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

log = logging.getLogger("lamatrix")
log.addHandler(RichHandler(markup=True))

import json  # noqa: E402

import numpy as np  # noqa: E402

from .bounded import *  # noqa: E402, F401, F403
from .combine import *  # noqa: E402, F401, F403
from .distributions import Distribution  # noqa: E402, F401, F403
from .distributions import DistributionsContainer  # noqa: E402
from .model import *  # noqa: E402, F401, F403
from .models.astrophysical import *  # noqa: E402, F401, F403
from .models.gaussian import *  # noqa: E402, F401, F403
from .models.simple import *  # noqa: E402, F401, F403
from .models.spline import *  # noqa: E402, F401, F403
from .models.sip import *  # noqa: E402, F401, F403


def _load_from_dict(dict):
    new = globals()[dict["object_type"]](**dict["initializing_kwargs"])
    new.priors = DistributionsContainer.from_dict(dict["priors"])
    if dict["posteriors"] is not None:
        new.posteriors = DistributionsContainer.from_dict(dict["posteriors"])
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
    if "models" in data_loaded.keys():
        models = [_load_from_dict(item) for _, item in data_loaded["models"].items()]
        new = globals()[data_loaded["object_type"]](*models)
    else:
        new = _load_from_dict(data_loaded)
    return new
