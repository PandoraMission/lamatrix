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
from .models.sip import *  # noqa: E402, F401, F403
from .models.spline import *  # noqa: E402, F401, F403


def parent_paths_with_models_deepest_first(tree):
    """Return paths to dicts that have 'models', ordered deepest-first."""
    out = []

    def rec(node, path):
        if not isinstance(node, dict):
            return
        if "models" in node:
            children = node["models"]
            if isinstance(children, dict):
                for k, child in children.items():
                    rec(child, path + ("models", k))
            elif isinstance(children, list):
                for i, child in enumerate(children):
                    rec(child, path + ("models", i))
            # After visiting all children, append current path
            out.append(tuple(path))

    rec(tree, ())
    return out


def get_by_path(tree, path):
    """Follow a tuple/list path into a nested dict/list and return the value."""
    node = tree
    for key in path:
        node = node[key]
    return node


def set_by_path(tree, path, value):
    """Set a nested dict/list element given a tuple path."""
    if not path:  # root replacement
        raise ValueError("Can't replace the root dictionary this way")

    # walk to the parent of the target
    node = tree
    for key in path[:-1]:
        node = node[key]

    # update the final key
    node[path[-1]] = value


def _load_from_dict_single_model(input_dict):
    new = globals()[input_dict["object_type"]](
        **input_dict["initializing_kwargs"],
        posteriors=DistributionsContainer.from_dict(input_dict["posteriors"]),
        priors=DistributionsContainer.from_dict(input_dict["priors"]),
    )
    return new


def _load_from_dict_joint_model(input_dict):
    new = globals()[input_dict["object_type"]](
        *[
            _load_from_dict_single_model(m[1]) if isinstance(m[1], dict) else m[1]
            for m in input_dict["models"].items()
        ],
        **input_dict["initializing_kwargs"],
        posteriors=DistributionsContainer.from_dict(input_dict["posteriors"]),
    )
    return new


def _load_from_dict(dict):
    if "models" in dict.keys():
        return _load_from_dict_joint_model(dict)
    else:
        return _load_from_dict_single_model(dict)


def load(filename):
    def process(arg):
        if isinstance(arg, dict):
            return {key: process(item) for key, item in arg.items()}
        if arg is None:
            return None
        elif isinstance(arg, str):
            if arg == "Infinity":
                return np.inf
            return arg
        elif isinstance(arg, (int, float, tuple)):
            return arg
        elif isinstance(arg, list):
            return np.asarray(arg)

    with open(filename, "r") as json_file:
        data_loaded = json.load(json_file)
    data_loaded = {key: process(item) for key, item in data_loaded.items()}
    paths = parent_paths_with_models_deepest_first(data_loaded)
    for path in paths:
        if path != ():
            set_by_path(
                data_loaded, path, _load_from_dict(get_by_path(data_loaded, path))
            )
    new = _load_from_dict(data_loaded)
    return new
