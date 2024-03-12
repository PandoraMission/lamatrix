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

from .generator import *  # noqa: E402, F401
