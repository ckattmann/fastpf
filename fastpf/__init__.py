"""
Performance-first Power Flow implemented in Python with AOT Compilation
"""

__version__ = "0.0.6"

from .log import logger
from .log import set_loglevel

from . import powerflow_methods

try:
    from . import powerflow_methods_cc
except ImportError:
    logger.info("Attempting to compile powerflow_methods_cc...")
    powerflow_methods.compile()
    from . import powerflow_methods_cc

from .data.parse_matpower_casefile import parse_matpower_casefile
from .data import testgrids
from .data import testloads

from .validation import validate_grid
from .process_grid import process_grid
from .process_grid import integrate_slacks_for_Yident

from .powerflow import ybusjacobi
from .powerflow import ybusgaussseidel
from .powerflow import ybusnewton
from .powerflow import zbusjacobi
from .powerflow import compare_methods

from .plotting import *


set_loglevel("INFO")
