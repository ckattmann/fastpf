"""
Performance-first Power Flow implemented in Python with AOT Compilation

.. include:: ../docs/introduction.md
"""

__version__ = "0.0.8"

from .log import logger
from .log import set_loglevel

from . import powerflow_methods

# First try finding powerflow_methods_cc
# no: Try compiling powerflow_methods_cc
# Also no: Use dynamic @jit

try:
    from . import powerflow_methods_cc
except ImportError:
    print("Attempting to compile powerflow_methods_cc...")
    try:
        powerflow_methods.compile()
    except:
        logger.warning("Could not compile powerflow_methods_")
    from . import powerflow_methods_cc

# from .data.parse_matpower_casefile import parse_matpower_casefile
from .data import testgrids
from .data import testloads

from .validation import validate_grid
from .process_grid import process_grid

# from .process_grid import integrate_slacks_for_Yident

from .powerflow import ybusjacobi
from .powerflow import ybusgaussseidel
from .powerflow import ybusnewton
from .powerflow import zbusjacobi
from .powerflow import compare_methods

from .plotting.plot_grid import plot_grid

# from .plotting import *

set_loglevel("INFO")
