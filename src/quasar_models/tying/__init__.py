"""
Tying model parameters is extremely useful when fitting models to data. However,
using simple functions, such as lambda expressions, to tie parameters is not 
well-suited for fitting routines as they cannot be pickled.

This submodule offers utilities for tying parameters in an efficient and 
reliable manner.
"""
__all__ = [
    "IdenticalTie",
]

from .identical_tie import IdenticalTie