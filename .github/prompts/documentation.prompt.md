---
description: "Use when documenting Python code -- specifically writing 
              docstrings for functions, classes, and modules."
---

# Writing Documentation Style: Guide for Developers and AI

This style guide provides best practices for writing clear, concise, and 
effective documentation for Python code. It covers guidelines for writing 
docstrings for functions, classes, and modules, as well as tips for maintaining 
consistency and readability in your documentation.

The documentation style is based on NumPy/SciPy docstring conventions. Please
review these before writing documentation:

- NumPy: https://numpydoc.readthedocs.io/en/latest/format.html
- SciPy: https://docs.scipy.org/doc/scipy/dev/missing-bits.html

There are some exceptions to these guidelines:
1. Never use single-line docstrings. Always use multi-line docstrings, even for simple functions.
2. Only include sections ("Parameters", "Returns", etc.) when these are relevant. For example, if the function does not return anything, omit the "Returns" section entirely. 
3. Always add a general description of the function/class/module at the beginning of every docstring.
4. Always add a "Notes" section at the end of every docstring.
5. Never add an "Examples" section to any docstring. This is not necessary for the purposes of this project, and it is better to keep the documentation concise and focused on the most important information.
6. Always use British English spelling: "initialise" instead of "initialize", for example. 

**CRITICAL Requirements for AI:**

1. **General Description:** The opening description line MUST be ONLY "Lorem ipsum." Do NOT add any additional text explaining the function/class. The user will write this. 

2. **Notes Section:** The Notes section must contain ONLY "Lorem ipsum." Do NOT write notes content. The user will write this.

3. **Parameter & Return Descriptions:** Write ONLY the type information. Do NOT include descriptions of what parameters do or what return values represent. The user will provide these descriptions.

### CORRECT Example:
```python
def evaluate(x: float, strength: float, sigma_v: float) -> float:
    """
    Lorem ipsum.

    Parameters
    ----------
    x : float
    strength : float
    sigma_v : float

    Returns
    -------
    float

    Notes
    -----
    Lorem ipsum.
    """
    pass
```

### INCORRECT Example (DO NOT DO THIS):
```python
def evaluate(x: float, strength: float, sigma_v: float) -> float:
    """
    Evaluates the profile at specified wavelengths.

    Computes the Gaussian profile value for the given wavelength.

    Parameters
    ----------
    x : float
        Wavelength at which to evaluate.
    strength : float
        Line strength parameter.
    sigma_v : float
        Velocity dispersion.

    Returns
    -------
    float
        Profile value at the wavelength.

    Notes
    -----
    This function is optimised for performance.
    """
    pass
```

## Pydantic's `@validate_function` Decorator

In the event that a function/method is wrapped by Pydantic's `@validate_function` decorator, the beginning of the docstring should show this clearly at the very top of the docstring. Follow these rules:

- For a function: ** PYDANTIC VALIDATED FUNCTION **
- For a method: ** PYDANTIC VALIDATED METHOD **
- For a class initialiser: ** PYDANTIC VALIDATED CLASS INITIALISER **

Follow these by a blank line, and then the general description of the function/class/module. For AI, remember that the general description should be "Lorem ipsum" as a placeholder, and the user will write the actual description.

## quasar-typing's Custom Type Annotations

Many Pydantic type-validated functions/methods use custom type annotations from quasar-typing to specify the expected types of parameters and return values. When you encounter these custom type annotations, use each type's docstring.

For example:

```python
from pydantic import validate_function
from quasar_typing.numpy import FloatVector

@validate_function
def wrong(v: FloatVector) -> None:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Parameters
    ----------
    v : FloatVector
    """
    pass

@validate_function
def correct(v: FloatVector) -> None:
    """
    ** PYDANTIC VALIDATED FUNCTION **

    Parameters
    ----------
    v : 1D numpy.array of floats
    """
    pass
```

## Numba's `@njit` Decorator

In the event that a function is wrapped by Numba's `@njit` decorator, the beginning of the docstring should show this clearly at the very top of the docstring. Follow these rules:

- `@njit`: ** NUMBA OPTIMISED FUNCTION **
- `@njit(parallel=True)`: ** NUMBA OPTIMISED FUNCTION (PARALLEL) **
- `@njit(fastmath=True)`: ** NUMBA OPTIMISED FUNCTION (FASTMATH) **
- `@njit(parallel=True, fastmath=True)`: ** NUMBA OPTIMISED FUNCTION (PARALLEL, FASTMATH) **

## functools' `@lru_cache` and `@cache` Decorators

In the event that a function.method is wrapped by `functools`' `@lru_cache` or `@cache` decorators, the beginning of the docstring should show this clearly at the very top of the docstring. Follow these rules:

- `@lru_cache`: ** LRU-CACHED FUNCTION **
- `@cache`: ** CACHED FUNCTION **