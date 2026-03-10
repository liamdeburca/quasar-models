from numpy import ndarray

def get_x_grid(edge: float, sigma_res: float) -> ndarray[float]:
    """
    Get the logarithmic wavelength grid for the Balmer continuum and series 
    calculations.

    Notes
    -----
    Thre grid is designed to cover the wavelength range from ~1000 Å to ~4000 Å
    with a resolution defined by `sigma_res`. This is achieved by using a ratio
    between the largest and smallest wavelengths of 4. To remain insensitive to 
    the choice of wavelength units, the reference wavelength is set to 1.1 times
    the Balmer edge, or ~4000 Å.

    Parameters
    ----------
    edge : float
        The Balmer edge wavelength in the same units as the desired output grid.
    sigma_res : float
        The desired resolution in logarithmic space (dln(lambda)).

    Returns
    -------
    x_grid : ndarray[float]
        The computed wavelength grid.
    """
    from numpy import ndarray, ceil, log, arange

    n: int = int(ceil(log(4) / log(1 + sigma_res)))
    edges: ndarray[float] = 1.1*edge / (1 + sigma_res)**arange(n)[::-1]

    return edges[:-1] * (1 + sigma_res / 2)

def find_bracket(val: float, ref: ndarray[float]) -> int:
    """
    Find the index of the largest element in `ref` that is less than or equal
    to `val`.

    Parameters
    ----------
    val : float
        The value to bracket.
    ref : ndarray[float]
        The reference array, must be sorted in ascending order.

    Returns
    -------
    idx : int
        The index of the largest element in `ref` that is less than or equal to
        `val`.
    """
    from numpy import searchsorted
    return searchsorted(ref, val, side='right') - 1