"""
Verification of wavelength-space Gaussian model derivatives.

Tests the analytical derivatives against numerical finite-difference derivatives.
This is a self-contained test that does NOT depend on conftest.py fixtures.
"""

import math

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

# ============================================================================
# Constants
# ============================================================================

GAUSS_AMP = 1 / math.sqrt(2 * math.pi)
FWHM_TO_SIGMA = 2 * math.sqrt(2 * math.log(2))
SIGMA_TO_FWHM = 1 / FWHM_TO_SIGMA
C_KMS = 299792.458
INV_C_KMS = 1.0 / C_KMS

dx = 0.25
x = 6000 + dx * np.arange(4000)

# ============================================================================
# Pure Python Implementation of Wavelength-Space Gaussian
# ============================================================================

def evaluate_x(
    strength,
    fwhm_v,      # km/s
    v_off,       # km/s
    wave,
):
    """
    Evaluate the wavelength-space Gaussian model.
    
    Parameters
    ----------
    strength : float
        Line strength (flux integral)
    fwhm_v : float
        Intrinsic FWHM in km/s
    v_off : float
        Velocity offset in km/s
    wave : float
        Rest wavelength
        
    Returns
    -------
    y : ndarray
        Evaluated flux
    """
    sigma_v_c = fwhm_v * FWHM_TO_SIGMA / C_KMS  # Velocity dispersion in c
    mean = wave * (1.0 + v_off / C_KMS)          # Observed wavelength
    sigma_lambda_v = mean * sigma_v_c
    sigma_lambda = math.hypot(sigma_lambda_v, dx)
    
    norm = strength * GAUSS_AMP / sigma_lambda

    z = (x - mean) / sigma_lambda
    z_sq = z * z
    exp_term = np.exp(-0.5 * z_sq)

    return norm * exp_term

def fit_deriv_x_strength(
    strength,
    fwhm_v,
    v_off,
    wave,
):
    return evaluate_x(1.0, fwhm_v, v_off, wave)


def fit_deriv_x_fwhm_v(
    strength,
    fwhm_v,
    v_off,
    wave,
):
    sigma_v_c = fwhm_v * FWHM_TO_SIGMA * INV_C_KMS
    mean = wave * (1.0 + v_off * INV_C_KMS)
    sigma_x = mean * sigma_v_c
    sigma = math.hypot(sigma_x, dx)
    inv_sigma = 1.0 / sigma
    norm = strength * GAUSS_AMP * inv_sigma

    k = sigma_x * mean * FWHM_TO_SIGMA * INV_C_KMS * inv_sigma**2

    z = (x - mean) * inv_sigma
    z_sq = z * z
    exp_term = np.exp(-0.5 * z_sq)

    return norm * exp_term * (z_sq - 1) * k


def fit_deriv_x_v_off(
    strength,
    fwhm_v,
    v_off,
    wave,
):
    sigma_v = fwhm_v * FWHM_TO_SIGMA
    sigma_v_c = sigma_v * INV_C_KMS
    mean = wave * (1.0 + v_off * INV_C_KMS)
    sigma_x = mean * sigma_v_c
    inv_sigma = 1.0 / math.hypot(sigma_x, dx)
    norm = strength * GAUSS_AMP * inv_sigma

    l = norm * wave * INV_C_KMS * inv_sigma
    m = sigma_x * sigma_v * INV_C_KMS * inv_sigma

    z = (x - mean) * inv_sigma
    z_sq = z * z
    exp_term = np.exp(-0.5 * z_sq)

    return l * exp_term * (z + m * (z_sq - 1))


def fit_deriv_x_all(
    strength,
    fwhm_v,
    v_off,
    wave,
):
    """
    Compute all three partial derivatives.
    
    Returns
    -------
    derivs : ndarray, shape (3, len(x))
        [df/dstrength, df/dfwhm_v, df/dv_off]
    """
    derivs = np.zeros((3, len(x)))

    derivs[0,:] = fit_deriv_x_strength(strength, fwhm_v, v_off, wave)
    derivs[1,:] = fit_deriv_x_fwhm_v(strength, fwhm_v, v_off, wave)
    derivs[2,:] = fit_deriv_x_v_off(strength, fwhm_v, v_off, wave)

    return derivs



# ============================================================================
# Numerical Derivatives (Finite Differences)
# ============================================================================

def numerical_derivative(
    strength,
    fwhm_v,
    v_off,
    wave,
    epsilon=1e-6,
    bounds=None,
):
    """
    Compute numerical derivatives using central differences.
    
    Parameters
    ----------
    epsilon : float
        Step size for finite differences
    bounds : dict
        Parameter bounds for clipping: {'strength': (min, max), ...}
    """
    if bounds is None:
        bounds = {
            'strength': (0.001, 1000.0),
            'fwhm_v': (10.0, 20000.0),
            'v_off': (-10000.0, 10000.0),
        }
    
    derivs = []
    
    # Derivative w.r.t. strength
    p_left = max(bounds['strength'][0], strength - epsilon)
    p_right = min(bounds['strength'][1], strength + epsilon)
    dp = p_right - p_left
    y_left = evaluate_x(p_left, fwhm_v, v_off, wave)
    y_right = evaluate_x(p_right, fwhm_v, v_off, wave)
    derivs.append((y_right - y_left) / dp)
    
    # Derivative w.r.t. fwhm_v
    p_left = max(bounds['fwhm_v'][0], fwhm_v - epsilon)
    p_right = min(bounds['fwhm_v'][1], fwhm_v + epsilon)
    dp = p_right - p_left
    y_left = evaluate_x(strength, p_left, v_off, wave)
    y_right = evaluate_x(strength, p_right, v_off, wave)
    derivs.append((y_right - y_left) / dp)
    
    # Derivative w.r.t. v_off
    p_left = max(bounds['v_off'][0], v_off - epsilon)
    p_right = min(bounds['v_off'][1], v_off + epsilon)
    dp = p_right - p_left
    y_left = evaluate_x(strength, fwhm_v, p_left, wave)
    y_right = evaluate_x(strength, fwhm_v, p_right, wave)
    derivs.append((y_right - y_left) / dp)
    
    return np.stack(derivs, axis=0)


# ============================================================================
# Test Suite
# ============================================================================

def test_gaussian_x_derivatives(
    n_tests=50,
    epsilon=1e-6,
    rtol=1e-3,
    atol=1e-6,
    verbose=True,
):
    """
    Run N tests verifying analytical vs. numerical derivatives.
    
    Parameters
    ----------
    n_tests : int
        Number of random parameter combinations to test
    n_wavelengths : int
        Number of wavelength points
    epsilon : float
        Step size for numerical derivatives
    rtol : float
        Relative tolerance for allclose
    atol : float
        Absolute tolerance for allclose
    verbose : bool
        Print results for each test
    """
    
    # Setup
    rng = default_rng(seed=42)
        
    # Rest wavelength for test line (H-alpha equivalent)
    wave_rest = 6548.0
    
    # Parameter bounds
    strength_bounds = (0.001, 100.0)
    fwhm_v_bounds = (100.0, 10000.0)
    v_off_bounds = (-5000.0, 5000.0)
    
    bounds = {
        'strength': strength_bounds,
        'fwhm_v': fwhm_v_bounds,
        'v_off': v_off_bounds,
    }
    
    # Run tests
    n_pass = 0
    n_fail = 0
    
    for test_idx in tqdm(range(n_tests), leave=False):
        strength = rng.uniform(*strength_bounds)
        fwhm_v = rng.uniform(*fwhm_v_bounds)
        v_off = rng.uniform(*v_off_bounds)
        
        # Compute derivatives
        df_analytical = fit_deriv_x_all(strength, fwhm_v, v_off, wave_rest)
        df_numerical = numerical_derivative(
            strength, fwhm_v, v_off, wave_rest,
            epsilon=epsilon, bounds=bounds
        )
        
        # Check each derivative
        test_pass = True
        for i, name in enumerate(['strength', 'fwhm_v', 'v_off']):
            match = np.allclose(df_analytical[i], df_numerical[i], rtol=rtol, atol=atol)
            if not match:
                test_pass = False
                n_fail += 1
                if verbose:
                    print(f"  ✗ {name:12s} MISMATCH")
            else:
                if verbose:
                    print(f"  ✓ {name:12s} OK")
        
        if test_pass:
            n_pass += 1
        
        if verbose:
            print(
                f"Test {test_idx+1:3d}: strength={strength:8.2f}, "
                f"fwhm_v={fwhm_v:8.1f}, v_off={v_off:8.1f}"
            )
    
    # Summary
    print("\n" + "="*70)
    print(f"Results: {n_pass} passed, {n_fail} failed out of {n_tests} tests")
    if n_fail == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {n_fail} TESTS FAILED")
    print("="*70)
    
    return n_fail == 0


# ============================================================================
# Individual Derivative Tests
# ============================================================================

def test_derivative_strength():
    """Test df/dstrength derivative."""
    print("\n" + "="*70)
    print("Testing df/dstrength")
    print("="*70)
    
    strength = 10.0
    fwhm_v = 1000.0
    v_off = 0.0
    wave = 6548.0
    
    df_analytical = fit_deriv_x_all(strength, fwhm_v, v_off, wave)
    df_numerical = numerical_derivative(strength, fwhm_v, v_off, wave, epsilon=1e-6)
    
    print(f"\nAnalytical df/dstrength: {df_analytical[0]}")
    print(f"Numerical df/dstrength:  {df_numerical[0]}")
    print(f"Match: {np.allclose(df_analytical[0], df_numerical[0], rtol=1e-3, atol=1e-6)}")


def test_derivative_fwhm_v():
    """Test df/dfwhm_v derivative."""
    print("\n" + "="*70)
    print("Testing df/dfwhm_v")
    print("="*70)
    
    strength = 10.0
    fwhm_v = 1000.0
    v_off = 0.0
    wave = 6548.0
    
    df_analytical = fit_deriv_x_all(strength, fwhm_v, v_off, wave)
    df_numerical = numerical_derivative(strength, fwhm_v, v_off, wave, epsilon=1e-6)
    
    print(f"\nAnalytical df/dfwhm_v: {df_analytical[1]}")
    print(f"Numerical df/dfwhm_v:  {df_numerical[1]}")
    print(f"Match: {np.allclose(df_analytical[1], df_numerical[1], rtol=1e-3, atol=1e-6)}")


def test_derivative_v_off():
    """Test df/dv_off derivative - debug both formulas."""
    print("\n" + "="*70)
    print("Testing df/dv_off (Debugging)")
    print("="*70)
    
    strength = 10.0
    fwhm_v = 1000.0
    v_off = 0.0
    wave = 6548.0
        
    df_numerical = numerical_derivative(strength, fwhm_v, v_off, wave, epsilon=1e-6)
    df_analytical = fit_deriv_x_all(strength, fwhm_v, v_off, wave)
        
    print(f"\nAnalytical df/dv_off: {df_analytical[2]}")
    print(f"Numerical df/dv_off:  {df_numerical[2]}")
    print(f"Match: {np.allclose(df_analytical[2], df_numerical[2], rtol=1e-3, atol=1e-6)}")


if __name__ == "__main__":
    import sys
    
    # Run comprehensive test suite
    success = test_gaussian_x_derivatives(
        n_tests=100_000,
        epsilon=1e-6,
        rtol=1e-3,
        atol=1e-6,
        verbose=False,
    )
    
    sys.exit(0 if success else 1)
