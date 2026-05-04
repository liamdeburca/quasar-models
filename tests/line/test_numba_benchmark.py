"""
Benchmark comparing Numba-optimized vs pure NumPy implementations.

This test measures:
1. Initial compilation time for Numba
2. Per-call speedup after compilation
3. Amortized cost over multiple calls
"""

import time
import numpy as np
from numpy import zeros_like, float64, exp, hypot

# Pure NumPy versions (no Numba)

def evaluate_python(
    x: float | np.ndarray,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    gauss_amp: float = 1 / (2 * np.pi)**0.5,
) -> float | np.ndarray:
    """Pure NumPy implementation of evaluate()."""
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma


def evaluate_sparse_python(
    x: np.ndarray,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    n_sigmas: float = 3.0,
    gauss_amp: float = 1 / (2 * np.pi)**0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy implementation of evaluate_sparse()."""
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)

    mask = (mean - n_sigmas * sigma <= x) & (x <= mean + n_sigmas * sigma)

    inv_sigma = 1.0 / sigma
    z = (x[mask] - mean) * inv_sigma
    y = gauss_amp * strength * exp(-0.5 * z * z) * inv_sigma

    return mask, y


def fit_deriv_numba_python(
    x: np.ndarray,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float,
    sigma_res: float,
    gauss_amp: float = 1 / (2 * np.pi)**0.5,   
    fixed_strength: bool = True,
    fixed_sigma_v: bool = True,
    fixed_v_off: bool = True,
) -> list[np.ndarray]:
    """Pure NumPy implementation of fit_deriv_numba()."""
    df_dstrength = zeros_like(x, dtype=float64)
    df_dsigma_v  = zeros_like(x, dtype=float64)
    df_dv_off    = zeros_like(x, dtype=float64)

    if not (fixed_strength and fixed_sigma_v and fixed_v_off):
        mean = wave * (1 + v_off)
        sigma_tot_sq = sigma_v * sigma_v + sigma_res * sigma_res
        sigma = mean * sigma_tot_sq**0.5
        inv_sigma = 1.0 / sigma
        z = (x - mean) * inv_sigma
        z_sq = z * z
        amp = gauss_amp * inv_sigma
        _f = amp * exp(-0.5 * z_sq)
        f = strength * _f

        if not fixed_strength:
            df_dstrength[:] = _f

        if not fixed_sigma_v:
            df_dsigma_v[:] = f * (z_sq - 1) * sigma_v / sigma_tot_sq

        if not fixed_v_off:
            df_dv_off[:] = f * (z * x * inv_sigma - 1) / (1 + v_off)

    return [df_dstrength, df_dsigma_v, df_dv_off]


def prime_python(
    x: float | np.ndarray,
    strength: float,
    sigma_v: float,
    v_off: float,
    wave: float = 1.0,
    sigma_res: float = 1.0,
    gauss_amp: float = 1 / (2 * np.pi)**0.5,
) -> float | np.ndarray:
    """Pure NumPy implementation of prime()."""
    mean = wave * (1 + v_off)
    sigma = mean * hypot(sigma_v, sigma_res)
    inv_sigma = 1.0 / sigma
    z = (x - mean) * inv_sigma
    return -z * inv_sigma * evaluate_python(
        x, 
        strength, sigma_v, v_off, 
        wave=wave, 
        sigma_res=sigma_res, 
        gauss_amp=gauss_amp,
    )


def _run_benchmark(
    func_name: str,
    func_python,
    func_numba,
    test_args: dict,
    n_iterations: int = 100,
) -> dict:
    """
    Generic benchmark runner for any function.
    
    Returns metrics dict with compilation time, speedup, etc.
    """
    import time as time_module
    
    # ========== Measure pure NumPy version (baseline) ==========
    times_python = []
    for _ in range(n_iterations):
        start = time_module.perf_counter()
        result_python = func_python(**test_args)
        end = time_module.perf_counter()
        times_python.append(end - start)
    
    time_python_avg = np.mean(times_python[10:])  # Exclude warmup
    time_python_std = np.std(times_python[10:])
    
    # ========== Measure Numba version (with compilation) ==========
    # First call (triggers compilation if not cached)
    compile_start = time_module.perf_counter()
    result_numba = func_numba(**test_args)
    compile_end = time_module.perf_counter()
    
    time_first_call = compile_end - compile_start
    
    # Subsequent calls (post-compilation)
    times_numba = []
    for _ in range(n_iterations):
        start = time_module.perf_counter()
        result_numba = func_numba(**test_args)
        end = time_module.perf_counter()
        times_numba.append(end - start)
    
    time_numba_avg = np.mean(times_numba[10:])  # Exclude warmup
    time_numba_std = np.std(times_numba[10:])
    
    # ========== Calculate metrics ==========
    speedup_per_call = time_python_avg / time_numba_avg
    
    # Calculate break-even point
    if speedup_per_call > 1:
        time_saved_per_call = time_python_avg - time_numba_avg
        breakeven_calls = time_first_call / time_saved_per_call
    else:
        breakeven_calls = float('inf')
    
    return {
        'func_name': func_name,
        'compilation_time': time_first_call,
        'python_avg': time_python_avg,
        'python_std': time_python_std,
        'numba_avg': time_numba_avg,
        'numba_std': time_numba_std,
        'speedup': speedup_per_call,
        'breakeven_calls': breakeven_calls,
        'result_python': result_python,
        'result_numba': result_numba,
    }


def _print_benchmark_results(metrics: dict):
    """Print formatted benchmark results."""
    print("\n" + "="*70)
    print(f"NUMBA BENCHMARK: {metrics['func_name']}()")
    print("="*70)
    print(f"Test iterations: 100")
    print()
    
    print("COMPILATION PHASE")
    print("-" * 70)
    print(f"First call (compilation):      {metrics['compilation_time']*1000:8.2f} ms")
    print()
    
    print("EXECUTION PERFORMANCE (after warmup)")
    print("-" * 70)
    print(f"Pure NumPy avg:                {metrics['python_avg']*1000:8.3f} ± {metrics['python_std']*1000:6.3f} ms")
    print(f"Numba JIT avg:                 {metrics['numba_avg']*1000:8.3f} ± {metrics['numba_std']*1000:6.3f} ms")
    print()
    
    print("SPEEDUP ANALYSIS")
    print("-" * 70)
    print(f"Per-call speedup:              {metrics['speedup']:8.2f}x")
    print(f"Break-even point:              {metrics['breakeven_calls']:8.0f} calls")
    
    if metrics['speedup'] > 1:
        time_saved = (metrics['python_avg'] - metrics['numba_avg']) * 1000
        print(f"Time saved per call:           {time_saved:8.3f} ms")
    print()
    
    # Cumulative analysis
    print("CUMULATIVE ANALYSIS")
    print("-" * 70)
    for n_calls in [1, 10, 50, 100, 500, 1000]:
        total_python = metrics['python_avg'] * n_calls
        total_numba = metrics['compilation_time'] + (metrics['numba_avg'] * n_calls)
        savings = total_python - total_numba
        print(f"{n_calls:4d} calls: NumPy={total_python*1000:8.1f}ms | Numba={total_numba*1000:8.1f}ms | "
              f"Savings={savings*1000:7.1f}ms ({savings/total_python*100:6.1f}%)")
    print()
    
    print("VERDICT")
    print("-" * 70)
    if metrics['breakeven_calls'] < 10:
        print(f"✓ Numba is WORTHWHILE (breaks even in {metrics['breakeven_calls']:.0f} calls)")
        verdict = "KEEP"
    elif metrics['breakeven_calls'] < 100:
        print(f"⚠ Numba is MARGINAL (breaks even in {metrics['breakeven_calls']:.0f} calls)")
        verdict = "MAYBE"
    else:
        print(f"✗ Numba is NOT WORTH IT (breaks even in {metrics['breakeven_calls']:.0f} calls)")
        verdict = "REMOVE"
    print("="*70)
    return verdict


def test_evaluate_numba_speedup():
    """Benchmark evaluate() function."""
    sigma_res: float = 2.3e-4
    x = 1400 * (1 + sigma_res)**np.arange(1000)
    
    params = {
        'x': x,
        'strength': 1.0,
        'sigma_v': 0.01,
        'v_off': 0.0,
        'wave': 1549.0,
        'sigma_res': sigma_res,
    }
    
    from quasar_models.line.evaluation import evaluate
    
    metrics = _run_benchmark(
        'evaluate',
        evaluate_python,
        evaluate,
        params,
    )
    
    _print_benchmark_results(metrics)
    
    # Verify correctness
    np.testing.assert_allclose(
        metrics['result_python'],
        metrics['result_numba'],
        rtol=1e-10,
    )


def test_evaluate_sparse_numba_speedup():
    """Benchmark evaluate_sparse() function."""
    sigma_res: float = 2.3e-4
    x = 1400 * (1 + sigma_res)**np.arange(1000)
    
    params = {
        'x': x,
        'strength': 1.0,
        'sigma_v': 0.01,
        'v_off': 0.0,
        'wave': 1549.0,
        'sigma_res': sigma_res,
    }
    
    from quasar_models.line.evaluation import evaluate_sparse
    
    metrics = _run_benchmark(
        'evaluate_sparse',
        evaluate_sparse_python,
        evaluate_sparse,
        params,
    )
    
    _print_benchmark_results(metrics)
    
    # Verify correctness
    np.testing.assert_array_equal(
        metrics['result_python'][0],
        metrics['result_numba'][0],
    )
    np.testing.assert_allclose(
        metrics['result_python'][1],
        metrics['result_numba'][1],
        rtol=1e-10,
    )


def test_fit_deriv_numba_speedup():
    """Benchmark fit_deriv_numba() function."""
    sigma_res: float = 2.3e-4
    x = 1400 * (1 + sigma_res)**np.arange(1000)
    
    params = {
        'x': x,
        'strength': 1.0,
        'sigma_v': 0.01,
        'v_off': 0.0,
        'wave': 1549.0,
        'sigma_res': sigma_res,
        'fixed_strength': False,
        'fixed_sigma_v': False,
        'fixed_v_off': False,
    }
    
    from quasar_models.line.evaluation import fit_deriv_numba
    
    metrics = _run_benchmark(
        'fit_deriv_numba',
        fit_deriv_numba_python,
        fit_deriv_numba,
        params,
    )
    
    _print_benchmark_results(metrics)
    
    # Verify correctness
    for i in range(3):
        np.testing.assert_allclose(
            metrics['result_python'][i],
            metrics['result_numba'][i],
            rtol=1e-10,
        )


def test_prime_numba_speedup():
    """Benchmark prime() function."""
    sigma_res: float = 2.3e-4
    x = 1400 * (1 + sigma_res)**np.arange(1000)
    
    params = {
        'x': x,
        'strength': 1.0,
        'sigma_v': 0.01,
        'v_off': 0.0,
        'wave': 1549.0,
        'sigma_res': sigma_res,
    }
    
    from quasar_models.line.evaluation import prime
    
    metrics = _run_benchmark(
        'prime',
        prime_python,
        prime,
        params,
    )
    
    _print_benchmark_results(metrics)
    
    # Verify correctness
    np.testing.assert_allclose(
        metrics['result_python'],
        metrics['result_numba'],
        rtol=1e-10,
    )


def test_all_numba_functions():
    """Run all benchmarks and print summary."""
    print("\n\n" + "#"*70)
    print("# NUMBA BENCHMARK SUITE: LINE MODULE")
    print("#"*70)
    
    results = []
    
    for test_func in [
        test_evaluate_numba_speedup,
        test_evaluate_sparse_numba_speedup,
        test_fit_deriv_numba_speedup,
        test_prime_numba_speedup,
    ]:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ Error in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n\n" + "#"*70)
    print("# SUMMARY")
    print("#"*70)
    print("All benchmarks completed. Review verdicts above.")
    print("#"*70 + "\n")


if __name__ == '__main__':
    test_all_numba_functions()
