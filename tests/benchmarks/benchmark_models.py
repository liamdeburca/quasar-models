"""
Benchmark script for custom Astropy models' evaluate and fit_deriv methods.

Benchmarks the following models:
- PowerLawModel
- GaussianModel
- IronModel
- BalmerModel
- HostGalaxyModel

All benchmarks are performed on a logarithmically binned wavelength grid
with v_res=2.3e-4 (as per QSO spectroscopy standards).
"""

import timeit
import numpy as np
from typing import Callable, NamedTuple
from tqdm import tqdm

from quasar_models import (
    PowerLawModel,
    GaussianModel,
    IronModel,
    BalmerModel,
    HostGalaxyModel,
)
from quasar_models.iron import IronTemplate
from quasar_models.balmer.continuum import BalmerContinuumTemplate
from quasar_models.balmer.series import BalmerSeriesTemplate
from quasar_models.host import HostGalaxyTemplate

from quasar_utils.setup import Info


# Configuration
V_RES = 2.3e-4
N_WAVELENGTHS = 10_000
N: int = 10_000
INFO = Info()

class BenchmarkResult(NamedTuple):
    """Container for benchmark results."""
    model_name: str
    method_name: str
    total_time: float
    mean_time: float
    std_time: float

    @property
    def mean_time_us(self) -> float:
        """Mean time in microseconds."""
        return self.mean_time * 1_000_000

    @property
    def std_time_us(self) -> float:
        """Standard deviation in microseconds."""
        return self.std_time * 1_000_000


def create_wavelength_grid(
    v_res: float = V_RES,
    n_wavelengths: int = N_WAVELENGTHS,
    start_wavelength: float = 1000.0,
) -> np.ndarray:
    """
    Create a logarithmically binned wavelength grid.

    Parameters
    ----------
    v_res : float
        Velocity resolution in units of c.
    n_wavelengths : int
        Number of wavelength points.
    start_wavelength : float
        Starting wavelength in Angstroms.

    Returns
    -------
    np.ndarray
        Logarithmically binned wavelength array.
    """
    return start_wavelength * (1 + v_res) ** np.arange(n_wavelengths)


def benchmark_method(method: Callable) -> tuple[float, float]:
    """
    Benchmark a method with multiple iterations.

    Parameters
    ----------
    method : Callable
        The method to benchmark.

    Returns
    -------
    tuple[float, float]
        (total_time, standard_deviation) in seconds.
    """
    times = []
    for _ in tqdm(range(N), desc='Benchmarking', leave=False):
        t = timeit.timeit(method, number=1)
        times.append(t)

    times = np.array(times)
    return times.mean(), times.std()


def benchmark_powerlaw(x: np.ndarray) -> BenchmarkResult:
    """Benchmark PowerLawModel evaluate and fit_deriv methods."""
    print("\n" + "=" * 70)
    print("Benchmarking PowerLawModel")
    print("=" * 70)

    model = PowerLawModel.create(
        x0=1450.0,
        y0=1.0,
        flux=1.0,
        alpha=-1.0,
        name='powerlaw',
    )

    # Benchmark evaluate
    print(f"  Evaluating on {x.size} wavelengths...")
    mean_eval, std_eval = benchmark_method(
        lambda: model.evaluate(x, model.flux.value, model.alpha.value),
    )
    result_eval = BenchmarkResult(
        model_name='PowerLawModel',
        method_name='evaluate',
        total_time=mean_eval * N,
        mean_time=mean_eval,
        std_time=std_eval,
    )
    print(f"    Mean: {result_eval.mean_time_us:.0f} ± {result_eval.std_time_us:.0f} µs")

    # Benchmark fit_deriv
    print(f"  Computing fit_deriv on {x.size} wavelengths...")
    mean_deriv, std_deriv = benchmark_method(
        lambda: model.fit_deriv(x, model.flux.value, model.alpha.value),
    )
    result_deriv = BenchmarkResult(
        model_name='PowerLawModel',
        method_name='fit_deriv',
        total_time=mean_deriv * N,
        mean_time=mean_deriv,
        std_time=std_deriv,
    )
    print(f"    Mean: {result_deriv.mean_time_us:.0f} ± {result_deriv.std_time_us:.0f} µs")

    return result_eval, result_deriv


def benchmark_gaussian(x: np.ndarray) -> BenchmarkResult:
    """Benchmark GaussianModel evaluate and fit_deriv methods."""
    print("\n" + "=" * 70)
    print("Benchmarking GaussianModel")
    print("=" * 70)

    model = GaussianModel.create(
        wave=1549.0,
        sigma_res=V_RES,
        strength=1.0,
        sigma_v=1e-3,
        v_off=0.0,
        name='gaussian',
    )

    # Benchmark evaluate
    print(f"  Evaluating on {x.size} wavelengths...")
    mean_eval, std_eval = benchmark_method(
        lambda: model.evaluate(x, model.strength.value, model.sigma_v.value, model.v_off.value),
    )
    result_eval = BenchmarkResult(
        model_name='GaussianModel',
        method_name='evaluate',
        total_time=mean_eval * N,
        mean_time=mean_eval,
        std_time=std_eval,
    )
    print(f"    Mean: {result_eval.mean_time_us:.0f} ± {result_eval.std_time_us:.0f} µs")

    # Benchmark fit_deriv
    print(f"  Computing fit_deriv on {x.size} wavelengths...")
    mean_deriv, std_deriv = benchmark_method(
        lambda: model.fit_deriv(x, model.strength.value, model.sigma_v.value, model.v_off.value),
    )
    result_deriv = BenchmarkResult(
        model_name='GaussianModel',
        method_name='fit_deriv',
        total_time=mean_deriv * N,
        mean_time=mean_deriv,
        std_time=std_deriv,
    )
    print(f"    Mean: {result_deriv.mean_time_us:.0f} ± {result_deriv.std_time_us:.0f} µs")

    return result_eval, result_deriv


def benchmark_iron(x: np.ndarray) -> BenchmarkResult:
    """Benchmark IronModel evaluate and fit_deriv methods."""
    print("\n" + "=" * 70)
    print("Benchmarking IronModel")
    print("=" * 70)

    # Load and adapt template
    _template = IronTemplate.load(path='vw2001', info=INFO)
    template = _template.createLogspace(sigma_res=V_RES, xr=x, keep_x=True)

    scale = INFO.iron['scale']
    model = IronModel.create(
        flux=1.0,
        fwhm=template.fwhm[10],  # Use a mid-range FWHM
        scale=scale,
        template=template,
        name='iron',
    )

    # Benchmark evaluate
    print(f"  Evaluating on {x.size} wavelengths...")
    mean_eval, std_eval = benchmark_method(
        lambda: model.evaluate(
            x,
            model.flux.value,
            model.fwhm.value,
            model.split.value,
            model.left.value,
            model.right.value,
        ),
    )
    result_eval = BenchmarkResult(
        model_name='IronModel',
        method_name='evaluate',
        total_time=mean_eval * N,
        mean_time=mean_eval,
        std_time=std_eval,
    )
    print(f"    Mean: {result_eval.mean_time_us:.0f} ± {result_eval.std_time_us:.0f} µs")

    # Benchmark fit_deriv
    print(f"  Computing fit_deriv on {x.size} wavelengths...")
    mean_deriv, std_deriv = benchmark_method(
        lambda: model.fit_deriv(
            x,
            model.flux.value,
            model.fwhm.value,
            model.split.value,
            model.left.value,
            model.right.value,
        ),
    )
    result_deriv = BenchmarkResult(
        model_name='IronModel',
        method_name='fit_deriv',
        total_time=mean_deriv * N,
        mean_time=mean_deriv,
        std_time=std_deriv,
    )
    print(f"    Mean: {result_deriv.mean_time_us:.0f} ± {result_deriv.std_time_us:.0f} µs")

    return result_eval, result_deriv


def benchmark_balmer(x: np.ndarray) -> BenchmarkResult:
    """Benchmark BalmerModel evaluate and fit_deriv methods."""
    print("\n" + "=" * 70)
    print("Benchmarking BalmerModel")
    print("=" * 70)

    # Load templates
    continuum_template = BalmerContinuumTemplate.load_from_cache(
        temp=INFO.balmer.temp,
        tau=INFO.balmer.tau,
        scale=INFO.balmer.scale,
        info=INFO,
    )
    series_template = BalmerSeriesTemplate.load_from_cache(
        name='sh1995',
        temp=INFO.balmer.temp,
        dens=INFO.balmer.dens,
        n_u_range=(INFO.balmer.n_u_min, INFO.balmer.n_u_max),
        info=INFO,
    )
    model = BalmerModel.create(
        flux=INFO.balmer.flux,
        fwhm=INFO.balmer.fwhm,  # Use a mid-range FWHM
        ratio=INFO.balmer.ratio,
        edge=INFO.balmer.edge,
        continuum_template=continuum_template,
        series_template=series_template,
        info=INFO,
        name='sh1995',
    )

    # Benchmark evaluate
    print(f"  Evaluating on {x.size} wavelengths...")
    mean_eval, std_eval = benchmark_method(
        lambda: model.evaluate(
            x,
            model.flux.value,
            model.fwhm.value,
            model.ratio.value,
        ),
    )
    result_eval = BenchmarkResult(
        model_name='BalmerModel',
        method_name='evaluate',
        total_time=mean_eval * N,
        mean_time=mean_eval,
        std_time=std_eval,
    )
    print(f"    Mean: {result_eval.mean_time_us:.0f} ± {result_eval.std_time_us:.0f} µs")

    # Benchmark fit_deriv
    print(f"  Computing fit_deriv on {x.size} wavelengths...")
    mean_deriv, std_deriv = benchmark_method(
        lambda: model.fit_deriv(
            x,
            model.flux.value,
            model.fwhm.value,
            model.ratio.value,
        ),
    )
    result_deriv = BenchmarkResult(
        model_name='BalmerModel',
        method_name='fit_deriv',
        total_time=mean_deriv * N,
        mean_time=mean_deriv,
        std_time=std_deriv,
    )
    print(f"    Mean: {result_deriv.mean_time_us:.0f} ± {result_deriv.std_time_us:.0f} µs")

    return result_eval, result_deriv


def benchmark_hostgalaxy(x: np.ndarray) -> BenchmarkResult:
    """Benchmark HostGalaxyModel evaluate and fit_deriv methods."""
    print("\n" + "=" * 70)
    print("Benchmarking HostGalaxyModel")
    print("=" * 70)

    # Load template
    template = HostGalaxyTemplate.load_from_cache(
        name=INFO.host.sources[0],
        age=INFO.host.ages[0],
        info=INFO,
    )

    model = HostGalaxyModel.create(
        flux=INFO.host.flux,
        fwhm=INFO.host.fwhm,  # Use a mid-range FWHM
        template=template,
        info=INFO,
        name=INFO.host.sources[0],
        age=INFO.host.ages[0],
    )

    # Benchmark evaluate
    print(f"  Evaluating on {x.size} wavelengths...")
    mean_eval, std_eval = benchmark_method(
        lambda: model.evaluate(x, model.flux.value, model.fwhm.value),
    )
    result_eval = BenchmarkResult(
        model_name='HostGalaxyModel',
        method_name='evaluate',
        total_time=mean_eval * N,
        mean_time=mean_eval,
        std_time=std_eval,
    )
    print(f"    Mean: {result_eval.mean_time_us:.0f} ± {result_eval.std_time_us:.0f} µs")

    # Benchmark fit_deriv
    print(f"  Computing fit_deriv on {x.size} wavelengths...")
    mean_deriv, std_deriv = benchmark_method(
        lambda: model.fit_deriv(x, model.flux.value, model.fwhm.value),
    )
    result_deriv = BenchmarkResult(
        model_name='HostGalaxyModel',
        method_name='fit_deriv',
        total_time=mean_deriv * N,
        mean_time=mean_deriv,
        std_time=std_deriv,
    )
    print(f"    Mean: {result_deriv.mean_time_us:.0f} ± {result_deriv.std_time_us:.0f} µs")

    return result_eval, result_deriv


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted summary table of all benchmark results."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)
    print(
        f"{'Model':<20} {'Method':<15} {'Mean (µs)':<15} {'Std (µs)':<15}"
    )
    print("-" * 90)

    for result in results:
        print(
            f"{result.model_name:<20} {result.method_name:<15} "
            f"{result.mean_time_us:>14.0f} {result.std_time_us:>14.0f}"
        )

    print("=" * 90)


def main() -> None:
    """Run all benchmarks."""
    print(f"\nConfiguration:")
    print(f"  Velocity resolution (v_res): {V_RES}")
    print(f"  Number of wavelengths: {N_WAVELENGTHS}")
    print(f"  Wavelength range: {1000.0:.1f} - {1000.0 * (1 + V_RES) ** (N_WAVELENGTHS - 1):.1f} Å")

    # Create wavelength grid
    x = create_wavelength_grid(v_res=V_RES, n_wavelengths=N_WAVELENGTHS)
    print(f"  Wavelength grid created: {x.shape}")

    # Run benchmarks
    results = []

    r1, r2 = benchmark_powerlaw(x)
    results.extend([r1, r2])

    r1, r2 = benchmark_gaussian(x)
    results.extend([r1, r2])

    r1, r2 = benchmark_iron(x)
    results.extend([r1, r2])

    r1, r2 = benchmark_balmer(x)
    results.extend([r1, r2])

    r1, r2 = benchmark_hostgalaxy(x)
    results.extend([r1, r2])

    # Print summary
    print_summary_table(results)


if __name__ == '__main__':
    main()
