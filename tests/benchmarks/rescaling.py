import math
from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter

from numpy import arange, array, full
from numpy.random import default_rng
from quasar_utils.setup import Info
from tqdm import tqdm

from quasar_models.host import HostGalaxyModel, HostGalaxyTemplate
from quasar_models.modeling import PrepareModel, SequentialModel

N: int = 10_000

SIGMA_RES: float = 2.3e-4
X_LB: float = 4000.0
X_UB: float = 10_000.0
INFO: Info = Info()
SEED: int = 42

n_pix = math.floor(math.log(X_UB / X_LB) / math.log(1 + SIGMA_RES))
x = X_LB * (1 + SIGMA_RES) ** arange(n_pix)

host_template = HostGalaxyTemplate.load_from_cache(
    name=INFO.host.sources[0],
    age=INFO.host.ages[0],
    info=INFO,
).createLogspace(
    sigma_res=SIGMA_RES,
)

host_model: HostGalaxyModel = HostGalaxyModel.create(
    1.0,
    1000.0,
    info=INFO,
    template=host_template,
    allow_interp_fitting=False,
    n_scales=INFO.convolution.n_scales,
    flux_bounds=INFO.host.flux_bounds,
    fwhm_bounds=INFO.host.fwhm_bounds,
)

@contextmanager
def rescaled_host_model() -> Generator[SequentialModel]:
    host_model.flux.fixed = False
    host_model.fwhm.fixed = True
    host_model.allow_interp_fitting = False
    try:
        with PrepareModel(x=x, model=host_model) as seq_model:
            yield seq_model
    finally:
        host_model.flux.fixed = False
        host_model.fwhm.fixed = False
        host_model.allow_interp_fitting = False

@contextmanager
def interped_host_model() -> Generator[SequentialModel]:
    host_model.flux.fixed = False
    host_model.fwhm.fixed = False
    host_model.allow_interp_fitting = True
    try:
        with PrepareModel(x=x, model=host_model) as seq_model:
            yield seq_model
    finally:
        host_model.flux.fixed = False
        host_model.fwhm.fixed = False
        host_model.allow_interp_fitting = False

@contextmanager
def exact_host_model() -> Generator[SequentialModel]:
    host_model.flux.fixed = False
    host_model.fwhm.fixed = False
    host_model.allow_interp_fitting = False
    try:
        with PrepareModel(x=x, model=host_model) as seq_model:
            yield seq_model
    finally:
        host_model.flux.fixed = False
        host_model.fwhm.fixed = False
        host_model.allow_interp_fitting = False

def benchmark_host_model(model: SequentialModel) -> tuple[list, float]:
    rng = default_rng(SEED)

    m = model._model
    if m.flux.fixed:
        fluxs = full(N, m.flux.value)
    else:
        fluxs = rng.uniform(*m.flux.bounds, size=N)

    if m.fwhm.fixed:
        fwhms = full(N, m.fwhm.value)
    else:
        fwhms = rng.uniform(*m.fwhm.bounds, size=N)


    ps = [
        array([args[i] for i in model._free_indices])
        for args in zip(fluxs, fwhms)
    ]
    start = perf_counter()
    ys = [
        model.evaluate(x, p) 
        for p in tqdm(
            ps,
            desc=str(m.evaluate_func),
            total=N,
        )
    ]
    t = (perf_counter() - start) / N * 1e6

    return ys, t

def main() -> None:    
    with rescaled_host_model() as f:
        print(f"Rescaled host model: {f._model.evaluate_func}")
        _, t1 = benchmark_host_model(f)

    with interped_host_model() as f:
        print(f"Interped host model: {f._model.evaluate_func}")
        _, t2 = benchmark_host_model(f)

    with exact_host_model() as f:
        print(f"Exact host model: {f._model.evaluate_func}")
        _, t3 = benchmark_host_model(f)

    print("Performance:")
    print(f"> With rescaling:     {t1:.3f} µs")
    print(f"> With interpolation: {t2:.3f} µs")
    print(f"> Exact:              {t3:.3f} µs")

if __name__ == "__main__":
    main()