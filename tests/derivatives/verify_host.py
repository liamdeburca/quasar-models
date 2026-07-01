import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from quasar_utils.setup import Info
from quasar_models.host import HostGalaxyModel, HostGalaxyTemplate

SIGMA_RES: float = 2.3e-4
x0: float = 1000.0
x = x0 * (1 + SIGMA_RES) ** np.arange(10_000)

np.random.seed(42)

INFO = Info()

template = HostGalaxyTemplate.load_from_cache(
    name=INFO.host.sources[0],
    age=INFO.host.ages[0],
    info=INFO,
)

TEMPLATE = template.createLogspace(
    sigma_res=SIGMA_RES,
    xr=x,
)
TEMPLATE.normalise(inplace=True)
MODEL = HostGalaxyModel.create(
    flux=INFO.host.flux,
    fwhm=INFO.host.fwhm,  # Use a mid-range FWHM
    template=TEMPLATE,
    info=INFO,
    name=INFO.host.sources[0],
    age=INFO.host.ages[0],
    allow_interp_fitting=False,
)
MODEL.flux.bounds = (0.1, 10.0)
MODEL.fwhm.fixed = True

MODEL._prepare_model(x)
print(MODEL.fit_deriv_func.__wrapped__)

def jac_analytic(p):
    return MODEL.jac(x, *p)

def jac_numeric(p):
    flux, fwhm = p

    dflux = min([
        1e-5,
        MODEL.flux.bounds[1] - flux,
        flux - MODEL.flux.bounds[0],
    ])
    df_dflux = MODEL.evaluate(x, flux + dflux, fwhm) - MODEL.evaluate(x, flux - dflux, fwhm)
    df_dflux /= 2 * dflux

    df_dfwhm = np.zeros_like(x)

    return np.stack([df_dflux, df_dfwhm], axis=0)

def perturb_model() -> None:
    MODEL.flux.value = np.random.uniform(*MODEL.flux.bounds)

def analyse_results(f, jac_a, jac_n):    
    # Mismatch relative to maximum absolute value of the numeric jacobian, weighted by the normalized flux
    deriv_mismatch = (jac_a - jac_n) / np.abs(jac_n).max(axis=1)[:, None]
    deriv_mismatch *= (f / f.sum())[None, :]
    
    # Sign mismatch weighted by the absolute value of the numeric jacobian
    sign_mismatch = (np.sign(jac_a) != np.sign(jac_n)).astype(float)
    sign_mismatch *= np.abs(jac_n) / np.abs(jac_n).max(axis=1)[:, None]

    return (
        np.maximum(deriv_mismatch.sum(axis=1).round(12), 1e-12),
        np.maximum(sign_mismatch.sum(axis=1).round(12), 1e-12),
    )

def main(N: int):
    mismatch = np.zeros((N, 2))
    sign_mismatch = np.zeros((N, 2))

    for i in tqdm(range(N), leave=False):
        perturb_model()
        p = MODEL.parameters
        f = MODEL.evaluate(x, *p)
        jac_a = jac_analytic(p)
        jac_n = jac_numeric(p)

        m, s = analyse_results(f, jac_a, jac_n)
        mismatch[i, :], sign_mismatch[i, :] = m, s

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Boxplot for mismatch
    axes[0].boxplot(mismatch, tick_labels=['flux', 'fwhm'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Mismatch (log scale)')
    axes[0].set_title('Distribution of Discrepancies')
    axes[0].grid(True, alpha=0.3)

    # Boxplot for sign mismatch
    axes[1].boxplot(sign_mismatch, tick_labels=['flux', 'fwhm'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Sign Mismatch (log scale)')
    axes[1].set_title('Distribution of Sign Mismatches')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(100)