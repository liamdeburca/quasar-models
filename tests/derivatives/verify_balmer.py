import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from quasar_utils.setup import Info
from quasar_models.balmer import BalmerModel
from quasar_models.balmer.continuum import BalmerContinuumTemplate
from quasar_models.balmer.series import BalmerSeriesTemplate

SIGMA_RES: float = 2.3e-4
x0: float = 1000.0
x = x0 * (1 + SIGMA_RES) ** np.arange(10_000)

np.random.seed(42)

INFO = Info()

# Load templates
continuum_template = BalmerContinuumTemplate.load_from_cache(
    temp=INFO.balmer.temp,
    tau=INFO.balmer.tau,
    scale=INFO.balmer.scale,
    info=INFO,
).createLogspace(
    sigma_res=SIGMA_RES,
    xr=x,
)

series_template = BalmerSeriesTemplate.load_from_cache(
    name='sh1995',
    temp=INFO.balmer.temp,
    dens=INFO.balmer.dens,
    n_u_range=(INFO.balmer.n_u_min, INFO.balmer.n_u_max),
    info=INFO,
).createLogspace(
    sigma_res=SIGMA_RES,
    xr=x,
)

MODEL = BalmerModel.create(
    flux=INFO.balmer.flux,
    fwhm=INFO.balmer.fwhm,
    ratio=INFO.balmer.ratio,
    edge=INFO.balmer.edge,
    continuum_template=continuum_template,
    series_template=series_template,
    info=INFO,
    name='sh1995',
    allow_interp_fitting=False,
)
MODEL.flux.bounds = (0.1, 10.0)
MODEL.fwhm.bounds = (continuum_template.fwhm[1], continuum_template.fwhm[-2])
MODEL.ratio.bounds = (0.1, 2.0)

# MODEL.ratio.fixed = True

MODEL._prepare_model(x)
print(MODEL.fit_deriv_func.__wrapped__)

def jac_analytic(p):
    return MODEL.jac(x, *p)

def jac_numeric(p):
    flux, fwhm, ratio = p

    dflux = min([
        1e-5,
        MODEL.flux.bounds[1] - flux,
        flux - MODEL.flux.bounds[0],
    ])
    df_dflux = MODEL.evaluate(x, flux + dflux, fwhm, ratio) - MODEL.evaluate(x, flux - dflux, fwhm, ratio)
    df_dflux /= 2 * dflux

    dfwhm = min([
        100 / 3e5, # ~100 km/s
        MODEL.fwhm.bounds[1] - fwhm,
        fwhm - MODEL.fwhm.bounds[0],
    ])
    df_dfwhm = MODEL.evaluate(x, flux, fwhm + dfwhm, ratio) - MODEL.evaluate(x, flux, fwhm - dfwhm, ratio)
    df_dfwhm /= 2 * dfwhm

    dratio = min([
        1e-3,
        MODEL.ratio.bounds[1] - ratio,
        ratio - MODEL.ratio.bounds[0],
    ])
    df_dratio = MODEL.evaluate(x, flux, fwhm, ratio + dratio) - MODEL.evaluate(x, flux, fwhm, ratio - dratio)
    df_dratio /= 2 * dratio

    return np.stack([df_dflux, df_dfwhm, df_dratio], axis=0)

def perturb_model() -> None:
    MODEL.flux.value = round(np.random.uniform(*MODEL.flux.bounds), 6)
    MODEL.fwhm.value = round(np.random.uniform(*MODEL.fwhm.bounds), 6)
    MODEL.ratio.value = round(np.random.uniform(*MODEL.ratio.bounds), 6)

def analyse_results(f, jac_a, jac_n):    
    # Mismatch relative to maximum absolute value of the numeric jacobian, weighted by the normalized flux
    deriv_mismatch = (jac_a - jac_n) / np.abs(jac_n).max(axis=1)[:, None]
    deriv_mismatch *= (f / f.sum())[None, :]
    
    # Sign mismatch weighted by the absolute value of the numeric jacobian
    sign_mismatch = (np.sign(jac_a) != np.sign(jac_n)).astype(float)
    sign_mismatch *= np.abs(jac_n) / np.abs(jac_n).sum(axis=1)[:, None]

    return (
        np.maximum(deriv_mismatch.sum(axis=1).round(12), 1e-12),
        np.maximum(sign_mismatch.sum(axis=1).round(12), 1e-12),
    )

def main(N: int):
    mismatch = np.zeros((N, 3))
    sign_mismatch = np.zeros((N, 3))

    for i in tqdm(range(N), leave=False):
        perturb_model()
        p = MODEL.parameters
        f = MODEL.evaluate(x, *p)
        jac_a = jac_analytic(p)
        jac_n = jac_numeric(p)

        m, s = analyse_results(f, jac_a, jac_n)
        mismatch[i, :], sign_mismatch[i, :] = m, s

        # plt.figure()
        # plt.plot(x, f, 'k')
        # plt.plot(x, jac_n[1], 'b')
        # plt.plot(x, jac_a[1], 'r')
        # plt.xlim(1000.0, 5000.0)
        # plt.show()
        # raise ValueError("Stop after first iteration for debugging.")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Boxplot for mismatch
    axes[0].boxplot(mismatch, tick_labels=['flux', 'fwhm', 'ratio'])
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Mismatch (log scale)')
    axes[0].set_title('Distribution of Discrepancies')
    axes[0].grid(True, alpha=0.3)

    # Boxplot for sign mismatch
    axes[1].boxplot(sign_mismatch, tick_labels=['flux', 'fwhm', 'ratio'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Sign Mismatch (log scale)')
    axes[1].set_title('Distribution of Sign Mismatches')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(100)