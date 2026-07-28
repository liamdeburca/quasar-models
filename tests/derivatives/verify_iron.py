import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from quasar_utils.setup import Info
from quasar_models.iron import IronModel, IronTemplate

SIGMA_RES: float = 2.3e-4
x0: float = 1000.0
x = x0 * (1 + SIGMA_RES) ** np.arange(10_000)

np.random.seed(42)

INFO = Info()

# Load and adapt template
template = IronTemplate.load(
    path="vw2001",
    info=INFO,
).createLogspace(sigma_res=SIGMA_RES, xr=x)

scale = INFO.iron["scale"]
MODEL = IronModel.create(
    1.0,
    template.fwhm[0],
    scale=scale,
    template=template,
    name="iron",
    allow_interp_fitting=False,
)
MODEL.flux.bounds = (0.1, 10.0)
MODEL.fwhm.bounds = (template.fwhm[1], template.fwhm[-2])
MODEL.split.bounds = (2200.0, 2700.0)

MODEL.left.bounds = (0.1, 1.0)
MODEL.split.fixed = MODEL.left.fixed = MODEL.right.fixed = False

# MODEL.right.value = 1.0
# MODEL.right.fixed = True

MODEL._prepare_model(x)
# del MODEL.fit_deriv_func
print(MODEL.fit_deriv_func.__wrapped__)


def jac_analytic(p):
    return MODEL.jac(x, *p)


def jac_numeric(p):
    flux, fwhm, split, left, right = p

    dflux = min(
        [
            1e-6,
            MODEL.flux.bounds[1] - flux,
            flux - MODEL.flux.bounds[0],
        ]
    )
    df_dflux = MODEL.evaluate(
        x, flux + dflux, fwhm, split, left, right
    ) - MODEL.evaluate(x, flux - dflux, fwhm, split, left, right)
    df_dflux /= 2 * dflux

    dfwhm = min(
        [
            100 / 3e5,  # ~100 km/s
            MODEL.fwhm.bounds[1] - fwhm,
            fwhm - MODEL.fwhm.bounds[0],
        ]
    )
    df_dfwhm = MODEL.evaluate(
        x, flux, fwhm + dfwhm, split, left, right
    ) - MODEL.evaluate(x, flux, fwhm - dfwhm, split, left, right)
    df_dfwhm /= 2 * dfwhm

    dsplit = min(
        [
            5.0 * split * SIGMA_RES,
            MODEL.split.bounds[1] - split,
            split - MODEL.split.bounds[0],
        ]
    )
    df_dsplit = MODEL.evaluate(
        x, flux, fwhm, split + dsplit, left, right
    ) - MODEL.evaluate(x, flux, fwhm, split - dsplit, left, right)
    df_dsplit /= 2 * dsplit

    dleft = min(
        [
            1e-5,
            MODEL.left.bounds[1] - left,
            left - MODEL.left.bounds[0],
        ]
    )
    df_dleft = MODEL.evaluate(
        x, flux, fwhm, split, left + dleft, right
    ) - MODEL.evaluate(x, flux, fwhm, split, left - dleft, right)
    df_dleft /= 2 * dleft

    dright = min(
        [
            1e-5,
            MODEL.right.bounds[1] - right,
            right - MODEL.right.bounds[0],
        ]
    )
    df_dright = MODEL.evaluate(
        x, flux, fwhm, split, left, right + dright
    ) - MODEL.evaluate(x, flux, fwhm, split, left, right - dright)
    df_dright /= 2 * dright

    return np.stack([df_dflux, df_dfwhm, df_dsplit, df_dleft, df_dright], axis=0)


def perturb_model() -> None:
    MODEL.flux.value = round(np.random.uniform(*MODEL.flux.bounds), 6)
    MODEL.fwhm.value = round(np.random.uniform(*MODEL.fwhm.bounds), 6)
    MODEL.split.value = round(np.random.uniform(*MODEL.split.bounds), 6)
    MODEL.left.value = round(np.random.uniform(*MODEL.left.bounds), 6)
    MODEL.right.value = round(np.random.uniform(*MODEL.right.bounds), 6)


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
    mismatch = np.zeros((N, 5))
    sign_mismatch = np.zeros((N, 5))

    for i in tqdm(range(N), leave=False):
        perturb_model()
        p = MODEL.parameters
        f = MODEL.evaluate(x, *p)
        # f0 = MODEL.evaluate(x, p[0], p[1], p[2], 1.0, 1.0)
        jac_a = jac_analytic(p)
        jac_n = jac_numeric(p)

        m, s = analyse_results(f, jac_a, jac_n)
        mismatch[i, :], sign_mismatch[i, :] = m, s

        # fig, axes = plt.subplots(2, 1, sharex=True)

        # axes[0].plot(x, f, 'k')
        # axes[0].plot(x, f0, 'g')

        # axes[1].plot(x, jac_n[2], 'b')
        # axes[1].plot(x, jac_a[2], 'r')
        # axes[1].set_xlim(1000.0, 3000.0)
        # plt.show()
        # raise ValueError("Stop after first iteration for debugging.")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Boxplot for mismatch
    axes[0].boxplot(mismatch, tick_labels=["flux", "fwhm", "split", "left", "right"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mismatch (log scale)")
    axes[0].set_title("Distribution of Discrepancies")
    axes[0].grid(True, alpha=0.3)

    # Boxplot for sign mismatch
    axes[1].boxplot(
        sign_mismatch, tick_labels=["flux", "fwhm", "split", "left", "right"]
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Sign Mismatch (log scale)")
    axes[1].set_title("Distribution of Sign Mismatches")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(100)
