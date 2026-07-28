import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from quasar_models.line import GaussianModel

SIGMA_RES: float = 2.3e-4
x0: float = 1000.0
x = x0 * (1 + SIGMA_RES) ** np.arange(10_000)

np.random.seed(42)

MODEL = GaussianModel.create(
    wave=1549.0,
    sigma_res=SIGMA_RES,
    strength=1.0,
    fwhm_v=1e-3,
    v_off=0.0,
)
MODEL.strength.bounds = (0.1, 10.0)
MODEL.fwhm_v.bounds = (1e-4, 1e-2)
MODEL.v_off.bounds = (-1e-2, 1e-2)


def jac_analytic(p):
    out = MODEL.jac(x, *p)
    return out


def jac_numeric(p):
    strength, fwhm_v, v_off = p

    dstrength = min(
        [
            1e-5,
            MODEL.strength.bounds[1] - strength,
            strength - MODEL.strength.bounds[0],
        ]
    )
    df_dstrength = MODEL.evaluate(
        x, strength + dstrength, fwhm_v, v_off
    ) - MODEL.evaluate(x, strength - dstrength, fwhm_v, v_off)
    df_dstrength /= 2 * dstrength

    dfwhm_v = min(
        [
            SIGMA_RES,
            MODEL.fwhm_v.bounds[1] - fwhm_v,
            fwhm_v - MODEL.fwhm_v.bounds[0],
        ]
    )
    df_dfwhm_v = MODEL.evaluate(x, strength, fwhm_v + dfwhm_v, v_off) - MODEL.evaluate(
        x, strength, fwhm_v - dfwhm_v, v_off
    )
    df_dfwhm_v /= 2 * dfwhm_v

    dv_off = min(
        [
            SIGMA_RES,
            MODEL.v_off.bounds[1] - v_off,
            v_off - MODEL.v_off.bounds[0],
        ]
    )
    df_dv_off = MODEL.evaluate(x, strength, fwhm_v, v_off + dv_off) - MODEL.evaluate(
        x, strength, fwhm_v, v_off - dv_off
    )
    df_dv_off /= 2 * dv_off

    return np.stack([df_dstrength, df_dfwhm_v, df_dv_off], axis=0)


def perturb_model() -> None:
    MODEL.strength.value = np.random.uniform(*MODEL.strength.bounds)
    MODEL.fwhm_v.value = np.random.uniform(*MODEL.fwhm_v.bounds)
    MODEL.v_off.value = np.random.uniform(*MODEL.v_off.bounds)


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

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Boxplot for mismatch
    axes[0].boxplot(mismatch, tick_labels=["strength", "fwhm_v", "v_off"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mismatch (log scale)")
    axes[0].set_title("Distribution of Discrepancies")
    axes[0].grid(True, alpha=0.3)

    # Boxplot for sign mismatch
    axes[1].boxplot(sign_mismatch, tick_labels=["strength", "fwhm_v", "v_off"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Sign Mismatch (log scale)")
    axes[1].set_title("Distribution of Sign Mismatches")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(100)
