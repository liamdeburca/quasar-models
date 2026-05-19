# Gaussian Line Model — Mathematical Derivation

## Model

The `GaussianModel` evaluates an emission line as a **flux-normalised Gaussian**
(i.e., the integral over $x$ equals `strength`).  The total line width is the
quadrature sum of the intrinsic velocity dispersion and the instrumental
resolution:

$$
\mu = \lambda_0 (1 + v_\text{off})
$$

$$
\sigma_\text{tot}^2 = \sigma_v^2 + \sigma_\text{res}^2
$$

$$
\sigma = \mu\,\sigma_\text{tot}
$$

$$
z = \frac{x - \mu}{\sigma}
$$

$$
f(x) = \frac{A}{\sqrt{2\pi}}\,\frac{1}{\sigma}\,e^{-z^2/2}
$$

where $A$ = `strength` is the flux integral of the line.  The factor
$g = 1/\sqrt{2\pi}$ is stored as `gauss_amp`.

| Symbol | Role | Parameter |
|--------|------|-----------|
| $A$ | Line strength (flux integral) | `strength` — **fittable** |
| $\sigma_v$ | Intrinsic velocity dispersion (in units of $c$) | `sigma_v` — **fittable** |
| $v_\text{off}$ | Velocity offset from rest wavelength | `v_off` — **fittable** |
| $\lambda_0$ | Rest wavelength | `wave` — fixed constant |
| $\sigma_\text{res}$ | Instrumental velocity resolution | `sigma_res` — fixed constant |

---

## Partial Derivatives

### Shared intermediates

$$
\sigma_\text{tot}^2 = \sigma_v^2 + \sigma_\text{res}^2, \qquad
\sigma = \mu\,\sigma_\text{tot}, \qquad
z = \frac{x - \mu}{\sigma}, \qquad
f = \frac{g\,A}{\sigma}\,e^{-z^2/2}
$$

All three derivatives can be expressed cleanly in terms of $f$ itself.

---

### With respect to `strength`

$$
\frac{\partial f}{\partial A} = \frac{f}{A} = \frac{g}{\sigma}\,e^{-z^2/2}
$$

---

### With respect to `sigma_v`

We use the chain rule through $\sigma$:

$$
\frac{\partial \sigma}{\partial \sigma_v} = \mu\,\frac{\sigma_v}{\sigma_\text{tot}}
= \frac{\sigma\,\sigma_v}{\sigma_\text{tot}^2}
$$

For the $\sigma$-dependence of $f$ (treating $\mu$ as constant):

$$
\frac{\partial f}{\partial \sigma} = \frac{f}{\sigma}\,(z^2 - 1)
$$

*(Proof: differentiate $e^{-z^2/2}/\sigma$ w.r.t.\ $\sigma$, using
$\partial z/\partial\sigma = -z/\sigma$.)*

Combining:

$$
\boxed{
\frac{\partial f}{\partial \sigma_v}
= f\,(z^2 - 1)\,\frac{\sigma_v}{\sigma_\text{tot}^2}
}
$$

---

### With respect to `v_off`

$$
\frac{\partial \mu}{\partial v_\text{off}} = \lambda_0 = \frac{\mu}{1+v_\text{off}}
$$

Both $\sigma$ and $z$ depend on $\mu$.  Treating $\sigma_\text{tot}$ as
constant and using $\sigma = \mu\,\sigma_\text{tot}$:

$$
\frac{\partial z}{\partial \mu}
= \frac{-\sigma - (x-\mu)\sigma_\text{tot}}{\sigma^2}
= -\frac{\sigma_\text{tot}\,x}{\sigma^2}
= -\frac{x}{\mu\,\sigma}
$$

The total $\mu$-derivative of $f = g\,A / (\mu\sigma_\text{tot})\, e^{-z^2/2}$ is:

$$
\frac{\partial f}{\partial \mu}
= \frac{f}{\mu}\!\left(z\,\frac{x}{\sigma} - 1\right)
= \frac{f}{\mu}\left(z\,x\,\sigma^{-1} - 1\right)
$$

*(Proof: write $f = g\,A\,({\mu\sigma_\text{tot}})^{-1} e^{-z^2/2}$, differentiate, and factor.)*

Applying the chain rule:

$$
\boxed{
\frac{\partial f}{\partial v_\text{off}}
= \frac{f}{1+v_\text{off}}\!\left(z\,x\,\sigma^{-1} - 1\right)
}
$$

---

## Implementation Validation

The `fit_deriv_numba` function in `evaluation.py` computes:

```python
# Shared intermediates
sigma_tot_sq = sigma_v**2 + sigma_res**2
sigma        = mean * sigma_tot_sq**0.5       # mean = mu
inv_sigma    = 1.0 / sigma
z            = (x - mean) * inv_sigma
amp          = gauss_amp * inv_sigma
_f           = amp * exp(-0.5 * z**2)         # f / strength
f            = strength * _f                  # f

df_dstrength = _f                             # f/A              ✓
df_dsigma_v  = f * (z_sq - 1) * sigma_v / sigma_tot_sq    ✓
df_dv_off    = f * (z * x * inv_sigma - 1) / (1 + v_off)  ✓
```

All three derivatives are **correctly implemented**.
