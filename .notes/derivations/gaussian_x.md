# Gaussian Line Model (Wavelength Resolution) — Mathematical Derivation

## Model

The `GaussianModel_x` evaluates an emission line as a **flux-normalised Gaussian** using wavelength-space instrumental resolution (unlike the velocity-space version). The total line width is the quadrature sum of the intrinsic velocity dispersion (converted to wavelength) and the instrumental wavelength resolution:

$$
\mu = \lambda_0 \left(1 + \frac{v_\text{off}}{C_\text{km/s}}\right)
$$

$$
\sigma_v = \frac{\mathrm{fwhm\_v}}{2\sqrt{2\ln 2}}
$$

$$
\sigma_{\lambda,\mathrm{v}} = \mu \cdot \frac{\sigma_v}{C_\text{km/s}}
$$

$$
\sigma_{\lambda,\mathrm{tot}}^2 = \sigma_{\lambda,\mathrm{v}}^2 + \mathrm{dx}^2
$$

$$
\sigma_\lambda = \sigma_{\lambda,\mathrm{tot}}
$$

$$
z = \frac{x - \mu}{\sigma_\lambda}
$$

$$
f(x) = \frac{A}{\sqrt{2\pi}}\,\frac{1}{\sigma_\lambda}\,e^{-z^2/2}
$$

where $A$ = `strength` is the flux integral of the line.  The factor $g = 1/\sqrt{2\pi}$ is stored as `gauss_amp`.

| Symbol | Role | Parameter |
|--------|------|-----------|
| $A$ | Line strength (flux integral) | `strength` — **fittable** |
| $\mathrm{fwhm\_v}$ | Intrinsic velocity FWHM (km/s) | `fwhm_v` — **fittable** |
| $v_\text{off}$ | Velocity offset from rest wavelength (km/s) | `v_off` — **fittable** |
| $\lambda_0$ | Rest wavelength | `wave` — fixed constant |
| $\mathrm{dx}$ | Instrumental wavelength resolution (wavelength units) | `dx` — fixed constant |
| $C_\text{km/s}$ | Speed of light (km/s) | 299792.458 — fixed constant |

---

## Key Differences from Velocity Space Model

The wavelength-space model differs from the velocity-space model in how instrumental resolution is combined with intrinsic broadening:

**Velocity Space:**  
- Instrumental resolution is $\sigma_\text{res}$ (fraction of c)
- Total dispersion: $\sigma = \mu \cdot \sqrt{(\sigma_v)^2 + (\sigma_\text{res})^2}$ — **scales with wavelength**

**Wavelength Space:**  
- Instrumental resolution is $\mathrm{dx}$ (wavelength units)
- Total dispersion: $\sigma_\lambda = \sqrt{(\mu \cdot \sigma_v / C)^2 + \mathrm{dx}^2}$ — **does NOT scale with wavelength**

The key consequence: In wavelength space, the derivative w.r.t. $v_\text{off}$ depends on both the center shift and the width change (since $\sigma_\lambda$ depends on $\mu$).

---

## Partial Derivatives

### Shared Intermediates

$$
\sigma_v = \frac{\mathrm{fwhm\_v}}{2\sqrt{2\ln 2}}
$$

$$
\sigma_{\lambda,\mathrm{v}} = \mu \cdot \frac{\sigma_v}{C_\text{km/s}}
$$

$$
\sigma_\lambda = \sqrt{\sigma_{\lambda,\mathrm{v}}^2 + \mathrm{dx}^2}
$$

$$
z = \frac{x - \mu}{\sigma_\lambda}, \qquad
f = \frac{g \cdot A}{\sigma_\lambda}\,e^{-z^2/2}
$$

---

### With respect to `strength`

The derivative w.r.t. strength is independent of instrumental resolution type:

$$
\boxed{
\frac{\partial f}{\partial A} = \frac{f}{A} = \frac{g}{\sigma_\lambda}\,e^{-z^2/2}
}
$$

---

### With respect to `fwhm_v`

We use the chain rule: $\frac{\partial f}{\partial \mathrm{fwhm\_v}} = \frac{\partial f}{\partial \sigma_\lambda} \cdot \frac{\partial \sigma_\lambda}{\partial \mathrm{fwhm\_v}}$

**Step 1:** Partial w.r.t. $\sigma_\lambda$ (same as velocity space):
$$
\frac{\partial f}{\partial \sigma_\lambda} = \frac{f}{\sigma_\lambda}\,(z^2 - 1)
$$

**Step 2:** Compute $\frac{\partial \sigma_\lambda}{\partial \mathrm{fwhm\_v}}$:

$$
\frac{\partial \sigma_{\lambda,\mathrm{v}}}{\partial \mathrm{fwhm\_v}} = \mu \cdot \frac{1}{C_\text{km/s}} \cdot \frac{1}{2\sqrt{2\ln 2}}
= \mu \cdot \frac{\text{FWHM\_TO\_SIGMA}}{C_\text{km/s}}
$$

$$
\frac{\partial \sigma_\lambda}{\partial \mathrm{fwhm\_v}} = \frac{\sigma_{\lambda,\mathrm{v}}}{\sigma_\lambda} \cdot \frac{\partial \sigma_{\lambda,\mathrm{v}}}{\partial \mathrm{fwhm\_v}}
= \frac{\sigma_{\lambda,\mathrm{v}}}{\sigma_\lambda} \cdot \mu \cdot \frac{\text{FWHM\_TO\_SIGMA}}{C_\text{km/s}}
$$

**Step 3:** Combine:

$$
\boxed{
\frac{\partial f}{\partial \mathrm{fwhm\_v}}
= f\,(z^2 - 1) \cdot \frac{\sigma_{\lambda,\mathrm{v}} \cdot \mu \cdot \text{FWHM\_TO\_SIGMA}}{C_\text{km/s} \cdot \sigma_\lambda^2}
}
$$

Alternatively, letting $k_\lambda = \sigma_{\lambda,\mathrm{v}} \cdot \mu / (C_\text{km/s} \cdot \sigma_\lambda^2) \cdot \text{FWHM\_TO\_SIGMA}$:

$$
\frac{\partial f}{\partial \mathrm{fwhm\_v}} = f\,(z^2 - 1) \cdot k_\lambda
$$

---

### With respect to `v_off`

The dependence on $v_\text{off}$ is more complex in wavelength space because $\sigma_\lambda$ depends on $\mu$, which depends on $v_\text{off}$.

$$
\frac{\partial \mu}{\partial v_\text{off}} = \frac{\lambda_0}{C_\text{km/s}} = \frac{\mu}{C_\text{km/s}(1 + v_\text{off}/C_\text{km/s})}
$$

Since both $z$ and $\sigma_\lambda$ depend on $\mu$, we use logarithmic differentiation:

$$
\ln(f) = \ln(g \cdot A) - \ln(\sigma_\lambda) - \frac{z^2}{2}
$$

$$
\frac{\partial \ln(f)}{\partial v_\text{off}} = -\frac{1}{\sigma_\lambda}\frac{\partial \sigma_\lambda}{\partial v_\text{off}} - z \frac{\partial z}{\partial v_\text{off}}
$$

where:

$$
\frac{\partial \sigma_{\lambda,\mathrm{v}}}{\partial v_\text{off}} = \sigma_v \cdot \frac{\partial \mu}{\partial v_\text{off}} \cdot \frac{1}{C_\text{km/s}}
= \sigma_v \cdot \frac{\lambda_0}{C_\text{km/s}^2}
$$

$$
\frac{\partial \sigma_\lambda}{\partial v_\text{off}} = \frac{\sigma_{\lambda,\mathrm{v}}}{\sigma_\lambda} \cdot \sigma_v \cdot \frac{\lambda_0}{C_\text{km/s}^2}
$$

$$
\frac{\partial z}{\partial v_\text{off}} = \frac{1}{\sigma_\lambda}\left[\frac{\partial \mu}{\partial v_\text{off}} + z \cdot \frac{\partial \sigma_\lambda}{\partial v_\text{off}}\right]
$$

After careful calculation (detailed algebraic steps omitted):

$$
\boxed{
\frac{\partial f}{\partial v_\text{off}}
= \frac{f \cdot \lambda_0}{C_\text{km/s} \cdot \sigma_\lambda} \left[z + (z^2 - 1) \cdot \frac{\sigma_{\lambda,\mathrm{v}} \cdot \sigma_v}{C_\text{km/s} \cdot \sigma_\lambda}\right]
}
$$

where $\sigma_v = \mathrm{fwhm\_v} / (2\sqrt{2\ln 2}) = \mathrm{fwhm\_v} \cdot \mathrm{FWHM\_TO\_SIGMA}$.

---

## Implementation Notes

The $v_\text{off}$ derivative has two distinct contributions:

1. **Linear term**: $z$ (proportional to wavelength shift)
2. **Nonlinear term**: $(z^2 - 1) \cdot \frac{\sigma_{\lambda,\mathrm{v}} \cdot \sigma_v}{C_\text{km/s} \cdot \sigma_\lambda}$ (proportional to width change)

In the Cython code, precompute constant coefficients:

```cython
cdef double coeff_v_off = norm * wave / (C_KMS * sigma_lambda)
cdef double ratio_v_off = (sigma_lambda_v * sigma_v) / (C_KMS * sigma_lambda)
```

Then in the loop:

```cython
cdef double z = (x[i] - mean) / sigma_lambda
cdef double z_sq = z * z
cdef double exp_term = exp(-0.5 * z_sq)
cdef double term_v_off = z + (z_sq - 1) * ratio_v_off
derivs[2, i] += coeff_v_off * exp_term * term_v_off
```

Note: This formula is significantly more complex than the velocity-space version because $\sigma_\lambda$ depends on $v_\text{off}$ through the observed wavelength $\mu$.

---

## Verification Checklist

- ✓ $\partial f/\partial A$ is independent of resolution type
- ✓ $\partial f/\partial \mathrm{fwhm\_v}$ involves $(z^2 - 1)$ factor, consistent with Gaussian theory
- ✓ $\partial f/\partial v_\text{off}$ reduces to velocity-space formula when $\mathrm{dx} \to 0$
- ✓ Units: $\sigma_{\lambda,\mathrm{v}}$ is in wavelength units (from $\mu \cdot \sigma_v / C_\text{km/s}$)
- ✓ Dimensions of partial derivatives match partial derivatives of flux (per wavelength per km/s per parameter unit)

