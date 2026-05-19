# Host Galaxy Model — Mathematical Derivation

## Model

The `HostGalaxyModel` evaluates a host galaxy spectral template convolved to a
given FWHM.  It is mathematically identical in structure to the iron template
model (without the split function):

$$
f(x) = \frac{F}{N}\,\mathcal{I}\!\left[\left(T_0 * G_{\Phi_k}\right)\right](x)
$$

where:

- $T_0(x')$ — template row at the closest available FWHM $\Phi_0 \le \Phi$
- $\Phi_k = \sqrt{\Phi^2 - \Phi_0^2}$ — residual Gaussian kernel FWHM
- $G_{\Phi_k}$ — Gaussian broadening kernel
- $\mathcal{I}$ — interpolation operator from template wavelength grid to data
  grid
- $N$ — template normalisation constant (fixed)

| Symbol | Role | Parameter |
|--------|------|-----------|
| $F$ | Overall flux scale | `flux` — **fittable** |
| $\Phi$ | Target FWHM | `fwhm` — **fittable** |
| $N$ | Template normalisation | fixed constant |

---

## Partial Derivatives

Let $\mathcal{C}[\cdot]$ abbreviate "convolve with $G_{\Phi_k}$ then apply
$\mathcal{I}$".

### $\partial f / \partial F$ (flux)

$$
\frac{\partial f}{\partial F} = \frac{1}{N}\,\mathcal{C}[T_0]
$$

### $\partial f / \partial \Phi$ (fwhm)

The FWHM dependence enters through the kernel
$G_{\Phi_k}$ with $\Phi_k = \sqrt{\Phi^2 - \Phi_0^2}$, so
$\partial \Phi_k / \partial \Phi = \Phi / \Phi_k$.

$$
\frac{\partial f}{\partial \Phi}
= \frac{F}{N}\,\mathcal{I}\!\left[T_0 * \frac{\partial G_{\Phi_k}}{\partial \Phi_k}\right]
  \cdot \frac{\Phi}{\Phi_k}
$$

In the interpolation mode (`evaluate_interp`), the template rows are linearly
interpolated in FWHM rather than convolved:

$$
T_\Phi = T_{\Phi_i} + \frac{T_{\Phi_{i+1}} - T_{\Phi_i}}{\Phi_{i+1} - \Phi_i}(\Phi - \Phi_i)
$$

$$
\frac{\partial f_\text{interp}}{\partial F} = \frac{T_\Phi}{N},
\qquad
\frac{\partial f_\text{interp}}{\partial \Phi} = \frac{F}{N}\,\mathcal{I}\!\left[\frac{T_{\Phi_{i+1}} - T_{\Phi_i}}{\Phi_{i+1} - \Phi_i}\right]
$$

---

## Implementation Validation

The `host/evaluation.py` module delegates **entirely** to
`utils/template/evaluation.py`:

```python
def evaluate(x, flux, fwhm, *, host_galaxy_template, ...):
    return evaluation.evaluate(x, flux, fwhm, template=host_galaxy_template, ...)

def fit_deriv(x, flux, fwhm, *, host_galaxy_template, ..., fixed):
    return evaluation.fit_deriv(x, flux, fwhm, template=host_galaxy_template, ..., fixed=fixed)
```

In `utils/template/evaluation.py`:

```python
# evaluate
f = convolve(template_data, template_fwhm, fwhm, sigma_res)
return flux * transform(f) / normalisation                           # ✓

# fit_deriv
df_dflux = transform(convolve(data, fwhm, sigma_res)) / normalisation  # ✓
df_dfwhm = flux * transform(convolve_deriv(data, fwhm, sigma_res)) / normalisation  # ✓
```

Both derivatives correctly include the division by $N$.  The host galaxy
`fit_deriv` is **correctly implemented**.
