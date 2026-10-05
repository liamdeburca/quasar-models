# Balmer Pseudo-Continuum Model — Mathematical Derivation

## Model

The `BalmerModel` combines a **Balmer continuum** template and a **Balmer
series** template, both convolved to a common FWHM, sharing a single flux
scale.  The relative contribution of the series template is set by `ratio`:

$$
f(x) = \frac{F}{N_c}\,\mathcal{C}_c(x;\Phi)
       + \frac{F\,\rho}{N_s}\,\mathcal{S}(x;\Phi)
$$

where:

- $\mathcal{C}_c(x;\Phi)$ — Balmer continuum template convolved to FWHM $\Phi$
- $\mathcal{S}(x;\Phi)$ — Balmer series template convolved to FWHM $\Phi$
- $N_c,\,N_s$ — respective normalisation constants

| Symbol | Role | Parameter |
|--------|------|-----------|
| $F$ | Overall flux scale | `flux` — **fittable** |
| $\Phi$ | Target FWHM | `fwhm` — **fittable** |
| $\rho$ | Series-to-continuum ratio | `ratio` — **fittable** |

Define the unit-flux convolved templates:

$$
C(x;\Phi) \equiv \frac{\mathcal{C}_c(x;\Phi)}{N_c}, \qquad
S(x;\Phi) \equiv \frac{\mathcal{S}(x;\Phi)}{N_s}
$$

so the model reads more compactly:

$$
f(x) = F\!\left[C(x;\Phi) + \rho\,S(x;\Phi)\right]
$$

---

## Partial Derivatives

### $\partial f / \partial F$ (flux)

$$
\frac{\partial f}{\partial F} = C(x;\Phi) + \rho\,S(x;\Phi)
$$

### $\partial f / \partial \Phi$ (fwhm)

$$
\frac{\partial f}{\partial \Phi}
= F\!\left[\frac{\partial C}{\partial \Phi} + \rho\,\frac{\partial S}{\partial \Phi}\right]
$$

where each FWHM derivative is obtained via the convolution derivative (see the
template evaluation derivation for details).

### $\partial f / \partial \rho$ (ratio)

$$
\frac{\partial f}{\partial \rho} = F\,S(x;\Phi)
$$

---

### Condensed form (shared intermediates)

```
S_1   = S(x; Φ)        [series template at unit flux]
C_1   = C(x; Φ)        [continuum template at unit flux]

f     = F · (C_1 + ρ S_1)

∂f/∂F = C_1 + ρ S_1
∂f/∂Φ = F · (∂C_1/∂Φ + ρ ∂S_1/∂Φ)
∂f/∂ρ = F · S_1
```

---

## Implementation Validation

### `evaluate` and `fit_deriv` in `balmer/evaluation.py`

The `evaluate` function:

```python
f_cont = template_evaluation.evaluate(x, flux, fwhm, template=continuum_template, ...)
f_series = template_evaluation.evaluate(x, flux, fwhm, template=series_template, ...)
return f_cont + ratio * f_series
```

This correctly computes $f = F(C + \rho S)$.

The `fit_deriv` function:

```python
df_dflux  = C_1 + ratio * S_1                                  # ✓
df_dfwhm  = flux * (∂C_1/∂Φ + ratio * ∂S_1/∂Φ)               # ✓
df_dratio = flux * S_1                                          # ✓
```

Both `evaluate` and `fit_deriv` are **correctly implemented** and **consistent**
with the intended formula $f = F[C + \rho S]$.
