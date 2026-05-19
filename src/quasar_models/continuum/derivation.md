# PowerLaw Model — Mathematical Derivation

## Model

The `PowerLawModel` evaluates a power-law spectrum normalised at a reference
wavelength $x_0$:

$$
f(x) = F \left(\frac{x}{x_0}\right)^{\alpha}
$$

| Symbol | Role | Parameter |
|--------|------|-----------|
| $F$ | Amplitude (flux at $x_0$) | `flux` — **fittable** |
| $\alpha$ | Spectral index | `alpha` — **fittable** |
| $x_0$ | Reference wavelength | fixed constant |

---

## Partial Derivatives

Define the shared intermediate

$$
p(x) = \left(\frac{x}{x_0}\right)^{\alpha}
\quad \Longrightarrow \quad
f = F \cdot p.
$$

### With respect to `flux`

$$
\frac{\partial f}{\partial F} = p = \left(\frac{x}{x_0}\right)^{\alpha}
$$

### With respect to `alpha`

Using $\frac{\partial}{\partial \alpha}(x/x_0)^{\alpha} = (x/x_0)^{\alpha}\ln(x/x_0)$:

$$
\frac{\partial f}{\partial \alpha} = F \cdot p \cdot \ln\!\left(\frac{x}{x_0}\right)
$$

---

## Implementation Validation

The `fit_deriv` function in `evaluation.py` computes

```python
_f = (x / x0)**alpha          # shared intermediate p
df_dflux  = _f                # ∂f/∂F  = p          ✓
df_dalpha = _f * flux * log(x / x0)   # ∂f/∂α  = F·p·ln(x/x0)  ✓
```

Both derivatives are **correctly implemented**.
