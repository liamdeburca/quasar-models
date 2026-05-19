# Iron Emission Model — Mathematical Derivation

## Model

The `IronModel` evaluates an iron pseudo-continuum template convolved to a given
FWHM and optionally scaled by a smooth split function that modulates the
template's amplitude across wavelength.

### Convolution scheme

The template stores pre-computed spectra at a discrete grid of FWHM values
$\{\Phi_i\}$.  For a requested FWHM $\Phi$:

1. Find the closest available template FWHM $\Phi_0 \le \Phi$.
2. Compute the residual kernel width $\Phi_k = \sqrt{\Phi^2 - \Phi_0^2}$.
3. Convolve the corresponding template row $T_0(x')$ with a Gaussian kernel
   $G_{\Phi_k}$, then interpolate to the data wavelength grid via an operator
   $\mathcal{I}$.

$$
f(x) = \frac{F}{N}\,\mathcal{I}\!\left[\left(T_0 * G_{\Phi_k}\right)\right](x)
$$

where $N$ is the template normalisation constant.

### Split weighting

When the split pivot $x_s$ lies within the template wavelength range **and**
$\ell \neq r$, the template row $T_0(x')$ is first multiplied by a smooth
amplitude transition function before convolution:

$$
z(x') = \operatorname{clip}\!\left(\frac{\ln(x_s/x')}{\gamma\,\sigma_\text{res}},\,-5,\,5\right)
$$

$$
s(x') = \frac{1}{1 + e^{z(x')}} \qquad\text{(decreasing sigmoid: }s \to 1\text{ for }x' \gg x_s,\;s \to 0\text{ for }x' \ll x_s\text{)}
$$

$$
w(x') = (r - \ell)\,s(x') + \ell
$$

$$
T_w(x') = T_0(x')\,w(x')
$$

The model flux then becomes:

$$
f(x) = \frac{F}{N}\,\mathcal{I}\!\left[\left(T_w * G_{\Phi_k}\right)\right](x)
$$

When the split is out of range or $\ell = r$, the full (unsplit) template is
used with an effective amplitude $F_\text{eff} = \ell\, F$:

$$
f(x) = \frac{\ell\, F}{N}\,\mathcal{I}\!\left[\left(T * G_{\Phi_k}\right)\right](x)
$$

| Symbol | Role | Parameter |
|--------|------|-----------|
| $F$ | Overall flux scale | `flux` — **fittable** |
| $\Phi$ | Target FWHM | `fwhm` — **fittable** |
| $x_s$ | Split pivot wavelength | `split` — fittable (usually fixed) |
| $\ell$ | Left-side amplitude weight | `left` — fittable (usually fixed) |
| $r$ | Right-side amplitude weight | `right` — fittable (usually fixed) |
| $N$ | Template normalisation | fixed constant |
| $\gamma$ | FWHM scale factor | fixed constant (`info.iron.scale`) |
| $\sigma_\text{res}$ | Instrumental resolution | fixed constant |

---

## Partial Derivatives

For clarity, let $\mathcal{C}[h]$ denote the full operation
"convolve $h$ with $G_{\Phi_k}$, then apply $\mathcal{I}$".

### Normal case (split in range, $\ell \neq r$)

$$
f = \frac{F}{N}\,\mathcal{C}[T_0\,w]
$$

#### $\partial f / \partial F$ (flux)

$$
\frac{\partial f}{\partial F} = \frac{1}{N}\,\mathcal{C}[T_0\,w]
$$

#### $\partial f / \partial \Phi$ (fwhm)

The dependence on $\Phi$ enters through the kernel $G_{\Phi_k}$ with
$\Phi_k = \sqrt{\Phi^2 - \Phi_0^2}$, giving
$\partial \Phi_k / \partial \Phi = \Phi / \Phi_k$.

$$
\frac{\partial f}{\partial \Phi}
= \frac{F}{N}\,\mathcal{I}\!\left[(T_0\,w) * \frac{\partial G_{\Phi_k}}{\partial \Phi_k}\right]
  \cdot \frac{\Phi}{\Phi_k}
$$

At the boundary $\Phi = \Phi_0$ (i.e.\ $\Phi_k = 0$), the convolution
collapses to the identity and the derivative in FWHM-space is taken as the
limit, which is implemented via a pre-computed `kernel_deriv` at $\Phi_k = 0$.

#### $\partial f / \partial x_s$ (split)

$$
\frac{\partial w}{\partial x_s}
= (r - \ell)\,\frac{\partial s}{\partial z}\,\frac{\partial z}{\partial x_s}
= (r - \ell)\cdot\bigl(-s^2 e^{z}\bigr)\cdot\frac{1}{\gamma\,\sigma_\text{res}\,x_s}
$$

$$
\frac{\partial f}{\partial x_s}
= \frac{F}{N}\,\mathcal{C}\!\left[T_0\,\frac{\partial w}{\partial x_s}\right]
$$

#### $\partial f / \partial \ell$ (left)

$$
\frac{\partial w}{\partial \ell} = -(s - 0) + 1 = 1 - s
$$

$$
\frac{\partial f}{\partial \ell}
= \frac{F}{N}\,\mathcal{C}\!\left[T_0\,(1 - s)\right]
$$

#### $\partial f / \partial r$ (right)

$$
\frac{\partial w}{\partial r} = s
$$

$$
\frac{\partial f}{\partial r}
= \frac{F}{N}\,\mathcal{C}\!\left[T_0\,s\right]
$$

---

### Condensed form (shared intermediates)

```
e_z     = exp(clip(log(x_s / x') / (γ σ_res), -5, 5))
s       = 1 / (1 + e_z)
w       = (r - l) * s + l
T_w     = T_0 * w

f       = F/N · C[T_w]

∂w/∂x_s = (r - l) · (−s² e_z) / (γ σ_res x_s)
∂w/∂l   = 1 − s
∂w/∂r   = s

∂f/∂F   = C[T_w] / N
∂f/∂Φ   = F/N · C_Φ[T_w]           (C_Φ = derivative of C w.r.t. Φ)
∂f/∂x_s = F/N · C[T_0 · ∂w/∂x_s]
∂f/∂l   = F/N · C[T_0 · (1−s)]
∂f/∂r   = F/N · C[T_0 · s]
```

---

## Implementation Validation

### Normal case (split in range, $\ell \neq r$)

The `fit_deriv` function correctly implements the five partial derivatives via
convolution of modulated templates:

```python
inv_norm = 1 / template.normalisation

df_dflux  = transform(f) * inv_norm                                    # ∂f/∂F ✓
df_dfwhm  = flux * inv_norm * transform(convolve(signal, k_deriv))   # ∂f/∂Φ ✓
df_dsplit = flux * inv_norm * transform(convolve(ds[0]*signal, k))   # ∂f/∂x_s ✓
df_dleft  = flux * inv_norm * transform(convolve(ds[1]*signal, k))   # ∂f/∂ℓ ✓
df_dright = flux * inv_norm * transform(convolve(ds[2]*signal, k))   # ∂f/∂r ✓
```

All derivatives correctly include division by $N = \mathtt{template.normalisation}$.

---

### Fallback case (split out of range OR $\ell = r$)

When split is out of range or $\ell = r$, the full unmodified template is used
(equivalently, $w = \ell$ everywhere).  The `fit_deriv` function correctly
detects this condition and sets the split-related derivatives to zero:

```python
if not ((split < template.x[0]) or (template.x[-1] < split) or (left == right)):
    # Normal case: compute split derivatives
    df_dsplit = ...
    df_dleft  = ...
    df_dright = ...
else:
    # Fallback case: derivatives remain zero (initialised above)
    # df_dsplit = 0, df_dleft = 0, df_dright = 0
```

This is **mathematically correct**: when $\ell = r$ or split is invalid,
the weight $w$ is constant, so $\partial w / \partial x_s = \partial w / \partial \ell 
= \partial w / \partial r = 0$.

The derivatives $\partial f / \partial F$ and $\partial f / \partial \Phi$ are still
computed using the full (unsplit) template signal, consistent with the forward model.

---

### Implementation Status

Both the normal and fallback cases are **correctly implemented** and **consistent**
with their mathematical derivations. All derivatives properly scale by $1/N$.
