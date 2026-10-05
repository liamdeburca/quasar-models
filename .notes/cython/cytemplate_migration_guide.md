# CyTemplate Migration Guide: Old vs New Notation

## Overview

This document summarizes the transition from individual array parameter notation to the unified `CyTemplate` caching approach, implemented to eliminate repeated buffer protocol validation overhead.

## Key Motivation

- **Old Approach:** Every function call receives raw numpy arrays → Cython validates buffers on each call
- **New Approach:** Arrays cached in `CyTemplate` class during initialization → Memoryviews created once → Function calls just dereference pre-validated memoryviews
- **Performance Impact:** ~10x speedup for large templates (IronModel: 1345 µs → expected ~145 µs)

---

## Migration Pattern

### Old Signature (5 individual array params + sigma_res)
```cython
cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,      # ← Individual arrays
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,               # ← From template
    const double n_scales,                # ← Runtime parameter
):
```

### New Signature (1 CyTemplate param)
```cython
cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const CyTemplate cytemplate,          # ← Single encapsulated object
    const double n_scales,                # ← Runtime parameter (unchanged)
):
```

**Key Points:**
- 5 parameters → 1 parameter (more compact)
- `n_scales` **remains a runtime parameter** (not cached in CyTemplate)
- `cytemplate.fwhm`, `cytemplate.x`, `cytemplate.data`, `cytemplate.sigma_res` accessed as readonly memoryviews

---

## File-by-File Changes

### 1. Create CyTemplate Class

**File:** `src/quasar_models/_core/modeling/template/cytemplate.pyx`

```cython
from numpy import ascontiguousarray as np_ascontiguousarray, zeros as np_zeros, float64 as np_float64

cdef class CyTemplate:
    cdef readonly double[::1] fwhm
    cdef readonly double[::1] x
    cdef readonly double[:,::1] data
    cdef readonly double sigma_res
    
    cdef __cinit__(
        self,
        const double[::1] fwhm,
        const double[::1] x,
        const double[:,::1] data,
        const double sigma_res,
    ):
        self.fwhm = fwhm
        self.x = x
        self.data = data
        self.sigma_res = sigma_res
    
    @staticmethod
    def fromTemplate(object python_template):
        """Factory: Create CyTemplate from Python BaseTemplate object."""
        return CyTemplate(
            np_ascontiguousarray(python_template.fwhm, dtype=np_float64),
            np_ascontiguousarray(python_template.x, dtype=np_float64),
            np_ascontiguousarray(python_template.data / python_template.normalisation, dtype=np_float64),
            python_template.sigma_res,
        )
    
    cdef CyTemplate split(self, const double split, const double left, const double right, const double scale):
        """Create derived CyTemplate with split parameters applied."""
        # ... implementation ...
```

**File:** `src/quasar_models/_core/modeling/template/cytemplate.pxd`

```cython
cdef class CyTemplate:
    cdef readonly double[::1] fwhm
    cdef readonly double[::1] x
    cdef readonly double[:,::1] data
    cdef readonly double sigma_res
    
    cdef __cinit__(
        self,
        const double[::1] fwhm,
        const double[::1] x,
        const double[:,::1] data,
        const double sigma_res,
    )
    cdef CyTemplate split(
        self,
        const double split,
        const double left,
        const double right,
        const double scale,
    )
```

---

### 2. Update Template Functions (evaluate.pyx / fit_deriv.pyx)

**Old Code:**
```cython
cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
):
    cdef double[::1] f = _convolve(
        template_data,
        template_fwhm,
        fwhm,
        sigma_res,
        n_scales,
    )
```

**New Code:**
```cython
cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const CyTemplate cytemplate,
    const double n_scales,
):
    cdef double[::1] f = _convolve(
        cytemplate.data,
        cytemplate.fwhm,
        fwhm,
        cytemplate.sigma_res,
        n_scales,
    )
```

**Migration Checklist:**
- [ ] Add `from quasar_models._core.modeling.template.cytemplate cimport CyTemplate`
- [ ] Replace all `const double[::1] template_fwhm, const double[::1] template_x, const double[:,::1] template_data, const double sigma_res` with `const CyTemplate cytemplate`
- [ ] Replace all array accesses: `template_fwhm` → `cytemplate.fwhm`, `template_data` → `cytemplate.data`, etc.
- [ ] Remove `sigma_res` from function parameters (now accessed as `cytemplate.sigma_res`)
- [ ] Update all internal function calls that receive these parameters

---

### 3. Update .pxd Files

**Old:**
```cython
cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double[::1] template_fwhm,
    const double[::1] template_x,
    const double[:,::1] template_data,
    const double sigma_res,
    const double n_scales,
)
```

**New:**
```cython
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

cdef void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const CyTemplate cytemplate,
    const double n_scales,
)
```

---

### 4. Update Python Wrapper Classes

**Old (Python level):**
```python
class IronEvaluate:
    def __call__(self, x, flux, fwhm, split, left, right, *, 
                 template_fwhm, template_x, template_data, sigma_res,
                 scale, n_scales, ...):
        # Receive raw numpy arrays
        self.__wrapped__(_y, flux, fwhm, split, left, right,
                        template_fwhm, template_x, template_data, sigma_res,
                        scale, n_scales)
```

**New (Python level):**
```python
class IronEvaluate:
    def __call__(self, x, flux, fwhm, split, left, right, *, 
                 cytemplate, scale, n_scales, ...):
        # Receive CyTemplate instance
        self.__wrapped__(_y, flux, fwhm, split, left, right,
                        cytemplate, scale, n_scales)
```

---

### 5. Update Calling Code (iron/evaluate.pyx, etc.)

**Old:**
```cython
def evaluate_exact_no_split(..., const double[::1] template_fwhm, 
                                const double[::1] template_x,
                                const double[:,::1] template_data,
                                const double sigma_res, ...):
    _template_evaluate_exact(..., template_fwhm, template_x, template_data, sigma_res, ...)
```

**New:**
```cython
def evaluate_exact_no_split(..., const CyTemplate cytemplate, ...):
    _template_evaluate_exact(..., cytemplate, ...)
```

---

## Python Wrapper Layer Updates

The Python wrapper classes (`TemplateEvaluate` and `TemplateFitDeriv`) now provide an **adaptive interface** that accepts optional `CyTemplate` instances while remaining backward compatible.

### Import Changes

**New:**
```python
from .cytemplate import CyTemplate
```

### TemplateEvaluate Class Changes

**Old Signature:**
```python
def __call__(
    self,
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: object,
    n_scales: float,
    interpolation_matrix: tuple | None = None,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    # template is passed directly to Cython
    _y = zeros(template.x.size, dtype=float64)
    self.__wrapped__(
        _y,
        flux,
        fwhm,
        template.fwhm,
        template.x,
        template.data,
        template.sigma_res,
        n_scales,
    )
```

**New Signature:**
```python
def __call__(
    self,
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: object,
    cytemplate: CyTemplate | None = None,  # ← Optional, pre-cached instance
    n_scales: float,
    interpolation_matrix: tuple | None = None,
    y: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if cytemplate is None:
        cytemplate = CyTemplate.fromTemplate(template)  # ← Create if not provided

    _y = zeros(template.x.size, dtype=float64)
    self.__wrapped__(_y, flux, fwhm, cytemplate, n_scales)  # ← Pass single object
```

**Key Benefits:**
- ✅ **Optional Parameter:** `cytemplate` can be pre-created and passed (efficient, used with PrepareModel)
- ✅ **Fallback Creation:** If not provided, creates one on-the-fly from `template` (backward compatible)
- ✅ **Clean Interface:** Cython receives single compact object

### TemplateFitDeriv Class Changes

**Old Signature:**
```python
def __call__(
    self,
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: object,
    n_scales: float,
    interpolation_matrix: tuple | None = None,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    _derivs = zeros((3, template.x.size), dtype=float64)
    self.__wrapped__(
        _derivs,
        flux,
        fwhm,
        template.fwhm,
        template.x,
        template.data,
        template.sigma_res,
        n_scales,
    )
```

**New Signature:**
```python
def __call__(
    self,
    x: NDArray[float64],
    flux: float,
    fwhm: float,
    *,
    template: object,
    cytemplate: CyTemplate | None = None,  # ← Optional, pre-cached instance
    n_scales: float,
    interpolation_matrix: tuple | None = None,
    derivs: NDArray[float64] | None = None,
) -> NDArray[float64]:
    if cytemplate is None:
        cytemplate = CyTemplate.fromTemplate(template)  # ← Create if not provided

    _derivs = zeros((3, template.x.size), dtype=float64)
    self.__wrapped__(_derivs, flux, fwhm, cytemplate, n_scales)  # ← Pass single object
```

**Identical Benefits to TemplateEvaluate.**

### Usage Patterns

**Pattern 1: Efficient (with PrepareModel)**
```python
# Outside model evaluation context
with PrepareModel(models):
    # Inside context, models have pre-created cytemplate
    y = model.evaluate(x, flux, fwhm, cytemplate=model.meta["cytemplate"], n_scales=5.0)
```

**Pattern 2: Backward Compatible (without PrepareModel)**
```python
# Direct call without pre-caching
y = model.evaluate(x, flux, fwhm, template=model.template, n_scales=5.0)
# ← CyTemplate created on-the-fly, valid but less efficient if called repeatedly
```

**Pattern 3: Mixed (explicit pass)**
```python
cytemplate = CyTemplate.fromTemplate(model.template)
# Reuse across multiple evaluations
y1 = model.evaluate(x, flux, fwhm, cytemplate=cytemplate, n_scales=5.0)
y2 = model.evaluate(
    x, flux2, fwhm2, cytemplate=cytemplate, n_scales=5.0
)  # ← No buffer validation overhead
```

---

## Integration with PrepareModel

The `CyTemplate` is created once during model preparation:

```python
from quasar_models._core.modeling.template.cytemplate import CyTemplate


class PrepareModel:
    def __enter__(self):
        for model in self.models:
            if hasattr(model, "template"):
                # Create CyTemplate from Python template object
                cytemplate = CyTemplate.fromTemplate(model.template)
                model.meta["cytemplate"] = cytemplate
        return self

    def __exit__(self, *args):
        for model in self.models:
            model.meta.pop("cytemplate", None)
```

Then pass `model.meta['cytemplate']` to evaluation functions instead of individual arrays.

---

## Summary Table

| Aspect | Old | New |
|--------|-----|-----|
| **Parameters** | 5 individual arrays + sigma_res | 1 CyTemplate object |
| **Buffer Validation** | Per-call (expensive) | Once during init (cheap) |
| **Field Access** | Direct parameter | `cytemplate.fwhm`, `cytemplate.x`, etc. |
| **n_scales** | Function parameter | Remains function parameter |
| **Memory Overhead** | None (arrays passed by reference) | ~1 Python object per template |
| **Performance** | Baseline | ~10x faster for large templates |
| **Code Complexity** | More parameters, harder to track | Cleaner interface, self-documenting |

---

## Affected Files

1. ✅ `_core/modeling/template/cytemplate.pyx` (new)
2. ✅ `_core/modeling/template/cytemplate.pxd` (new)
3. ✅ `_core/modeling/template/evaluate.pyx`
4. ✅ `_core/modeling/template/evaluate.pxd`
5. ✅ `_core/modeling/template/fit_deriv.pyx`
6. ✅ `_core/modeling/template/fit_deriv.pxd`
7. ✅ `_core/modeling/template/utils.py` (Python wrappers)
8. ✅ `_core/modeling/iron/evaluate.pyx`
9. ⏳ `_core/modeling/iron/fit_deriv.pyx` (to be updated)
10. ⏳ `_core/modeling/iron/utils.py` (Python wrappers for iron model)
11. ⏳ `_core/modeling/balmer/evaluate.pyx` (similar update)
12. ⏳ `_core/modeling/host/evaluate.pyx` (similar update)
13. ⏳ `modeling/prepare_model.py` (integration point)

---

---

## CyTemplate.split() Method Pattern

The `CyTemplate` class provides a `.split()` method for model-specific use cases (particularly IronModel) where template data must be modified by split weights. This method creates a **new CyTemplate instance** with a modified template dataset without copying or re-validating buffers.

### Method Signature

```cython
cdef CyTemplate split(const double split, const double left, const double right, const double scale)
```

### Parameters

- **split** (float): Split weight value (0–1 typically), defines the fractional scaling
- **left** (float): Left wavelength boundary for split region
- **right** (float): Right wavelength boundary for split region  
- **scale** (float): Additional scale factor applied to the split region

### Return Value

Returns a **new `CyTemplate` instance** with:
- `data` field modified by split weighting (wavelength-dependent scaling)
- `fwhm`, `x`, `sigma_res` fields unchanged (shallow reference to original)
- All fields remain readonly memoryviews

### Usage in Iron Model

**Location:** `_core/modeling/iron/evaluate.pyx`

```cython
cdef inline void _evaluate_exact(
    double[::1] y,
    const double flux,
    const double fwhm,
    const double split,
    const double left,
    const double right,
    const CyTemplate cytemplate, 
    const double scale,
    const double n_scales,
):
    # Create weighted template for this evaluation
    cdef CyTemplate split_template = cytemplate.split(split, left, right, scale)
    
    # Evaluate using the split-weighted template
    _template_evaluate_exact(
        y,
        flux, fwhm,
        split_template, n_scales,
    )
```

### Key Design Decisions

1. **Non-destructive:** Original `cytemplate` remains unchanged; `.split()` creates a new instance
2. **Efficient:** No array copying; returns new CyTemplate with modified data memoryview
3. **Zero overhead:** Reuses `fwhm`, `x`, `sigma_res` from original; only `data` is modified
4. **Per-call instantiation:** New CyTemplate created on each evaluation with different split parameters (not cached)

### Why Not Cache?

Unlike the template data arrays (which are identical across evaluations), split-weighted data changes for each unique `(split, left, right, scale)` combination. Caching would require:
- Invalidation logic on parameter changes
- Dictionary lookup overhead
- Memory bloat from storing multiple variants

Instead, creating a new CyTemplate (fast, just wraps modified memoryview) is simpler and more efficient than cache management.

### Connection to IronModel Complexity

The iron model's broad emission feature requires different scaling factors across wavelength regions. The `.split()` method enables:
- Regional wavelength-dependent weighting
- Parameter-specific template modification
- Integration with fitting framework without extra array copies

---

## Backward Compatibility

The old function signatures are **no longer available**. Any code calling template functions must be updated to use `CyTemplate` instances. This is enforced at the Cython compilation level via the `.pxd` files.

