from quasar_models._core.utils cimport (
    multiply_and_add_to,
)
from quasar_models._core.convolution.utils cimport (
    _convolve_deriv,
    _identify_closest_idx_for_deriv,
)
from quasar_models._core.modeling.template.evaluate cimport (
    _evaluate_exact,
    _evaluate_interp,
)
from quasar_models._core.modeling.template.cytemplate cimport CyTemplate

### By CONVOLUTION

cdef void _fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _evaluate_exact(
        derivs[0, :],
        1.0, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_exact_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )

cdef void _fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    cdef double[::1] _df = _convolve_deriv(
        cytemplate.data,
        cytemplate.fwhm,
        fwhm,
        cytemplate.sigma_res,
        n_scales,
    )
    multiply_and_add_to(derivs[1, :], _df, flux)

def fit_deriv_exact_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_fwhm(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )

cdef void _fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_only_flux(
        derivs,
        flux, fwhm,
        cytemplate, n_scales,
    )
    _fit_deriv_exact_only_fwhm(
        derivs,
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_exact_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_exact_all(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )

### By INTERPOLATION

cdef void _fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _evaluate_interp(
        derivs[0, :],
        1.0,
        fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_only_flux(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )

cdef void _fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    cdef int idx
    cdef double fwhm0, fwhm1, k

    if cytemplate.fwhm.shape[0] > 1:
        # fwhm-derivative is possible    
        idx = _identify_closest_idx_for_deriv(cytemplate.fwhm, fwhm)
        if idx == cytemplate.fwhm.shape[0] - 1:
            idx -= 1

        fwhm0 = cytemplate.fwhm[idx]
        fwhm1 = cytemplate.fwhm[idx + 1]
        k = flux / (fwhm1 - fwhm0)

        multiply_and_add_to(derivs[1, :], cytemplate.data[idx + 1, :], k)
        multiply_and_add_to(derivs[1, :], cytemplate.data[idx, :], -k)

def fit_deriv_interp_only_fwhm(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_fwhm(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )

cdef void _fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_only_flux(
        derivs,
        flux, fwhm,
        cytemplate, n_scales,
    )
    _fit_deriv_interp_only_fwhm(
        derivs,
        flux, fwhm,
        cytemplate, n_scales,
    )

def fit_deriv_interp_all(
    double[:,::1] derivs,
    const double flux,
    const double fwhm,
    CyTemplate cytemplate,
    const double n_scales,
):
    _fit_deriv_interp_all(
        derivs, 
        flux, fwhm, 
        cytemplate, n_scales
    )