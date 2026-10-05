from .utils import TemplateEvaluate, TemplateFitDeriv

### By CONVOLUTION

evaluate_exact = TemplateEvaluate("evaluate_exact")
fit_deriv_exact_only_flux = TemplateFitDeriv("fit_deriv_exact_only_flux")
fit_deriv_exact_only_fwhm = TemplateFitDeriv("fit_deriv_exact_only_fwhm")
fit_deriv_exact_all = TemplateFitDeriv("fit_deriv_exact_all")

### By INTERPOLATION

evaluate_interp = TemplateEvaluate("evaluate_interp")
fit_deriv_interp_only_flux = TemplateFitDeriv("fit_deriv_interp_only_flux")
fit_deriv_interp_only_fwhm = TemplateFitDeriv("fit_deriv_interp_only_fwhm")
fit_deriv_interp_all = TemplateFitDeriv("fit_deriv_interp_all")
