# The `PrepareModel` class

The goal of the `PrepareModel` class is to prepare models for fitting, or other events with repeated evaluation. For standard models, inheriting from the `QuasarModel` base class, this means:
1. Identifying the correct `evaluate_func` function (usually trivial).
2. Identifying the correct `fit_deriv_func` function based on the combination of fixed and free parameters.

For `TemplateModel` models, model preparation also includes calculating and saving transformation matrices, as the models will initially be evaluated on the inherent wavelength grid of the template, and then transformed to the provided wavelength grid. This avoids repeated calculation of the transformation matrices.

The `__init__` method does the following:
1. Sets the wavelength array. 
2. Sets the model instance, or a copy thereof if specified.
3. Saves whether to copy the model instance or not.

The `PrepareModel` class is designed to be used as a context manager. This way, `evaluate_func`, `fit_deriv_func` and transformation matrices are only used within the context manager, and properly discarded afterwards. 

## The `__enter__` method

The `__enter__` method does the following:
- Runs the `_prepare_model` method of each submodel. If the submodel is a subclass of `TemplateModel`, the wavelength grid is also passed. 

## The `__exit__` method

The `__exit__` method does the following:
- Runs the `_unprepare_model` method of each submodel.