# The `QuasarModel` Class

The objective of the `QuasarModel` class is to create a more efficient modelling framework for fitting quasar spectra, while still relying on Astropy's model framework (API). 

This markdown file describes the `QuasarModel` class, specifically the (abstract) methods and properties that are required to be defined. 

## Basic Properties

### `pure_name`

The `pure_name` property is a substring of the `name` property, containing only the characters before the first `#` character appears.

### `sorting_key`

The `sorting_key` property is a 2-tuple of floats used to sort models. Models are sorted based on their types (the first element). For submodels that are often used in large numbers, specifically emission line models, the second element helps sort the models based on their wavelength. 

## Evaluation Properties

### `evaluate_func`

Returns a function that evaluates the model's contribution to the combined signal. The value of the property is set by the `_choose_evaluate_func` method, which is called by the `prepare_model` method.

### `fit_deriv_func`

Returns a function that evaluates the partial derivatives of the model's contribution w.r.t. each parameter. The value of the property is set by the `_choose_fit_deriv_func` method, which is called by the `prepare_model` method.

## Model Preparation

### `prepare_model`

Calls the `_choose_evaluate_func` and `_choose_fit_deriv_func` methods to set the `evaluate_func` and `fit_deriv_func` properties.

### `unprepare_model`

Resets the `evaluate_func` and `fit_deriv_func` properties to their default values. The default values are defined on the subclass.

## Evaluation Methods

### `evaluate`

Evaluates the model's contribution to the combined signal. All method signatures adhere to the following pattern of arguments:
- `x`: The wavelength array (1D).
- `*params`: The model's parameters.
- `y` (optional): The current combined signal array for inplace evaluation, i.e. a new array is not created, but `y` is updated and returned. If `y` is not provided, a new array is created and returned.

### `fit_deriv`

Creates a list of each of the partial derivatives of the model's contribution w.r.t. each parameter. All method signatures adhere to the following pattern of arguments:
- `x`: The wavelength array (1D).
- `*params`: The model's parameters.
- `derivs` (optional): This method is primarily used by Astropy's fitting framework, where inplace evaluation is not used. This argument is included for consistency. 

**Note:** The `fit_deriv` method corresponds to creating a list of the output of the `partial_deriv` method.

### `partial_deriv`

Creates a 2d array of the partial derivatives of the model's contribution w.r.t. each parameter. All method signatures adhere to the following pattern of arguments:
- `x`: The wavelength array (1D).
- `*params`: The model's parameters.
- `derivs` (optional): This method is only used by the `SequentialModel` class, where inplace evaluation is used. If `derivs` is not provided, a new array is created and returned. If `derivs` is provided, it is updated and returned.

## Instance Evaluation Methods

All instance evaluation methods `instance_evaluate`, `instance_fit_deriv`, and `instance_partial_deriv` are identical to their counterparts, except that the `*params` argument is not included. Instead, all parameter values are extracted from the model instance.

These methods are primarily used by the `SequentialModel` class.

Although these methods are defined on the parent class, and will work on any subclasses, it may still be worthwhile overriding them to avoid the use of the `parameters` property which creates a numpy array. 

## Pydantic Validation Methods

Basic Pydantic validation methods are defined on the `QuasarModel` class, and are inherited by all subclasses. Validation is basic (simple type validation), with no coercion.