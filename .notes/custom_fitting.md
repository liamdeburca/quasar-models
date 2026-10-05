# Custom Fitting Package

## Overall Goal

The goal of the custom fitting package is to provide a more correct alternative to the flexible Astropy fitting module, while still relying on Astropy's model framework (API). The key improvement lies in the interpretation of 'tied' parameters, i.e. parameters whose values depend on one or multiple other parameters.

## Issues with `astropy.modeling`

In essence, the Astropy modeling framework translates compound models into functions which are then treated by Scipy's `least_squares` function. It needs:

- `fun`: a function that computes the array of residuals.
- `jac`: a function that computes the Jacobian matrix of the residuals w.r.t. the parameters.

The issue with the standard Astropy implementation is that, if parameter 'y' is tied to parameter 'x', the Jacobian matrix should reflect that a change to the model prediction through 'x' will also change the model prediction through 'y'. As Astropy has no way of identifying the relation between 'x' and 'y', it will not include the secondary effect of 'x'. This is wrong.

## My Situation

In my work, I only use simple, linear relations between tied parameters:

$$y = a \cdot x\,(+ b)$$

As a result, as long as I can enforce that the tied parameters are linear functions of the free parameters, I can compute the Jacobian matrix correctly. For example, the derivative of the model prediction at index 'i' w.r.t. parameter 'x' is now:

$$\frac{\partial f_i}{\partial x} = \frac{\partial M_i}{\partial x} + a \cdot \frac{\partial M_i}{\partial y}$$

where the second term is not included in the Astropy implementation.

## The Astropy Model Framework

For all intents and purposes, the Astropy consists of the following building blocks:

1. `CompoundModel`: A collection of individual, in my case, `Fittable1DModel` models.
2. `Fittable1DModel`: A class with defined `Parameter` objects, a defined `evaluate` method (calculates the model's contribution), and optionally, a defined `fit_deriv` method (calculates the partial derivative of the model's contribution w.r.t. each parameter). 
3. `Parameter`: A class describing each model parameter. There a three types:
    - 'free': no restrictions. 
    - 'fixed': the value of the parameter is fixed, or frozen.
    - 'tied': the value of the parammeter depends on the model.  

The compound models I create are all sequential, i.e. all submodels are added in series. This makes evaluating the entire compound model, and calculating its partial derivates (assuming `fit_deriv` methods are all defined) simple. 

The role of the compound model is to act as a bridge between Scipy's `least_squares` and the individual models:

- As only fixed parameters can be updated, it needs to translate between a truncated parameter array used by Scipy, and the full parameter array used by Astropy models. 
- It also needs to update the tied parameters.
- It needs to calculate the residual array using the combined model contributions.
- It needs to calculate the combined Jacobian matrix using only rows corresponding to free parameters.

## My Solution

The implementation consists of 3 main components:

1. **`SequentialModel`**: A new and simplified `CompoundModel`-like class that takes advantage of the sequential model structure (all '+' operators).
2. **`BaseModel`**: An abstract base class extending Astropy's `Fittable1DModel` that defines the interface for submodels to be used in `SequentialModel`.
3. **`LinearTie`**: A callable class used for tying parameters with linear relationships.

### `LinearTie`

A callable pydantic dataclass with the following coefficients defining a linear relation:

- `a` (float): Slope coefficient
- `b` (float): Intercept

$$y = a\cdot x + b$$

where $y$ is the tied parameter's value. The class tells the `SequentialModel` how to find $x$ via:

- `model_name` (str): Name of the submodel containing the free parameter
- `parameter_name` (str): Name of the free parameter

The `__call__` method computes the tied parameter's value directly from a compound model. The `__bool__` method always returns `True` to differentiate from `False` or `None`.

When the `SequentialModel` is instantiated, it validates that:

- All tied parameters point to free parameters (not fixed, not already tied)
- All referenced submodels exist and are unique
- Tie chains (parameter tied to another tied parameter) are not allowed

#### Combining Linear Ties

In certain cases, a parameter A ($p_{A}$) may be tied to parameter B ($p_{B}$) with coefficients $a_{A}$ and $b_{A}$, which is itself tied to parameter C ($p_{C}$) with coefficients $a_{B}$ and $b_{B}$. In this case, the `SequentialModel` will combine the two ties into a single tie for A, using the following logic:

$$p_{A} = a_{A} \cdot p_{B} + b_{A} = a_{A} \cdot (a_{B} \cdot p_{C} + b_{B}) + b_{A} = (a_{A} \cdot a_{B}) \cdot p_{C} + (a_{A} \cdot b_{B} + b_{A})$$

Therefore:

$$a_{A}' = a_{A} \cdot a_{B}$$
$$b_{A}' = a_{A} \cdot b_{B} + b_{A}$$

The general case, for an arbitrary number of ties, the new coefficients are:

$$a_{A}' = \prod_{i=1}^{n} a_{i}$$
$$b_{A}' = \sum_{i=1}^{n} b_{i} \cdot \prod_{j=1}^{i-1} a_{j}$$

where $n$ is the number of ties in the chain, and the empty product (when $i=1$) evaluates to 1.

#### Updating Parameter Bounds

When tying two parameters, the bounds of the free parameter should be truncated based on the bounds of the tied parameter. This way, all parameter values will remain within their bounds, even when applied to the original model.

Let's say that parameter A ($p_{A}$) is tied to parameter B ($p_{B}$) with coefficients $a$ and $b$. The bounds of $p_{A}$ are $[l_{A}, u_{A}]$, and the bounds of $p_{B}$ are $[l_{B}, u_{B}]$. Then, the bounds of $p_{B}$ should be updated as follows:

- If $a > 0$:
  - $l_{B}' = \max(l_{B}, (l_{A} - b) / a)$
  - $u_{B}' = \min(u_{B}, (u_{A} - b) / a)$
- If $a < 0$:
  - $l_{B}' = \max(l_{B}, (u_{A} - b) / a)$
  - $u_{B}' = \min(u_{B}, (l_{A} - b) / a)$


### `BaseModel`

An abstract base class extending Astropy's `Fittable1DModel` that defines the required interface for submodels used in `SequentialModel`. Concrete implementations must provide:

**Abstract Properties:**

- `evaluate_func`: Returns the callable that evaluates the model
- `fit_deriv_func`: Returns the callable that computes partial derivatives

**Abstract Methods:**

- `_choose_evaluate_func()`: Logic to select the appropriate evaluation function
- `_choose_fit_deriv_func()`: Logic to select the appropriate derivative function
- `evaluate(x, *params, y=None)`: Evaluates the model, optionally accumulating into output array `y`

The `evaluate` method supports in-place accumulation via the optional `y` parameter for memory efficiency in compound models.

**Utility Features:**
- `__iter__()`: Allows iteration, yielding self
- `pure_name` property: Extracts the model name without the instance identifier (before '#')
- `sorting_key` property: Abstract property for model ordering 

### `SequentialModel`

A dataclass-based implementation that manages the fitting workflow for sequential compound models.

#### Initialization & Validation

The `__init__` method:
1. Validates that compound models use only the '+' operator
2. Extracts submodels from the model structure
3. Collects all parameters across submodels
4. Validates all parameters via `_validate_params` (which checks initial values and tied attributes)
5. Prepares free, tied, and fixed parameter sets
6. Computes start/stop indices for parameter slicing
7. Validates submodel names match any LinearTie references

#### Parameter Classification

Parameters are classified into three mutually exclusive categories:

**Free Parameters** (`_prepare_free`):
- Not fixed and not tied
- Subject to optimization by scipy

**Tied Parameters** (`_prepare_tied`):
- Tied to a free parameter via `LinearTie`
- Computed dynamically from free parameters
- Validation ensures:
  - Target parameter is not fixed
  - No tie chains (tied to another tied parameter)

**Fixed Parameters** (`_prepare_fixed`):
- Explicitly fixed or tied to a fixed parameter
- Never updated during fitting

Each category stores:
- A numpy array of indices into the full parameter array (`_*_indices`)
- A tuple of the actual `Parameter` objects (`_*_parameters`)
- The count of parameters in that category (`_n_*`)

Tied parameters additionally store:
- `_tied_to_indices`: indices of the parameters they depend on
- `_tied_as`: coefficients (slope) for each tie relationship
- `_tied_bs`: intercepts for each tie relationship

#### Helper Methods

**`_get_full_parameter_array(free_params)`**:
- Expands a truncated free parameter array to the full parameter array
- Starts with initial values
- Sets free parameters from input
- Computes tied parameters using: `tied_i = a_i * full[tied_to_i] + b_i`
- This is the central mechanism for translating between scipy's truncated parameter space and the model's full parameter space

**`_calculate_start_stop_indices()`**:
- Pre-computes the parameter range for each submodel
- Enables efficient slicing of parameter arrays when calling submodel methods

#### Evaluation

**`evaluate(x, params)`**:
- Takes wavelength array `x` and free parameters `params`
- Expands `params` to full array using `_get_full_parameter_array`
- Initializes output array `y` with zeros
- For each submodel, extracts its parameters via slicing and calls `submodel.evaluate(x, *submodel_params, y=y)`
- Returns accumulated model predictions
- All submodel evaluations are in-place accumulations for efficiency

#### Partial Derivatives

**`partial_deriv(x, params)`**:
- Computes the full Jacobian matrix of all parameters (free + tied)
- Initializes a matrix of shape `(n_total_params, n_data_points)`
- For each submodel, calls `submodel.partial_deriv(x, *submodel_params, derivs=output_slice)`
- **Crucially, applies the chain rule for tied parameters**:
  - For each tied parameter: `full_derivs[tied_to_i] += a_i * full_derivs[tied_i]`
  - This accounts for the secondary effect of free parameters on tied parameters
  - This is the key improvement over standard Astropy fitting
- Returns only the rows corresponding to free parameters

**`fit_deriv(x, params)`**:
- Wrapper that returns the result of `partial_deriv` as a list (for Astropy compatibility) 

## Validation Functions

The module provides helper validation functions used during initialization:

**`_validate_compound_model(model)`**:
- Ensures the compound model uses only the '+' operator
- Traverses the tree from right to left (via the `.left` attribute)
- Raises `ValueError` if any other operator is found

**`_validate_initial_values(params)`**:
- Checks all parameter values are finite
- For non-fixed parameters, verifies values lie within bounds
- Ensures bounds are not identical (unless fixed)
- Raises `ValueError` for any violations

**`_validate_tied_parameters(params)`**:
- Checks all `tied` attributes are `False`, `None`, or `LinearTie` instances
- Raises `TypeError` if an invalid type is encountered
- (Currently skips warnings for fixed tied parameters; this is commented out)

**`_validate_submodel_names(submodels, linear_ties)`**:
- Verifies all `LinearTie` objects reference existing submodels
- Ensures referenced submodel names are unique (count == 1)
- Raises `ValueError` if references are missing or non-unique

## Integration with Scipy's `least_squares`

To actually use the model for fitting, the `SequentialModel` instance provides two methods:

**`fun(x, data)`**:
- Computes the weighted residual vector for scipy
- Takes free parameter values `x` and data dictionary with keys: `"x"` (wavelengths), `"y"` (data), `"w"` (weights)
- Computes residuals as: $z_i = w_i \cdot (f_i - y_i)$
- Uses einsum for efficient element-wise multiplication

**`jac(x, data)`**:
- Computes the weighted Jacobian matrix for scipy
- Takes the Jacobian from `partial_deriv` and applies weights
- Computes: $J_{ij} = w_i \cdot \frac{\partial f_i}{\partial p_j}$
- Transposes to scipy's expected shape (n_data, n_params)

**Properties for scipy integration:**
- `x0`: Initial free parameter values
- `bounds`: (lower_bounds, upper_bounds) for free parameters
- `parameters`: Property forwarding to underlying model's parameter array

**Additional Methods:**

**`get_final_model(params=None, copy=True)`**:
- Returns a new model instance with final fitted parameter values
- If `params` is provided and has size != `_n`, expands to full array
- Otherwise uses provided params or current values
- Can return a copy or the original model

**`_map_free_params(params)`**:
- Sets free parameter values on the underlying submodels
- Validates input shape and size
- Used primarily for updating model state

## Mathematical Definitions

To stay consistent with Astropy's API, the custom fitting package uses the following definitions:

**Residual Vector:**
$$z_{i} = w_{i} \cdot (f_{i} - y_{i})$$

where:
- $\vec{z}$ is the vector of residuals
- $\vec{f}$ is the model prediction
- $\vec{y}$ is the observed data
- $\vec{w}$ is the vector of weights (inverse uncertainties)

**Jacobian Matrix:**
$$J_{ij} = \frac{\partial z_{i}}{\partial p_{j}} = w_{i} \cdot \frac{\partial f_{i}}{\partial p_{j}}$$

where $p_{j}$ is the j-th free parameter.

**Key Improvement with Tied Parameters:**

When parameter $y$ is tied to parameter $x$ as $y = a \cdot x + b$, the Jacobian correctly includes both direct and indirect contributions:

$$\frac{\partial f_i}{\partial x} = \frac{\partial M_i}{\partial x} + a \cdot \frac{\partial M_i}{\partial y}$$

This is achieved by the chain rule application in `partial_deriv`.

## Utility Methods

**Dunder Methods:**
- `__str__()`: Returns string representation of underlying model
- `__repr__()`: Returns repr of underlying model
- `__getitem__(idx)`: Accesses submodels by index or name (like CompoundModel)

## Implementation Status

### Implemented ✓
- Full parameter classification system (free, tied, fixed)
- Tie chain validation
- In-place model evaluation with parameter slicing
- Correct Jacobian computation with tied parameter chain rules
- Full scipy integration via `fun` and `jac` methods
- Comprehensive input validation
- Efficient vectorized numpy operations

### Design Decisions Made
- Used `_get_full_parameter_array()` instead of separate `_calculate_*_indices` methods for cleaner implementation
- Leveraged vectorized numpy for tied parameter Jacobian correction instead of loops
- No separate `_map_tied_params()` since ties are handled in `_get_full_parameter_array()`
- Used dataclass instead of explicit class definition for cleaner initialization