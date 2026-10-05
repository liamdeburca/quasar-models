In practice, fitting a (compound) model involves the following classes:
1. `PrepareModel`: prepares each submodel for repeated evaluation, and identifies the correct Cython functions to use for evaluation and partial derivatives.
2. `LMLSQFitter`/`TRFLSQFitter`/`DogboxLSQFitter`: wrappers for Scipy's `least_squares` function, which handle the fitting process.