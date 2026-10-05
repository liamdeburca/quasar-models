# Test: Runs

The goal of the first test ('runs') is to ensure that `SequentialModel` instances can be created successfully from single and compound models, and that they run successfully. For an instance creation to be successful, the `SequentialModel` should be instantiated without errors. For a `SequentialModel` to run successfully, its `evaluate` method should be called without errors, and the output should be equivalent to the output of the underlying model(s) that it was created from.

## Test 1: `test_from_powerlaw_model`

Tests whether a `SequentialModel` can be created from a single `PowerLawModel`.

In addition, verifies:

- `.n_submodels == 1`
- `._n == 2`
- `._n_free == 2`
- `._n_tied == 0`
- `._n_fixed == 0`

## Test 2: `test_from_gaussian_model`

Tests whether a `SequentialModel` can be created from a single `GaussianModel`.

In addition, verifies:

- `.n_submodels == 1`
- `._n == 3`
- `._n_free == 3`
- `._n_tied == 0`
- `._n_fixed == 0`

## Test 3: `test_from_iron_model`

Tests whether a `SequentialModel` can be created from a single `IronModel`.

In addition, verifies:

- `.n_submodels == 1`
- `._n == 5`
- `._n_free == 2`
- `._n_tied == 0`
- `._n_fixed == 3`

## Test 4: `test_from_balmer_model`

Tests whether a `SequentialModel` can be created from a single `BalmerModel`.

In addition, verifies:

- `.n_submodels == 1`
- `._n == 3`
- `._n_free == 3`
- `._n_tied == 0`
- `._n_fixed == 0`

## Test 5: `test_from_host_galaxy_model`

Tests whether a `SequentialModel` can be created from a single `HostGalaxyModel`.

In addition, verifies:

- `.n_submodels == 1`
- `._n == 2`
- `._n_free == 1`
- `._n_tied == 0`
- `._n_fixed == 1`

## Running the tests

Use the conda environment `quasar` and the command `pytest -v tests/sequential_model/test_runs.py` to run the tests.
