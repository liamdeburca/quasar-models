from collections.abc import Iterator
from math import inf

from emcee import State
from numpy import dot, empty, float64, fromiter, ndim, stack, where
from quasar_typing.numpy import (
    BoolVector,
    FloatMatrix,
    FloatVector,
    RandomState_,
)
from scipy.linalg import det, inv
from scipy.stats import multivariate_normal
from tqdm import tqdm

from ..sequential_model import SequentialModel


def log_prior(
    x: FloatVector | list[FloatVector] | FloatMatrix, 
    model: SequentialModel,
    bounds: tuple[FloatVector, FloatVector] | None = None,
) -> float | FloatVector:
    """
    Compute the log prior probability of the parameters `x` given the model.
    """
    if bounds is None:
        bounds = model.bounds
    lbs, ubs = bounds

    if ndim(x) == 2:
        return where(
            (x < lbs).any(axis=1) | (x > ubs).any(axis=1), 
            -inf, 
            0.0,
        )

    if (x < lbs).any() or (x > ubs).any():
        return -inf
    return 0.0

def log_likelihood(
    x: FloatVector | list[FloatVector] | FloatMatrix,
    model: SequentialModel, 
    data: dict[str, FloatVector],
) -> float | FloatVector:
    """
    Compute the log likelihood of the parameters `x` given the model and data.
    """
    if ndim(x) == 2:
        out = empty(x.shape[0], dtype=float64)
        for i, xi in enumerate(x):
            z = model.fun(xi, data)
            out[i] = -0.5 * dot(z, z)
        return out
    
    z = model.fun(x, data)
    return -0.5 * dot(z, z)


def log_posterior(
    x: FloatVector | list[FloatVector] | FloatMatrix, 
    *,
    model: SequentialModel, 
    data: dict[str, FloatVector],
    bounds: tuple[FloatVector, FloatVector] | None = None,
) -> float | FloatVector:
    """
    Compute the log posterior probability of the parameters `x` given the model and data.
    """
    prob: float | FloatVector = log_prior(x, model, bounds=bounds)
    mask: bool | BoolVector = (prob != -inf)
    if (ndim(x) == 2) and mask.any():
        prob[mask] += log_likelihood(x[mask,:], model, data)
    elif mask:
        prob += log_likelihood(x, model, data)

    return prob

### INSTANTIATING SAMPLERS

def get_fisher_information_matrix(
    x: FloatVector, 
    model: SequentialModel, 
    data: dict[str, FloatVector],
) -> FloatMatrix:
    """
    Compute the Fisher information matrix for the parameters `x` given the model and data.
    """
    jac = model.jac(x, data)
    return jac.T @ jac


def get_covariance_matrix(fisher: FloatMatrix) -> FloatMatrix:
    if det(fisher) == 0.0:
        raise ValueError("Non-invertible Fisher information matrix.")
    cov = inv(fisher)
    if det(cov) == 0.0:
        raise ValueError("Non-invertible covariance matrix.")

    return cov

def get_parameter_samples(
    n_samples: int, 
    model: SequentialModel,
    data: dict[str, FloatVector],
    random_state: RandomState_,
    max_attempts: int | None = None,
) -> Iterator[FloatVector]:
    """
    Create 'n_samples' samples of the model parameters from a multivariate 
    normal distribution based on the Fisher information matrix. The samples are
    clipped using the parameter bounds of the model.
    """
    x0 = model.x0
    lb, ub = model.bounds
    assert not (lb == ub).any()

    fisher = get_fisher_information_matrix(x0, model, data)
    cov = get_covariance_matrix(fisher)
    dist = multivariate_normal(
        mean=x0, 
        cov=cov, 
        seed=random_state, 
        allow_singular=True,
    )

    n_created: int = 0
    n_attempts: int = 0

    loop = tqdm(
        range(n_samples), 
        desc="Generating parameter samples", 
        total=n_samples,
        leave=False,
    )
    while True:
        if max_attempts is not None and n_attempts >= max_attempts:
            loop.close()
            raise RuntimeError(
                f"Only created {n_created}/{n_samples} samples within "
                f"{max_attempts} attempts!"
            )

        x = dist.rvs()
        n_attempts += 1
        if (x < lb).any() or (x > ub).any():
            continue
        n_created += 1
        loop.update(1)
        yield x

        if n_created == n_samples:
            break
    loop.close()

def transform_samples_to_state(
    samples: list[FloatVector] | FloatMatrix,
    model: SequentialModel,
    data: dict[str, FloatVector],
    bounds: tuple[FloatVector, FloatVector] | None = None,
) -> State:
    coords = stack(samples, axis=0) if isinstance(samples, list) else samples
    log_prob = fromiter(
        (log_posterior(x, model=model, data=data, bounds=bounds) for x in coords),
        dtype=float64,
    )
    return State(coords, log_prob=log_prob)
