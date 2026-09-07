from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from time import perf_counter
from typing import Union
from warnings import warn

from astropy.modeling import CompoundModel
from emcee import EnsembleSampler
from emcee.autocorr import AutocorrError
from numpy import array, float64, int_, ndim, ones, stack
from quasar_typing.numpy import (
    FloatCube,
    FloatMatrix,
    FloatVector,
    RandomState_,
)
from quasar_typing.scipy import OptimizeResult_

from ..base_model import BaseModel
from ..sequential_model import SequentialModel
from ..stop_conditions import (
    ftol_stop_batched,
    xtol_stop_batched,
)
from .utils import (
    get_parameter_samples,
    log_posterior,
    transform_samples_to_state,
)

logger = getLogger(__name__)


def _log_prob_fn(
    x: FloatVector | list[FloatVector] | FloatMatrix, 
    *,
    model: SequentialModel, 
    data: dict[str, FloatVector],
    bounds: tuple[FloatVector, FloatVector] | None = None,
) -> float:
    if ndim(x) == 2:
        return stack(
            [log_posterior(xi, model=model, data=data, bounds=bounds) for xi in x],
            axis=0,
            dtype=float64,
        )
    return log_posterior(x, model=model, data=data, bounds=bounds)

@dataclass
class MCMCFitter:
    fit_info: OptimizeResult_ | None = field(default=None, init=False)
    sol: FloatVector | None = field(default=None, init=False)

    sampler: EnsembleSampler | None = field(default=None, init=False)
    samples: FloatCube | None = field(default=None, init=False)
    ps: FloatVector | None = field(default=None, init=False)

    def __call__(
        self,
        model: Union[BaseModel, CompoundModel, SequentialModel],
        x: FloatVector,
        y: FloatVector,
        *,
        dy: FloatVector | None = None,
        weights: FloatVector | None = None,

        get_model: bool = False,
        inplace: bool | None = None,

        progress: bool = False,
        vectorize: bool = True,
        pool: object | None = None,

        random_state: RandomState_,
        n_walkers: int = 100,
        n_walkers_per_dim: int = 10,
        max_nfev: int = 1000,
        batch_size: int = 100,
        ftol: float | None = 1e-4,
        xtol: float | None = 1e-4,
        max_attempts: int | None = None,
    ) -> Union[BaseModel, CompoundModel, None]:
        """
        Fit the model to the data using MCMC sampling.
        """
        if not isinstance(model, SequentialModel):
            model = SequentialModel(model)

        bounds = model.bounds
        n_free = model._n_free
        n_walkers = max(n_walkers, n_walkers_per_dim * n_free)

        data: dict[str, FloatVector] = {
            "x": x,
            "y": y,
        }
        if (dy is None) and (weights is None):
            data["w"] = ones(x.size, dtype=float64)
        elif weights is not None:
            data["w"] = weights
        else:
            data["w"] = 1.0 / dy

        _samples = list(get_parameter_samples(
            n_walkers,
            model,
            data,
            random_state=random_state,
            max_attempts=max_attempts,
        ))

        _n = len(_samples)
        if _n == 0:
            msg = "Could not generate any valid initial parameter samples in "\
                f"{max_attempts} attempts."
            logger.critical(msg)
            raise ValueError(msg)
        elif _n < n_walkers:
            logger.debug(
                f"Only generated {_n} valid initial parameter samples "
                f"in {max_attempts} attempts."
            )

        n_walkers = min(n_walkers, _n)
        initial_state = transform_samples_to_state(_samples, model, data, bounds=bounds)

        t_start = perf_counter()
        self.sampler = EnsembleSampler(
            n_walkers,
            n_free,
            partial(_log_prob_fn, model=model, data=data, bounds=bounds),
            vectorize=vectorize,
            pool=pool,
        )
        self.autocorr_time = 1.0
        with suppress(AutocorrError):
            status: int = 0
            message = f"MCMC fitting did not converge within {max_nfev=} "\
                "iterations."

            if (xtol is None) and (ftol is None):
                batch_size = max_nfev

            count: int = 0
            while count < max_nfev:
                _ = self.sampler.run_mcmc(
                    initial_state, 
                    n := min(batch_size, max_nfev - count),
                    progress=progress,
                )
                count += n

                # Stop if conditions are satisfied for any of the walkers in the last batch
                ftol_stop = False
                if ftol is not None:
                    fs = self.sampler.get_log_prob()[-n:]   # (n, n_walkers)
                    ftol_stop = any(
                        ftol_stop_batched(ftol, fs[...,i]) 
                        for i in range(n_walkers)
                    )

                xtol_stop = False
                if xtol is not None:
                    xs = self.sampler.get_chain()[-n:]      # (n, n_walkers, n_free)
                    xtol_stop = any(
                        xtol_stop_batched(xtol, xs[:,i,:]) 
                        for i in range(n_walkers)
                    )

                if ftol_stop and xtol_stop:
                    status = 4
                    message = f"MCMC fitting stopped after {count} iterations "\
                        f"due to 'ftol' and 'xtol' conditions being satisfied."
                    break
                if ftol_stop:
                    status = 2
                    message = f"MCMC fitting stopped after {count} iterations "\
                        f"due to 'ftol' condition being satisfied."
                    break
                if xtol_stop:
                    status = 3
                    message = f"MCMC fitting stopped after {count} iterations "\
                        f"due to 'xtol' condition being satisfied."
                    break

            self.autocorr_time = self.sampler.get_autocorr_time() 

        t_elapsed: float = perf_counter() - t_start

        self.samples = self.sampler.get_chain(flat=True) # (nsteps x nwalkers, ndim)
        self.ps = self.sampler.get_log_prob(flat=True) # (nsteps x nwalkers)

        idx_best = self.ps.argmax()
        x_best = self.samples[idx_best]
        self.sol = model._get_full_parameter_array(x_best)

        active_mask: list[int] = []
        for val, lb, ub in zip(x_best, *model.bounds):
            if val <= lb:
                active_mask.append(-1)
            elif val >= ub:
                active_mask.append(1)
            else:
                active_mask.append(0)

        self.fit_info = OptimizeResult_(
            x_best,
            -self.ps[idx_best],
            fun=model.fun(x_best, data),
            jac=None,
            grad=None,
            optimality=None,
            active_mask=array(active_mask, dtype=int_),
            nfev=self.ps.size,
            njev=0,
            status=status,
            message=message,
            success=True,
        )

        chi2n = self.fit_info.reduced_chi2
        n_lb = self.fit_info.n_lb
        n_ub = self.fit_info.n_ub
        nfev = self.fit_info.nfev
        status = self.fit_info.status
        message = self.fit_info.message

        if self.fit_info.success:
            msg = f"OptimizeResult [SUCCESS ({status})]: "
            log = logger.debug
        else:
            msg = f"OptimizeResult [FAILURE ({status})]: "
            log = logger.info
        log(msg + f"chi2/dof={chi2n:.1f}, {nfev=}, {n_lb=}, {n_ub=}, {message=}, {t_elapsed=:.1f} ms")

        if get_model:
            if inplace is None:
                raise ValueError(
                    "If `get_model` is True, a boolean value must be provided "
                    "for the `inplace` argument."
                )

            return model.get_final_model(
                params=x_best,
                copy=not inplace,
            )
        elif inplace is not None:
            msg = "'inplace' should only be specified when 'get_model' is True."
            warn(msg, UserWarning)
