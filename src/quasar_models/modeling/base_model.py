from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import ClassVar, Literal, Self

from astropy.modeling import Fittable1DModel
from numpy import float64, ndarray
from numpy.typing import NDArray
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function


class BaseModel(ABC, Fittable1DModel):
    model_type: ClassVar[Literal["pl", "fe", "ba", "hg", "em"]]
    
    def __iter__(self) -> Iterator[Self]:
        yield self

    @property
    def pure_name(self) -> str:
        return self.name.split("#")[0]

    @property
    @abstractmethod
    def sorting_key(self) -> tuple[float, float]:
        """
        Return a tuple used for sorting models.
        """

    @classmethod
    def _transform_if_ndarray(cls, param: float | NDArray[float64]) -> float:
        return param.item() if isinstance(param, ndarray) else param

    @classmethod
    def _transform_args_if_ndarray(cls, *args: float | NDArray[float64]) -> tuple[float, ...]:
        return tuple(cls._transform_if_ndarray(arg) for arg in args)

    # Evaluation

    @property
    @abstractmethod
    def evaluate_func(self) -> Callable[..., NDArray[float64]]: ...

    @abstractmethod
    def _choose_evaluate_func(self) -> None: ...

    @property
    @abstractmethod
    def fit_deriv_func(self) -> Callable[..., NDArray[float64]]: ...

    @abstractmethod
    def _choose_fit_deriv_func(self) -> None: ...

    @abstractmethod
    def evaluate(
        self,
        x: NDArray[float64],
        *params,
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]: ...

    @abstractmethod
    def fit_deriv(
        self,
        x: NDArray[float64],
        *args,
        derivs: NDArray[float64] | None = None,
    ) -> list[NDArray[float64]]: ...

    @abstractmethod
    def partial_deriv(
        self,
        x: NDArray[float64],
        *args,
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]: ...

    # Instance evaluation

    def instance_evaluate(
        self,
        x: NDArray[float64],
        y: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        return self.evaluate(
            x,
            *self.parameters,
            y=y,
        )

    def instance_fit_deriv(
        self,
        x: NDArray[float64],
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        return self.fit_deriv(
            x,
            *self.parameters,
            derivs=derivs,
        )

    def instance_partial_deriv(
        self,
        x: NDArray[float64],
        derivs: NDArray[float64] | None = None,
    ) -> NDArray[float64]:
        return self.partial_deriv(
            x,
            *self.parameters,
            derivs=derivs,
        )

    # Model preparation

    def prepare_model(self) -> None:
        self._choose_evaluate_func()
        self._choose_fit_deriv_func()

    def unprepare_model(self) -> None:
        del self.evaluate_func
        del self.fit_deriv_func

    # Pydantic validation

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = (
                f"Expected {cls.__name__} instance, got {type(value).__name__} instead."
            )
            raise PydanticCustomError("validation_error", msg)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)
