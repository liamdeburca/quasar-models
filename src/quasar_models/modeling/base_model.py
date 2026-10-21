from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Literal, Self

from astropy.modeling import Fittable1DModel
from numpy import ndarray
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function
from quasar_typing.numpy import FloatArray
from quasar_utils.setup import Info


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
    def _transform_if_ndarray(cls, param: float | FloatArray) -> float:
        return param.item() if isinstance(param, ndarray) else param

    @classmethod
    def _transform_args_if_ndarray(
        cls, *args: float | FloatArray
    ) -> tuple[float, ...]:
        return tuple(cls._transform_if_ndarray(arg) for arg in args)

    # Evaluation

    @property
    @abstractmethod
    def evaluate_func(self) -> Callable[..., FloatArray]: ...

    @abstractmethod
    def _choose_evaluate_func(self) -> None: ...

    @property
    @abstractmethod
    def fit_deriv_func(self) -> Callable[..., FloatArray]: ...

    @abstractmethod
    def _choose_fit_deriv_func(self) -> None: ...

    @property
    @abstractmethod
    def kwargs(self) -> dict[str, Any]: ...

    @abstractmethod
    def _set_kwargs(self) -> None: ...

    @abstractmethod
    def evaluate(
        self,
        x: FloatArray,
        *params,
        y: FloatArray | None = None,
    ) -> FloatArray: ...

    @abstractmethod
    def fit_deriv(
        self,
        x: FloatArray,
        *args,
        derivs: FloatArray | None = None,
    ) -> list[FloatArray]: ...

    @abstractmethod
    def partial_deriv(
        self,
        x: FloatArray,
        *args,
        derivs: FloatArray | None = None,
    ) -> FloatArray: ...

    # Instance evaluation

    def instance_evaluate(
        self,
        x: FloatArray,
        y: FloatArray | None = None,
    ) -> FloatArray:
        return self.evaluate(
            x,
            *self.parameters,
            y=y,
        )

    def instance_fit_deriv(
        self,
        x: FloatArray,
        derivs: FloatArray | None = None,
    ) -> list[FloatArray]:
        return self.fit_deriv(
            x,
            *self.parameters,
            derivs=derivs,
        )

    def instance_partial_deriv(
        self,
        x: FloatArray,
        derivs: FloatArray | None = None,
    ) -> FloatArray:
        return self.partial_deriv(
            x,
            *self.parameters,
            derivs=derivs,
        )

    # Model preparation

    def prepare_model(
        self,
        fixed: dict[str, bool] | None = None,
    ) -> None:
        self._choose_evaluate_func()
        self._choose_fit_deriv_func(fixed=fixed)
        self._set_kwargs()

    def unprepare_model(self) -> None:
        del self.evaluate_func
        del self.fit_deriv_func
        del self.kwargs

    # Pydantic validation

    @classmethod
    def _validate(cls, value: object) -> Self:
        if not isinstance(value, cls):
            msg = (
                f"Expected {cls.__name__} instance, "
                f"got {type(value).__name__} instead."
            )
            raise PydanticCustomError("validation_error", msg)
        return value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        return no_info_plain_validator_function(cls._validate)

    ### Serialization

    @classmethod
    def _serialize_helper(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "name" not in data:
            raise ValueError("Missing 'name' key in serialization data.")
        return {cls.__name__: data}

    @abstractmethod
    def serialize(self, info: Info) -> dict[str, Any]:
        """
        Create a dictionary representation of the model instance.
        """

    @classmethod
    @abstractmethod
    def deserialize(cls, data: dict[str, Any], info: Info) -> Self:
        """
        Create a model instance from its dictionary representation.
        """
