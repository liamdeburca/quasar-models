"""
Pydantic-compatible type annotation.
"""

from pathlib import Path
from typing import Self

from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import no_info_plain_validator_function

from .io import convert_path


class IronTemplateFile(Path):
    @classmethod
    def _validate(cls, value: object) -> Self:
        try:
            value = Path(value)
        except Exception as e:
            msg = (
                f"Value of type {type(value).__name__} could not be coerced to a Path."
            )
            raise PydanticCustomError("validation_error", msg) from e

        if value.is_absolute():
            if not value.suffix == ".fits":
                msg = (
                    f"Absolute file path must have a `.fits` extension, "
                    f"got {value.suffix}."
                )
                raise PydanticCustomError("validation_error", msg)

            if not value.exists():
                msg = f"Absolute file path does not exist: {value}."
                raise PydanticCustomError("validation_error", msg)

            return value

        return convert_path(str(value))

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return no_info_plain_validator_function(cls._validate)
