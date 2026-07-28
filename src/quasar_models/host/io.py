__all__ = [
    "PATH_TO_CACHE",
    "PATH_TO_DATA",
    "load",
    "load_from_cache",
    "save",
    "save_to_cache",
]

from pathlib import Path
from typing import Literal, Protocol

from astropy.io import fits
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_utils.setup import Info

from quasar_models.modeling.template.io import BaseTemplateProtocol, _load, _save

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parent / ".cache"
PATH_TO_DATA: Path = _this_file.parent / ".data"


class HostGalaxyTemplateProtocol(BaseTemplateProtocol, Protocol):
    name: Literal["bc2003"]
    age: int


def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(path.removesuffix(".fits") + ".fits")
        if "/" not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path


def convert_params_to_name(
    name: Literal["bc2003"],
    age: int,
) -> str:
    return f"{name}:age{age:0>14_}"


def convert_params_to_path(
    name: Literal["bc2003"],
    age: int,
) -> AbsoluteFITSPath:
    return PATH_TO_CACHE / (convert_params_to_name(name, age) + ".fits")


def save(
    *, template: HostGalaxyTemplateProtocol, path: str | AbsoluteFITSPath, info: Info
) -> AbsoluteFITSPath:
    path = convert_path(path)
    hdul = _save(
        template=template,
        info=info,
    )
    hdul[0].header["AGE"] = (template.age, "age (years)")
    hdul.writeto(path, overwrite=True)
    return path


def save_to_cache(
    *,
    template: HostGalaxyTemplateProtocol,
    info: Info,
) -> AbsoluteFITSPath:
    return save(
        template=template,
        path=convert_params_to_path(template.name, template.age),
        info=info,
    )


def load(
    *,
    path: str | AbsoluteFITSPath,
    info: Info,
) -> dict:
    path = convert_path(path)
    kwargs = _load(
        path=path,
        info=info,
    )
    with fits.open(path) as hdul:
        kwargs["age"] = hdul[0].header["AGE"]

    kwargs["x_norm"] = info.host.x_norm
    kwargs["fwhm_norm"] = info.host.fwhm_norm
    kwargs["normalisation"] = None

    return kwargs


def load_from_cache(
    *,
    name: str,
    age: int,
    info: Info,
) -> dict:
    return load(
        path=convert_params_to_path(name, int(age)),
        info=info,
    )


if __name__ == "__main__":
    PATH_TO_CACHE.mkdir(exist_ok=True)
