__all__ = [
    'PATH_TO_CACHE', 'PATH_TO_DATA',
    'convert_path',
    'save', 'load',
    'save_to_cache', 'load_from_cache',
]

from typing import Protocol, Literal
from pathlib import Path

from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_utils.setup import Info
from quasar_models.modeling.template.io import _save, _load, BaseTemplateProtocol

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parent / ".cache"
PATH_TO_DATA: Path = _this_file.parent / ".data"

class IronTemplateProtocol(BaseTemplateProtocol, Protocol):
    name: Literal['vw2001', 'v2003', 'bw'] | str

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(path.removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def save(
    *, 
    template: IronTemplateProtocol, 
    path: str | AbsoluteFITSPath,
    info: Info,
) -> AbsoluteFITSPath:
    path = convert_path(path)
    hdul = _save(
        template=template,
        info=info,
    )
    hdul.writeto(path, overwrite=True)
    return path

def save_to_cache(
    *,
    template: IronTemplateProtocol,
    info: Info,
) -> AbsoluteFITSPath:
    return save(
        template=template,
        path=template.name,
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
    kwargs['x_norm'] = None
    kwargs['fwhm_norm'] = info.iron.fwhm_norm
    kwargs['normalisation'] = None
    return kwargs

def load_from_cache(
    *, 
    name: Literal['vw2001', 'v2003', 'bw'] | str, 
    info: Info,
) -> dict:
    return load(
        path=convert_path(name),
        info=info
    )

if __name__ == "__main__":
    PATH_TO_CACHE.mkdir(exist_ok=True)