__all__ = [
    'PATH_TO_CACHE', 'PATH_TO_DATA',
    'convert_path', 'convert_params_to_path',
    'save', 'load',
    'save_to_cache', 'load_from_cache',
]

from typing import Protocol
from pathlib import Path
from astropy.io import fits
from astropy.units import Unit

from quasar_models.modeling.template.io import _save, _load, BaseTemplateProtocol
from quasar_typing.numpy import FloatVector
from quasar_typing.pathlib import AbsoluteFITSPath, AnyAbsoluteFITSPath

from quasar_utils.setup import Info

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parents[1] / ".cache"
PATH_TO_DATA: Path = _this_file.parents[1] / ".data"

class BalmerContinuumTemplateProtocol(BaseTemplateProtocol, Protocol):
    temp: float
    tau: float
    scale: float
    boltz: float

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(str(path).removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def convert_params_to_path(
    *,
    temp: float,
    tau: float,
    scale: float,
    info: Info,
) -> AnyAbsoluteFITSPath:    
    return PATH_TO_CACHE / \
        "continuum:edge{:.1f}_temp{:.1e}_tau{:.1f}_scale{:.1f}.fits".format(
            info.units.getWavelength(info.balmer.edge).to('angstrom').value,
            info.units.getTemperature(temp).to('K').value,
            info.units.getDensity(tau).to('cm^-3').value,
            scale,
        )

def save(
    *,
    template: BalmerContinuumTemplateProtocol,
    path: str | AbsoluteFITSPath,
    info: Info
) -> AbsoluteFITSPath:
    path = convert_path(path)
    hdul = _save(
        template=template,
        info=info,
    )

    t_unit: str = info.units.temp_unit.to_string()
    hdul[1].columns.add_col(fits.Column(
        name='temp',
        format='D',
        unit=t_unit,
        array=[template.temp],
    ))
    hdul[1].columns.add_col(fits.Column(
        name='tau',
        format='D',
        array=[template.tau],
    ))
    hdul[1].columns.add_col(fits.Column(
        name='scale',
        format='D',
        array=[template.scale],
    ))

    hdul.writeto(path, overwrite=True)
    return path

def save_to_cache(
    *,
    template: BalmerContinuumTemplateProtocol,
    info: Info,
) -> AbsoluteFITSPath:
    path = convert_params_to_path(
        temp=template.temp,
        tau=template.tau,
        scale=template.scale,
        info=info,
    )
    return save(
        template=template, 
        path=path,
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
    kwargs['x_norm'] = info.balmer.edge
    kwargs['fwhm_norm'] = info.balmer.fwhm_norm
    kwargs['normalisation'] = None
    kwargs['boltz'] = info.units.getBoltzmannFactor()
    
    with fits.open(path) as hdul:
        t_unit = Unit(hdul[1].columns[2].unit)

        def transform_temperature(arr: FloatVector) -> FloatVector:
            return info.units.getTemperature(arr * t_unit)

        kwargs['temp'] = transform_temperature(hdul[1].data['temp'])[0]
        kwargs['tau'] = hdul[1].data['tau'][0]
        kwargs['scale'] = hdul[1].data['scale'][0]

    return kwargs
    
def load_from_cache(
    *,
    temp: float,
    tau: float,
    scale: float,
    info: Info,
) -> dict:
    path = convert_params_to_path(
        temp=temp, 
        tau=tau, 
        scale=scale, 
        info=info,
    )
    return load(
        path=path,
        info=info,
    )