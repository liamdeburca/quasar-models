__all__ = [
    'PATH_TO_CACHE', 'PATH_TO_DATA',
    'convert_path', 'convert_params_to_path',
    'save', 'load',
    'save_to_cache', 'load_from_cache',
]

from typing import Protocol, Literal
from pathlib import Path
from astropy.io import fits
from astropy.units import Unit

from quasar_typing.numpy import SortedFloatVector, FloatVector
from quasar_typing.pathlib import AbsoluteFITSPath, AnyAbsoluteFITSPath

from quasar_utils.setup import Info

from ...utils.template.io import _save, _load, BaseTemplateProtocol

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parents[1] / ".cache"
PATH_TO_DATA: Path = _this_file.parents[1] / ".data"

class BalmerSeriesTemplateProtocol(BaseTemplateProtocol, Protocol):
    name: Literal['SH1995']

    waves: SortedFloatVector
    weights: FloatVector
    temp: float
    dens: float
    n_u_range: tuple[int, int]

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(str(path).removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def convert_params_to_path(
    *,
    name: Literal['sh1995'],
    temp: float,
    dens: float,
    n_u_range: tuple[int, int],
    info: Info,
    mode: Literal['save', 'load'],
) -> AnyAbsoluteFITSPath:
    _edge = int(info.units.getWavelength(info.balmer.edge).to('angstrom').value)
    _temp = int(info.units.getTemperature(temp).to('K').value)
    _dens = int(info.units.getDensity(dens).to('cm^-3').value)
    n_u_min = min(n_u_range)
    n_u_max = max(n_u_range)
    
    path = PATH_TO_CACHE / "series-{}:edge{}_temp{:.1e}_dens{:.1e}_nu{}-{}.fits".format(
        name, _edge, _temp, _dens, n_u_min, n_u_max,
    )
    if path.exists() or mode == 'save':
        return path

    def is_a_match(p: Path) -> bool:
        n_u_min = int(p.stem.split('_nu')[-1].split('-')[0])
        n_u_max = int(p.stem.split('_nu')[-1].split('-')[1])
        return n_u_min <= n_u_range[0] and n_u_max >= n_u_range[1]
    
    pattern = "series-{}:edge{}_temp{}_dens{}_nu*-*.fits".format(
        name, _edge, _temp, _dens,
    )
    for potential_path in PATH_TO_CACHE.glob(pattern):
        if is_a_match(potential_path):
            return potential_path
            
    msg = "No cached template for name={}, edge={}, temp={}, dens={}, " \
        "n_u_range={}".format(name, info.balmer.edge, temp, dens, n_u_range)
    raise FileNotFoundError(msg)

def save(
    *,
    template: BalmerSeriesTemplateProtocol,
    path: str | AbsoluteFITSPath,
    info: Info,
) -> AbsoluteFITSPath:
    path = convert_path(path)
    hdul = _save(
        template=template,
        info=info,
    )

    x_unit: str = info.units.wavelength_unit.to_string()
    t_unit: str = info.units.temp_unit.to_string()
    d_unit: str = info.units.dens_unit.to_string()

    hdul[0].header['N_WAVES'] = template.waves.size

    hdul[1].columns.add_col(fits.Column(
        name='waves',
        format='D',
        unit=x_unit,
        array=template.waves,
    ))
    hdul[1].columns.add_col(fits.Column(
        name='weights',
        format='D',
        array=template.weights,
    ))
    hdul[1].columns.add_col(fits.Column(
        name='temp',
        format='D',
        unit=t_unit,
        array=[template.temp],
    ))
    hdul[1].columns.add_col(fits.Column(
        name='dens',
        format='D',
        unit=d_unit,
        array=[template.dens],
    ))
    hdul[1].columns.add_col(fits.Column(
        name='n_u_range',
        format='I',
        array=list(template.n_u_range),
    ))

    hdul.writeto(path, overwrite=True)
    return path

def save_to_cache(
    *,
    template: BalmerSeriesTemplateProtocol,
    info: Info,
) -> AbsoluteFITSPath:
    path = convert_params_to_path(
        name=template.name,
        temp=template.temp,
        dens=template.dens,
        n_u_range=template.n_u_range,
        info=info,
        mode='save',
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

    with fits.open(path) as hdul:
        x_unit = Unit(hdul[1].columns[1].unit)
        t_unit = Unit(hdul[1].columns[4].unit)
        d_unit = Unit(hdul[1].columns[5].unit)

        def transform_wavelength(arr: FloatVector) -> FloatVector:
            return info.units.getWavelength(arr * x_unit)
        
        def transform_temperature(arr: FloatVector) -> FloatVector:
            return info.units.getTemperature(arr * t_unit)
        
        def transform_density(arr: FloatVector) -> FloatVector:
            return info.units.getDensity(arr * d_unit)
        
        n_waves = hdul[0].header['N_WAVES']

        kwargs['waves'] = transform_wavelength(hdul[1].data['waves'][:n_waves])
        kwargs['weights'] = hdul[1].data['weights'][:n_waves]
        kwargs['temp'] = transform_temperature(hdul[1].data['temp'])[0]
        kwargs['dens'] = transform_density(hdul[1].data['dens'])[0]
        kwargs['n_u_range'] = tuple(hdul[1].data['n_u_range'][:2])

    return kwargs
    
def load_from_cache(
    *,
    name: Literal['sh1995'],
    temp: float,
    dens: float,
    n_u_range: tuple[int, int],
    info: Info,
    find_any: bool = True,
) -> dict:
    path = convert_params_to_path(
        name=name, 
        temp=temp, 
        dens=dens, 
        n_u_range=n_u_range, 
        info=info, 
        mode='load' if find_any else 'save',
    )
    return load(path=path, info=info)