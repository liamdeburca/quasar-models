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

from quasar_typing.numpy import FloatMatrix, SortedFloatVector, FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_typing.pathlib import AbsoluteFITSPath, AnyAbsoluteFITSPath

from quasar_utils.setup import Info

from ...utils.template.io import drop_nonpos

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parents[1] / ".cache"
PATH_TO_DATA: Path = _this_file.parents[1] / ".data"

class BalmerSeriesTemplateProtocol(Protocol):
    fwhm: SortedFloatVector
    x: SortedFloatVector
    data: FloatMatrix
    info: Info
    is_logspace: bool
    name: Literal['sh1995']
    path: AbsoluteFITSPath | None
    _alpha_matrix: csr_matrix_ | None
    _beta_matrix: csr_matrix_ | None
    _xn: SortedFloatVector | None

    waves: SortedFloatVector
    weights: FloatVector
    temp: float
    dens: float
    n_u_range: tuple[int, int]

    fwhm_norm: float
    normalisation: float

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(str.removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def convert_params_to_path(
    name: Literal['sh1995'],
    temp: float,
    dens: float,
    n_u_range: tuple[int, int],
    *,
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
    template: BalmerSeriesTemplateProtocol,
    path: str | AbsoluteFITSPath,
) -> AbsoluteFITSPath:
    path = convert_path(path)

    v_unit: str = template.info.units.velocity_unit.to_string()
    x_unit: str = template.info.units.wavelength_unit.to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()
    t_unit: str = template.info.units.temp_unit.to_string()
    d_unit: str = template.info.units.dens_unit.to_string()
    
    hdul = fits.HDUList()

    hdu = fits.PrimaryHDU(data=template.data)
    
    hdr = hdu.header
    hdr['NAME'] = template.name
    hdr['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdr['CTYPE2'] = ('x', 'spectral axis')
    hdr['BUNIT'] = (f_unit, 'flux unit')
    hdr['LOGSPACE'] = 'y' if template.is_logspace else 'n'

    hdul.append(hdu)

    col_fwhm = fits.Column(
        name='fwhm',
        format='F',
        unit=v_unit,
        array=template.info.units.getC(template.fwhm).to(v_unit).value,
    )
    col_x = fits.Column(
        name='x',
        format='F',
        unit=x_unit,
        array=template.x,
    )
    col_waves = fits.Column(
        name='waves',
        format='F',
        unit=x_unit,
        array=template.waves,
    )
    col_weights = fits.Column(
        name='weights',
        format='F',
        array=template.weights,
    )
    col_temp = fits.Column(
        name='temp',
        format='F',
        unit=t_unit,
        array=[template.temp],
    )
    col_dens = fits.Column(
        name='dens',
        format='F',
        unit=d_unit,
        array=[template.dens],
    )
    col_n_u = fits.Column(
        name='n_u_range',
        format='I',
        array=list(template.n_u_range),
    )
    hdu = fits.BinTableHDU.from_columns([
        col_fwhm, col_x, 
        col_waves, col_weights, 
        col_temp, col_dens, col_n_u, 
    ])
    hdul.append(hdu)

    if template._alpha_matrix is not None:
        col_xn = fits.Column(
            name = 'xn',
            format = 'F',
            unit = x_unit,
            array = template._xn,
        )
        col_alpha_data = fits.Column(
            name = 'alpha_data',
            format = 'D',
            array = template._alpha_matrix.data,
        )
        col_alpha_indices = fits.Column(
            name = 'alpha_indices',
            format = 'K',
            array = template._alpha_matrix.indices,
        )
        col_alpha_indptr = fits.Column(
            name = 'alpha_indptr',
            format = 'K',
            array = template._alpha_matrix.indptr,
        )
        col_beta_data = fits.Column(
            name = 'beta_data',
            format = 'D',
            array = template._beta_matrix.data,
        )
        col_beta_indices = fits.Column(
            name = 'beta_indices',
            format = 'K',
            array = template._beta_matrix.indices,
        )
        col_beta_indptr = fits.Column(
            name = 'beta_indptr',
            format = 'K',
            array = template._beta_matrix.indptr,
        )
        hdu = fits.BinTableHDU.from_columns([
            col_xn,
            col_alpha_data, col_alpha_indices, col_alpha_indptr,
            col_beta_data, col_beta_indices, col_beta_indptr,
        ])
        
        hdr = hdu.header
        
        # no. of _xn values
        hdr['XN_VAL'] = template._xn.size
        # Alpha-matrix
        hdr['ASHAPE'] = "{}/{}".format(*template._alpha_matrix.shape)
        # no. of alpha-matric values
        hdr['A_VAL'] = template._alpha_matrix.data.size
        # no. of alpha-matrix indices
        hdr['A_IND'] = template._alpha_matrix.indices.size
        # no. of alpha-matrix index pointers
        hdr['A_PTR'] = template._alpha_matrix.indptr.size
        # Beta-matrix
        hdr['BSHAPE'] = "{}/{}".format(*template._beta_matrix.shape)
        # no. of beta-matrix values
        hdr['B_VAL'] = template._beta_matrix.data.size
        # no. of beta-matrix indices
        hdr['B_IND'] = template._beta_matrix.indices.size
        # no. of beta-matrix index pointers
        hdr['B_PTR'] = template._beta_matrix.indptr.size

        hdul.append(hdu)

    hdul.writeto(path, overwrite=True)

    return path

def save_to_cache(template: BalmerSeriesTemplateProtocol) -> AbsoluteFITSPath:
    return save(
        template, 
        convert_params_to_path(
            template.name, template.temp, template.dens, template.n_u_range, 
            info=template.info,
            mode='save',
        ),
    )

def load(
    path: str | AbsoluteFITSPath,
    info: Info,
) -> tuple[tuple, dict]:
    path = convert_path(path)

    with fits.open(path) as hdul:
        hdu0: fits.PrimaryHDU = hdul[0]
        hdu1: fits.BinTableHDU = hdul[1]

        v_unit = Unit(hdu1.columns[0].unit)
        x_unit = Unit(hdu1.columns[1].unit)
        f_unit = Unit(hdu0.header['BUNIT'])
        t_unit = Unit(hdu1.columns[4].unit)
        d_unit = Unit(hdu1.columns[5].unit)

        def transform_velocity(arr: FloatVector) -> FloatVector:
            return info.units.getC(arr * v_unit)

        def transform_wavelength(arr: FloatVector) -> FloatVector:
            return info.units.getWavelength(arr * x_unit)
        
        def transform_flux(arr: FloatMatrix) -> FloatMatrix:
            return info.units.getFlux(arr * f_unit)
        
        def transform_temperature(arr: FloatVector) -> FloatVector:
            return info.units.getTemperature(arr * t_unit)
        
        def transform_density(arr: FloatVector) -> FloatVector:
            return info.units.getDensity(arr * d_unit)
                                
        args = (
            drop_nonpos(transform_velocity(hdu1.data['fwhm'])),
            drop_nonpos(transform_wavelength(hdu1.data['x'])),
            transform_flux(hdu0.data),
            drop_nonpos(transform_wavelength(hdu1.data['waves'])),
            drop_nonpos(hdu1.data['weights']),
            transform_temperature(hdu1.data['temp'])[0],
            transform_density(hdu1.data['dens'])[0],
            tuple(hdu1.data['n_u_range'][:2]),
        )
        kwargs = {
            'name': hdu0.header['NAME'],
            'is_logspace': hdu0.header['LOGSPACE'] == 'y',
            'path': path,
            'info': info,
        }

        if len(hdul) > 2:
            hdu2: fits.BinTableHDU = hdul[2]

            kwargs['_xn'] = transform_wavelength(hdu2.data['xn'])
            kwargs['_alpha_matrix'] = csr_matrix_(
                (
                    hdu2.data['alpha_data'], 
                    hdu2.data['alpha_indices'], 
                    hdu2.data['alpha_indptr'],
                ),
                shape=tuple(map(int, hdu2.header['ASHAPE'].strip().split('/'))),
            )
            kwargs['_beta_matrix'] = csr_matrix_(
                (
                    hdu2.data['beta_data'], 
                    hdu2.data['beta_indices'], 
                    hdu2.data['beta_indptr'],
                ), 
                shape=tuple(map(int, hdu2.header['BSHAPE'].strip().split('/'))),
            )

    return args, kwargs
    
def load_from_cache(
    name: Literal['sh1995'],
    temp: float,
    dens: float,
    n_u_range: tuple[int, int],
    *,
    info: Info,
    find_any: bool = True,
) -> tuple[tuple, dict]:
    path = convert_params_to_path(
        name, temp, dens, n_u_range, 
        info=info, 
        mode='load' if find_any else 'save',
    )
    return load(path, info)