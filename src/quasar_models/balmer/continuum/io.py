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

from quasar_typing.numpy import FloatMatrix, SortedFloatVector, FloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_typing.pathlib import AbsoluteFITSPath, AnyAbsoluteFITSPath

from quasar_utils.setup import Info

from ...utils.template.io import drop_nonpos

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parents[1] / ".cache"
PATH_TO_DATA: Path = _this_file.parents[1] / ".data"

class BalmerContinuumTemplateProtocol(Protocol):
    fwhm: SortedFloatVector
    x: SortedFloatVector
    data: FloatMatrix
    info: Info
    is_logspace: bool
    name: str
    path: AbsoluteFITSPath | None
    _alpha_matrix: csr_matrix_ | None
    _beta_matrix: csr_matrix_ | None
    _xn: SortedFloatVector | None

    temp: float
    tau: float
    scale: float

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(str.removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def convert_params_to_path(
    temp: float,
    tau: float,
    scale: float,
    *,
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
    template: BalmerContinuumTemplateProtocol,
    path: str | AbsoluteFITSPath,
) -> AbsoluteFITSPath:
    path = convert_path(path)

    v_unit: str = template.info.units.velocity_unit.to_string()
    x_unit: str = template.info.units.wavelength_unit.to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()
    t_unit: str = template.info.units.temp_unit.to_string()
    
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
    col_temp = fits.Column(
        name='temp',
        format='F',
        unit=t_unit,
        array=[template.temp],
    )
    col_tau = fits.Column(
        name='tau',
        format='F',
        array=[template.tau],
    )
    col_scale = fits.Column(
        name='scale',
        format='F',
        array=[template.scale],
    )
    hdu = fits.BinTableHDU.from_columns(
        [col_fwhm, col_x, col_temp, col_tau, col_scale],
    )
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

def save_to_cache(template: BalmerContinuumTemplateProtocol) -> AbsoluteFITSPath:
    return save(
        template, 
        convert_params_to_path(
            template.temp, template.tau, template.scale, 
            info=template.info,
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
        t_unit = Unit(hdu1.columns[2].unit)

        def transform_velocity(arr: FloatVector) -> FloatVector:
            return info.units.getC(arr * v_unit)

        def transform_wavelength(arr: FloatVector) -> FloatVector:
            return info.units.getWavelength(arr * x_unit)
        
        def transform_flux(arr: FloatMatrix) -> FloatMatrix:
            return info.units.getFlux(arr * f_unit)
        
        def transform_temperature(arr: FloatVector) -> FloatVector:
            return info.units.getTemperature(arr * t_unit)
                
        args = (
            drop_nonpos(transform_velocity(hdu1.data['fwhm'])),
            drop_nonpos(transform_wavelength(hdu1.data['x'])),
            transform_flux(hdu0.data),
            transform_temperature(hdu1.data['temp'])[0],
            hdu1.data['tau'][0],
            hdu1.data['scale'][0],
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
    temp: float,
    tau: float,
    scale: float,
    *,
    info: Info,
) -> tuple[tuple, dict]:
    
    return load(
        convert_params_to_path(temp, tau, scale, info=info),
        info=info,
    )