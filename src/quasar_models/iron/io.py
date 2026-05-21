__all__ = [
    'PATH_TO_CACHE', 'PATH_TO_DATA',
    'convert_path',
    'save', 'load',
    'save_to_cache', 'load_from_cache',
]

from typing import Protocol, Literal
from pathlib import Path
from astropy.io import fits
from astropy.units import Unit

from quasar_typing.numpy import FloatVector, FloatMatrix, SortedFloatVector
from quasar_typing.scipy import csr_matrix_
from quasar_typing.pathlib import AbsoluteFITSPath

from quasar_utils.setup import Info

_this_file: Path = Path(__file__).resolve()
PATH_TO_CACHE: Path = _this_file.parent / ".cache"
PATH_TO_DATA: Path = _this_file.parent / ".data"

class IronTemplateProtocol(Protocol):
    fwhm: SortedFloatVector
    x: SortedFloatVector
    data: FloatMatrix
    info: Info
    is_logspace: bool
    name: Literal['vw2001', 'v2003', 'bw']
    path: AbsoluteFITSPath | None
    _alpha_matrix: csr_matrix_ | None
    _beta_matrix: csr_matrix_ | None
    _xn: SortedFloatVector | None

    x_norm: float
    fwhm_norm: float
    normalisation: float

def convert_path(path: str | AbsoluteFITSPath) -> AbsoluteFITSPath:
    if isinstance(path, str):
        path = Path(path.removesuffix('.fits') + '.fits')
        if '/' not in path.as_posix():
            path = PATH_TO_CACHE / path
    return path

def save(
    template: IronTemplateProtocol, 
    path: str | AbsoluteFITSPath,
) -> AbsoluteFITSPath:
    path = convert_path(path)

    v_unit: str = template.info.units.velocity_unit.to_string()
    x_unit: str = template.info.units.wavelength_unit.to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()

    # [0] Primary HDU -  data and metadata

    hdul: fits.HDUList = fits.HDUList()

    hdu: fits.PrimaryHDU = fits.PrimaryHDU(data=template.data)
    hdu.header['NAME'] = template.name
    hdu.header['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdu.header['CTYPE2'] = ('x', 'spectral axis')
    hdu.header['BUNIT'] = (f_unit, 'flux unit')
    hdu.header['LOGSPACE'] = 'y' if template.is_logspace else 'n'

    if template.path is not None: 
        hdu.header['PATH'] = str(template.path)

    hdul.append(hdu)

    # [1] Binary table HDU - fwhm and x arrays

    col_fwhm: fits.Column = fits.Column(
        name = 'fwhm',
        format = 'F',
        unit = v_unit,
        array = template.info.units.getC(template.fwhm).to(v_unit).value,
    )
    col_x: fits.Column = fits.Column(
        name = 'x',
        format = 'F',
        unit = x_unit,
        array = template.x,
    )
    hdul.append(fits.BinTableHDU.from_columns([col_fwhm, col_x]))

    # [2] Binary table HDU - sparse matrices (if they exist)

    if getattr(template, '_alpha_matrix', None) is not None:
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
        hdu: fits.BinTableHDU = fits.BinTableHDU.from_columns([
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

def save_to_cache(template: IronTemplateProtocol) -> AbsoluteFITSPath:
    return save(
        template,
        path=template.name,
    )

def load(
    path: str | AbsoluteFITSPath, 
    info: Info,
) -> tuple[tuple, dict]:
    path = convert_path(path)

    with fits.open(path) as hdul:
        v_unit = Unit(hdul[1].columns[0].unit)
        x_unit = Unit(hdul[1].columns[1].unit)
        f_unit = Unit(hdul[0].header['BUNIT'])

        def transform_velocity(arr: FloatVector) -> FloatVector:
            return info.units.getC(arr * v_unit)

        def transform_wavelength(arr: FloatVector) -> FloatVector:
            return info.units.getWavelength(arr * x_unit)
        
        def transform_flux(arr: FloatMatrix) -> FloatMatrix:
            return info.units.getFlux(arr * f_unit)

        data = transform_flux(hdul[0].data)
        fwhm = transform_velocity(hdul[1].data['fwhm'])[:data.shape[0]]
        x = transform_wavelength(hdul[1].data['x'])[:data.shape[1]]
        
        args = (fwhm, x, data)
        kwargs = {
            'info': info,
            'name': path.stem,
            'path': path,
            'is_logspace': (hdul[0].header['LOGSPACE'].lower() == 'y'),
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

def load_from_cache(name: Literal['vw2001', 'v2003', 'bw'], *, info: Info) -> tuple[tuple, dict]:
    return load(
        convert_path(name), 
        info,
    )

if __name__ == "__main__":
    PATH_TO_CACHE.mkdir(exist_ok=True)