from numpy import float64
from numpy.typing import NDArray
from astropy.io.fits import HDUList, PrimaryHDU, BinTableHDU, Column
from pathlib import Path
from scipy.sparse import csr_matrix
from quasar_utils.setup import Info

from ..utils.template import BaseTemplate, drop_nonpos

def _save(template: BaseTemplate, path: Path) -> None:
    """
    Saves the given BalmerTemplate instance to a FITS file at the specified
    path.

    The FITS file with consist of the following HDUs:
    - [0]: Primary HDU containing the template data and metadata in the header.
    - [1]: Binary table HDU containing the fwhm and x arrays, and optionally the 
           waves and weights arrays if the template is not based on another 
           template.
    - [2]: (Optional) Binary table HDU containing the sparse matrices if they 
           exist in the template.
    """
    v_unit: str = template.info.units['velocity_unit'].to_string()
    x_unit: str = template.info.units['wavelength_unit'].to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()
    t_unit: str = template.info.units['temp_unit'].to_string()
    d_unit: str = template.info.units['dens_unit'].to_string()

    def check(attr: str) -> bool:
        return getattr(template, attr, attr) is not None
    
    hdul: HDUList = HDUList()

    # [0] Primary HDU -  data and metadata

    hdu: PrimaryHDU = PrimaryHDU(data=template.data)
    hdr = hdu.header

    hdr['NAME'] = template.name
    hdr['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdr['CTYPE2'] = ('x', 'spectral axis')
    hdr['BUNIT'] = (f_unit, 'flux unit')
    hdr['LOGSPACE'] = 'y' if template.is_logspace else 'n'
    hdr['AS_TEMP'] = (
        'y' if template.based_on_template else 'n',
        'based on template',
    )

    hdr['TUNIT'] = (t_unit, 'temperature unit')
    hdr['DUNIT'] = (d_unit, 'density unit')

    if check('edge'): hdr['EDGE'] = (template.edge, 'edge')
    if check('temp'): hdr['TEMP'] = (template.temp, 'temperature')
    if check('dens'): hdr['DENS'] = (template.dens, 'density')
    if check('tau'):  hdr['TAU'] = (template.tau, 'tau')
    if check('scale'): hdr['SCALE'] = (template.scale, 'scale')
    if check('ratio'): hdr['RATIO'] = (template.ratio, 'ratio')
    if check('case'): hdr['CASE'] = (template.case, 'rec. case')
    if check('n_l'): hdr['N_L'] = (template.n_l, 'n_l')
    if check('n_u'): hdr['N_U'] = (template.n_u, 'n_u')

    if check('path'): hdr['PATH'] = str(template.path)

    hdul.append(hdu)

    # [1] Binary table HDU - fwhm and x arrays, and optionally waves and weights arrays

    col_fwhm: Column = Column(
        name='fwhm',
        format='F',
        unit=v_unit,
        array=template.info.units.getC(template.fwhm).to(v_unit).value,
    )
    col_x: Column = Column(
        name='x',
        format='F',
        unit=x_unit,
        array=template.x,
    )
    cols: list[Column] = [col_fwhm, col_x]

    if check('waves'):
        col_waves: Column = Column(
            name='waves',
            format='F',
            unit=x_unit,
            array=template.waves
        )
        cols.append(col_waves)

    if check('weights'):
        col_weights: Column = Column(
            name='weights',
            format='F',
            array=template.weights,
        )
        cols.append(col_weights)

    hdu = BinTableHDU.from_columns(cols)
    hdul.append(hdu)

    # [2] Binary table HDU - sparse matrices (if they exist)

    if check('_alpha_matrix'):
        col_xn = Column(
            name = 'xn',
            format = 'F',
            unit = x_unit,
            array = template._xn,
        )
        col_alpha_data = Column(
            name = 'alpha_data',
            format = 'D',
            array = template._alpha_matrix.data,
        )
        col_alpha_indices = Column(
            name = 'alpha_indices',
            format = 'K',
            array = template._alpha_matrix.indices,
        )
        col_alpha_indptr = Column(
            name = 'alpha_indptr',
            format = 'K',
            array = template._alpha_matrix.indptr,
        )
        col_beta_data = Column(
            name = 'beta_data',
            format = 'D',
            array = template._beta_matrix.data,
        )
        col_beta_indices = Column(
            name = 'beta_indices',
            format = 'K',
            array = template._beta_matrix.indices,
        )
        col_beta_indptr = Column(
            name = 'beta_indptr',
            format = 'K',
            array = template._beta_matrix.indptr,
        )
        hdu: BinTableHDU = BinTableHDU.from_columns([
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

def _load(path: Path, info: Info):
    """
    Loads the specified BalmerTemplate from a FITS file.
    """
    from astropy.io.fits import open as fits_open
    from astropy.units import Unit

    # Watch out for circular imports
    from .balmer_template import BalmerTemplate

    with fits_open(path) as hdul:
        table = hdul[1]

        v_unit: Unit = Unit(table.columns[0].unit)
        x_unit: Unit = Unit(table.columns[1].unit)
        f_unit: Unit = Unit(hdul[0].header['BUNIT'])
        t_unit: Unit = Unit(hdul[0].header['TUNIT'])
        d_unit: Unit = Unit(hdul[0].header['DUNIT'])

        def transform_wavelength(arr: NDArray[float64]) -> NDArray[float64]:
            return info.units.getWavelength(arr * x_unit)
        def transform_velocity(arr: NDArray[float64]) -> NDArray[float64]:
            return info.units.getC(arr * v_unit)
        def transform_flux(arr: NDArray[float64]) -> NDArray[float64]:
            return info.units.getFlux(arr * f_unit)
        def transform_temperature(arr: NDArray[float64]) -> NDArray[float64]:
            return info.units.getTemperature(arr * t_unit)
        def transform_density(arr: NDArray[float64]) -> NDArray[float64]:
            return info.units.getDensity(arr * d_unit)

        args: tuple = (
            transform_velocity(drop_nonpos(table.data['fwhm'])),
            transform_wavelength(drop_nonpos(table.data['x'])),
            transform_flux(hdul[0].data),
        )
        kwargs: dict = {
            'name': hdul[0].header['NAME'],
            'is_logspace': hdul[0].header['LOGSPACE'] == 'y',
            'based_on_template': hdul[0].header['AS_TEMP'] == 'y',
            'path': path,
            'info': info,
        }

        def check(key: str) -> bool:
            return key in hdul[0].header.keys()

        if check('edge'): 
            kwargs['edge'] = transform_wavelength(hdul[0].header['EDGE'])
        if check('temp'): 
            kwargs['temp'] = transform_temperature(hdul[0].header['TEMP'])
        if check('dens'): 
            kwargs['dens'] = transform_density(hdul[0].header['DENS'])

        if check('tau'):  kwargs['tau'] = hdul[0].header['TAU']
        if check('scale'): kwargs['scale'] = hdul[0].header['SCALE']
        if check('ratio'): kwargs['ratio'] = hdul[0].header['RATIO']
        if check('case'):  kwargs['case'] = hdul[0].header['CASE']
        if check('n_l'): kwargs['n_l'] = hdul[0].header['N_L']
        if check('n_u'): kwargs['n_u'] = hdul[0].header['N_U']

        def check(key: str) -> bool:
            # Check if the column name exists in the BinTableHDU
            return key in hdul[1].columns.names

        if check('waves'): kwargs['waves'] = \
            transform_wavelength(drop_nonpos(table.data['waves']))
        if check('weights'): kwargs['weights'] = \
            drop_nonpos(table.data['weights'])

        template: BalmerTemplate = BalmerTemplate(*args, **kwargs)

        if len(hdul) > 2:
            hdr = hdul[2].header
            data = hdul[2].data

            template._xn = transform_wavelength(data['xn'][:int(hdr['XN_VAL'])])

            template._alpha_matrix = csr_matrix(
                (
                    data['alpha_data']   [:int(hdr['A_VAL'])], 
                    data['alpha_indices'][:int(hdr['A_IND'])], 
                    data['alpha_indptr'] [:int(hdr['A_PTR'])],
                ),
                shape=tuple([int(s) for s in hdr['ASHAPE'].strip().split('/')]),
            )
            template._beta_matrix = csr_matrix(
                (
                    data['beta_data']   [:int(hdr['B_VAL'])], 
                    data['beta_indices'][:int(hdr['B_IND'])], 
                    data['beta_indptr'] [:int(hdr['B_PTR'])],
                ),
                shape=tuple([int(s) for s in hdr['BSHAPE'].strip().split('/')]),
            )

        return template