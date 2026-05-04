from pathlib import Path
from astropy.io.fits import PrimaryHDU, HDUList, Column, BinTableHDU

from quasar_utils.setup import Info
from ..utils.template import BaseTemplate, drop_nonpos

def _save(template: BaseTemplate, path: Path) -> HDUList:
    """
    Saves the given IronTemplate instance to a FITS file at the specified path.

    The FITS file will consists of the following HDUs:
    - [0]: Primary HDU containing the template data and metadata in the header.
    - [1]: Binary table HDU containing the fwhm and x arrays.
    - [2]: (Optional) Binary table HDU containing the sparse matrices if they 
           exist in the template.

    Parameters
    ----------
    template : BaseTemplate
        The IronTemplate instance to be saved.
    path : Path
        The file path where the FITS file will be saved.

    Returns
    -------
    HDUList
        The HDUList object representing the FITS file that was saved.

    Notes
    -----
    Lorem ipsum.
    """
    v_unit: str = template.info.units['velocity_unit'].to_string()
    x_unit: str = template.info.units['wavelength_unit'].to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()

    # [0] Primary HDU -  data and metadata

    hdul: HDUList = HDUList()

    hdu: PrimaryHDU = PrimaryHDU(data=template.data)
    hdu.header['NAME'] = template.name
    hdu.header['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdu.header['CTYPE2'] = ('x', 'spectral axis')
    hdu.header['BUNIT'] = (f_unit, 'flux unit')
    hdu.header['LOGSPACE'] = 'y' if template.is_logspace else 'n'

    if template.path is not None: 
        hdu.header['PATH'] = str(template.path)

    hdul.append(hdu)

    # [1] Binary table HDU - fwhm and x arrays

    col_fwhm: Column = Column(
        name = 'fwhm',
        format = 'F',
        unit = v_unit,
        array = template.info.units.getC(template.fwhm).to(v_unit).value,
    )
    col_x: Column = Column(
        name = 'x',
        format = 'F',
        unit = x_unit,
        array = template.x,
    )
    hdul.append(BinTableHDU.from_columns([col_fwhm, col_x]))

    # [2] Binary table HDU - sparse matrices (if they exist)

    if getattr(template, '_alpha_matrix', None) is not None:
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

    return hdul

def _load(path: Path, info: Info) -> BaseTemplate:
    """
    Loads the specified IronTemplate from a FITS file.
    """
    from astropy.io.fits import open as fits_open
    from astropy.units import Unit

    # Watch out for circular imports here
    from .iron_template import IronTemplate

    with fits_open(path) as hdul:
        v_unit: Unit = Unit(hdul[1].columns[0].unit)
        x_unit: Unit = Unit(hdul[1].columns[1].unit)
        f_unit: Unit = Unit(hdul[0].header['BUNIT'])

        def _velocity(val):
            return info.units.getC(val * v_unit)
        def _wavelength(val):
            return info.units.getWavelength(val * x_unit)
        def _flux(val):
            return info.units.getFlux(val * f_unit)
        
        if 'PATH' in hdul[0].header: 
            _path = Path(hdul[0].header['PATH'])
        else:                        
            _path = path

        table = hdul[1].data
        template: IronTemplate = IronTemplate(
            _velocity(drop_nonpos(table['fwhm'])),
            _wavelength(drop_nonpos(table['x'])),
            _flux(hdul[0].data),
            info = info,
            name = str(hdul[0].header['NAME']),
            path = _path,
            is_logspace = (hdul[0].header['LOGSPACE'].lower() == 'y'),
        )

        if len(hdul) > 2:
            from scipy.sparse import csr_matrix

            hdr = hdul[2].header
            data = hdul[2].data

            template._xn = _wavelength(data['xn'][:int(hdr['XN_VAL'])])

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