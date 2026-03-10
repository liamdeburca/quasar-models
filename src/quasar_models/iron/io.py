from pathlib import Path
from astropy.io.fits import PrimaryHDU, HDUList, Column, BinTableHDU

from quasar_utils.setup import Info
from ..utils.template import BaseTemplate, drop_nonpos

def _save(temp: BaseTemplate, path: Path) -> None:
    """
    Saves the IronTemplate to a FITS file.
    """
    v_unit: str = temp.info.units['velocity_unit'].to_string()
    x_unit: str = temp.info.units['wavelength_unit'].to_string()
    f_unit: str = temp.info.units.getFluxUnit().to_string()

    def _velocity(val):
        return temp.info.units.getC(val).to(v_unit).value

    hdul: HDUList = HDUList()

    hdu: PrimaryHDU = PrimaryHDU(data=temp.data)
    hdu.header['NAME'] = temp.name
    hdu.header['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdu.header['CTYPE2'] = ('x', 'spectral axis')
    hdu.header['BUNIT'] = (f_unit, 'flux unit')
    hdu.header['LOGSPACE'] = 'y' if temp.is_logspace else 'n'

    if temp.path is not None: 
        hdu.header['PATH'] = str(temp.path)

    hdul.append(hdu)

    col_fwhm: Column = Column(
        name = 'fwhm',
        format = 'F',
        unit = v_unit,
        array = _velocity(temp.fwhm),
    )
    col_x: Column = Column(
        name = 'x',
        format = 'F',
        unit = x_unit,
        array = temp.x,
    )
    hdul.append(BinTableHDU.from_columns([col_fwhm, col_x]))

    if getattr(temp, '_alpha_matrix', None) is not None:
        col_xn = Column(
            name='xn',
            format='F',
            unit = x_unit,
            array = temp._xn,
        )
        col_alpha_data = Column(
            name='alpha_data',
            format='D',
            array=temp._alpha_matrix.data.flatten(),
        )
        col_alpha_indices = Column(
            name='alpha_indices',
            format='K',
            array=temp._alpha_matrix.indices,
        )
        col_alpha_indptr = Column(
            name='alpha_indptr',
            format='K',
            array=temp._alpha_matrix.indptr,
        )
        col_beta_data = Column(
            name='beta_data',
            format='D',
            array=temp._beta_matrix.data.flatten(),
        )
        col_beta_indices = Column(
            name='beta_indices',
            format='K',
            array=temp._beta_matrix.indices,
        )
        col_beta_indptr = Column(
            name='beta_indptr',
            format='K',
            array=temp._beta_matrix.indptr,
        )
        hdu: BinTableHDU = BinTableHDU.from_columns([
            col_xn,
            col_alpha_data, col_alpha_indices, col_alpha_indptr,
            col_beta_data, col_beta_indices, col_beta_indptr,
        ])

        hdr = hdu.header
        
        # no. of _xn values
        hdr['XN_VAL'] = temp._xn.size
        # Alpha-matrix
        hdr['ASHAPE'] = "{}/{}".format(*temp._alpha_matrix.shape)
        # no. of alpha-matric values
        hdr['A_VAL'] = temp._alpha_matrix.data.size
        # no. of alpha-matrix indices
        hdr['A_IND'] = temp._alpha_matrix.indices.size
        # no. of alpha-matrix index pointers
        hdr['A_PTR'] = temp._alpha_matrix.indptr.size
        # Beta-matrix
        hdr['BSHAPE'] = "{}/{}".format(*temp._beta_matrix.shape)
        # no. of beta-matrix values
        hdr['B_VAL'] = temp._beta_matrix.data.size
        # no. of beta-matrix indices
        hdr['B_IND'] = temp._beta_matrix.indices.size
        # no. of beta-matrix index pointers
        hdr['B_PTR'] = temp._beta_matrix.indptr.size

        hdul.append(hdu)

    hdul.writeto(path, overwrite=True)

def _load(path: Path, info: Info):
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

        table = hdul[1].data
        template: IronTemplate = IronTemplate(
            _velocity(drop_nonpos(table['fwhm'])),
            _wavelength(drop_nonpos(table['x'])),
            _flux(hdul[0].data),
            info  = info,
            name  = str(hdul[0].header['NAME']),
            path  = path,
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