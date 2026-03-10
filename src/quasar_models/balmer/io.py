from astropy.io.fits import HDUList, PrimaryHDU, BinTableHDU, Column
from pathlib import Path
from scipy.sparse import csr_matrix
from quasar_utils.setup import Info

from ..utils.template import BaseTemplate, drop_nonpos

def _save(template: BaseTemplate, path: Path) -> None:
    v_unit: str = template.info.units['velocity_unit'].to_string()
    x_unit: str = template.info.units['wavelength_unit'].to_string()
    f_unit: str = template.info.units.getFluxUnit().to_string()
    t_unit: str = template.info.units['temp_unit'].to_string()
    d_unit: str = template.info.units['dens_unit'].to_string()

    def _velocity(val):
        return template.info.units.getC(val).to(v_unit).value

    hdul: HDUList = HDUList()

    hdul.append(PrimaryHDU(
        data = template.info.units.getFlux(template.data).to(f_unit).value,
    ))

    hdr = hdul[0].header

    hdr['NAME'] = template.name
    hdr['CTYPE1'] = ('fwhm', 'fwhm axis')
    hdr['CTYPE2'] = ('x', 'spectral axis')
    hdr['LOGSPACE'] = 'y' if template.is_logspace else 'n'
    hdr['AS_TEMP'] = (
        'y' if template.based_on_template else 'n',
        'based on template',
    )

    if not template.based_on_template:
        hdr['EDGE']  = (template.edge,  'edge')
        hdr['TEMP']  = (template.temp,  'temperature')
        hdr['DENS']  = (template.dens,  'density')
        hdr['TAU']   = (template.tau,   'tau')
        hdr['SCALE'] = (template.scale, 'scale')
        hdr['RATIO'] = (template.ratio, 'ratio')
        hdr['CASE']  = (template.case,  'rec. case')
        hdr['N_L']   = (template.n_l,   'n_l')
        hdr['N_U']   = (template.n_u,   'n_u')

    hdr['FUNIT'] = (f_unit, 'flux unit')
    hdr['TUNIT'] = (t_unit, 'temperature unit')
    hdr['DUNIT'] = (d_unit, 'density unit')
    
    if template.path is not None: 
        hdul[0].header['PATH'] = str(template.path)

    cols: list[Column] = []

    cols.append(Column(
        name = 'fwhm',
        format = 'F',
        unit = v_unit,
        array = _velocity(template.fwhm),
    ))
    cols.append(Column(
        name = 'x',
        format = 'F',
        unit = x_unit,
        array = template.x,
    ))

    if not template.based_on_template:
        cols.append(Column(
            name = 'waves',
            format = 'F',
            unit = x_unit,
            array = template.waves,
        ))
        cols.append(Column(
            name = 'weights',
            format = 'F',
            array = template.weights,
        ))

    hdul.append(BinTableHDU.from_columns(cols))

    if getattr(template, '_alpha_matrix', None) is not None:
        col_xn = Column(
            name='xn',
            format='F',
            unit = x_unit,
            array = template._xn,
        )
        col_alpha_data = Column(
            name='alpha_data',
            format='D',
            array=template._alpha_matrix.data.flatten(),
        )
        col_alpha_indices = Column(
            name='alpha_indices',
            format='K',
            array=template._alpha_matrix.indices,
        )
        col_alpha_indptr = Column(
            name='alpha_indptr',
            format='K',
            array=template._alpha_matrix.indptr,
        )
        col_beta_data = Column(
            name='beta_data',
            format='D',
            array=template._beta_matrix.data.flatten(),
        )
        col_beta_indices = Column(
            name='beta_indices',
            format='K',
            array=template._beta_matrix.indices,
        )
        col_beta_indptr = Column(
            name='beta_indptr',
            format='K',
            array=template._beta_matrix.indptr,
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

        v_unit: Unit = Unit(hdul[1].columns[0].unit)
        x_unit: Unit = Unit(hdul[1].columns[1].unit)
        f_unit: Unit = Unit(hdul[0].header['FUNIT'])
        t_unit: Unit = Unit(hdul[0].header['TUNIT'])
        d_unit: Unit = Unit(hdul[0].header['DUNIT'])

        def _velocity(val):
            return info.units.getC(val * v_unit)
        def _wavelength(val):
            return info.units.getWavelength(val * x_unit)
        def _flux(val):
            return info.units.getFlux(val * f_unit)
        def _temperature(val):
            return info.units.getTemperature(val * t_unit)
        def _density(val):
            return info.units.getDensity(val * d_unit)

        table = hdul[1].data
        
        args: tuple = (
            _velocity(drop_nonpos(table['fwhm'])),
            _wavelength(drop_nonpos(table['x'])),
            _flux(hdul[0].data),
        )
        kwargs: dict = {}

        kwargs['name'] = hdul[0].header['NAME']
        kwargs['is_logspace'] = (hdul[0].header['LOGSPACE'] == 'y')
        kwargs['based_on_template'] = (hdul[0].header['AS_TEMP'] == 'y')
        kwargs['path'] = path

        if not kwargs['based_on_template']:
            kwargs['edge']  = _wavelength(hdul[0].header['EDGE'])
            kwargs['temp']  = _temperature(hdul[0].header['TEMP'])
            kwargs['dens']  = _density(hdul[0].header['DENS'])
            kwargs['tau']   = hdul[0].header['TAU']
            kwargs['scale'] = hdul[0].header['SCALE']
            kwargs['ratio'] = hdul[0].header['RATIO']
            kwargs['case']  = hdul[0].header['CASE']
            kwargs['n_l']   = hdul[0].header['N_L']
            kwargs['n_u']   = hdul[0].header['N_U']

            kwargs['waves'] = _wavelength(table['waves'])
            kwargs['weights'] = table['weights']

        template: BalmerTemplate = BalmerTemplate(
            *args,
            **kwargs,
            info = info,
        )

        if len(hdul) > 2:
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