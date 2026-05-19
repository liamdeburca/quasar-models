"""
This script contains utilities for setting up various host galaxy templates. 

It currently supports data from the following:
- Bruzual & Charlot (2003), a.k.a. 'bc2003'.
"""

__all__ = ['BC2003']

from pathlib import Path
from numpy import array, float64, isclose
from astropy.io import fits
from astropy.units import Unit

from quasar_utils.setup import Info
from quasar_typing.numpy import FloatVector

from quasar_models.host.io import PATH_TO_DATA, PATH_TO_CACHE
from quasar_models.host.host_galaxy_template import HostGalaxyTemplate

class BC2003:
    name: str = 'bc2003'
    paths: list[Path] = sorted(PATH_TO_DATA.glob("tau06_z02_*_001.fits"))

    x_lb: float = 3200.0
    x_ub: float = 9500.0

    def __init__(self):
        self.info: Info = Info()
        self.fwhm: FloatVector = array([0], dtype=float64)
        self.__post_init__()

    def __post_init__(self):
        PATH_TO_CACHE.mkdir(exist_ok=True)

    @classmethod
    def get_age_from_path(cls, path: Path) -> int:
        return 1000 * int(
            path.name \
                .removeprefix('tau06_z02_') \
                .removesuffix('_001.fits')
        )

    def main(self) -> None:
        for path in self.paths:
            age = self.get_age_from_path(path)

            with fits.open(path) as hdul:
                x = hdul[1].data['WAVELENGTH'].astype(float64)
                y = hdul[1].data['FLUX'].astype(float64)
                mask = (self.x_lb <= x) & (x <= self.x_ub)

                x = x[mask]
                y = y[mask]

                template = HostGalaxyTemplate(
                    self.fwhm, 
                    self.info.units.getWavelength(x * Unit('angstrom')), 
                    y[None,:],
                    info=self.info,
                    is_logspace=False,
                    name=self.name,
                    path=None,
                    age=age,
                )
                template.save_to_cache()
                del template

def main() -> None:
    BC2003().main()

if __name__ == '__main__':
    main()