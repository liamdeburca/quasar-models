__all__ = [
    "VProfileCopy1G",
    "VProfileCopy2G",
    "VProfileCopy3G",
    "VProfileCopy4G",
    "VProfileCopy5G",
    "_VProfileCopy",
]

from typing import ClassVar, Literal

from astropy.modeling import Parameter

from ._vprofilecopy import _VProfileCopy


class VProfileCopy1G(_VProfileCopy):
    n_profiles: ClassVar[Literal[1]] = 1

    strength_scale = Parameter(default=1, bounds=(0, None))

    strength_1 = Parameter(default=1, fixed=True)
    fwhm_v_1 = Parameter(default=1e-3, fixed=True)
    v_off_1 = Parameter(default=0, fixed=True)


class VProfileCopy2G(_VProfileCopy):
    n_profiles: ClassVar[Literal[2]] = 2

    strength_scale = Parameter(default=1, bounds=(0, None))

    strength_1 = Parameter(default=1, fixed=True)
    fwhm_v_1 = Parameter(default=1e-3, fixed=True)
    v_off_1 = Parameter(default=0, fixed=True)

    strength_2 = Parameter(default=1, fixed=True)
    fwhm_v_2 = Parameter(default=1e-3, fixed=True)
    v_off_2 = Parameter(default=0, fixed=True)


class VProfileCopy3G(_VProfileCopy):
    n_profiles: ClassVar[Literal[3]] = 3

    strength_scale = Parameter(default=1, bounds=(0, None))

    strength_1 = Parameter(default=1, fixed=True)
    fwhm_v_1 = Parameter(default=1e-3, fixed=True)
    v_off_1 = Parameter(default=0, fixed=True)

    strength_2 = Parameter(default=1, fixed=True)
    fwhm_v_2 = Parameter(default=1e-3, fixed=True)
    v_off_2 = Parameter(default=0, fixed=True)

    strength_3 = Parameter(default=1, fixed=True)
    fwhm_v_3 = Parameter(default=1e-3, fixed=True)
    v_off_3 = Parameter(default=0, fixed=True)


class VProfileCopy4G(_VProfileCopy):
    n_profiles: ClassVar[Literal[4]] = 4

    strength_scale = Parameter(default=1, bounds=(0, None))

    strength_1 = Parameter(default=1, fixed=True)
    fwhm_v_1 = Parameter(default=1e-3, fixed=True)
    v_off_1 = Parameter(default=0, fixed=True)

    strength_2 = Parameter(default=1, fixed=True)
    fwhm_v_2 = Parameter(default=1e-3, fixed=True)
    v_off_2 = Parameter(default=0, fixed=True)

    strength_3 = Parameter(default=1, fixed=True)
    fwhm_v_3 = Parameter(default=1e-3, fixed=True)
    v_off_3 = Parameter(default=0, fixed=True)

    strength_4 = Parameter(default=1, fixed=True)
    fwhm_v_4 = Parameter(default=1e-3, fixed=True)
    v_off_4 = Parameter(default=0, fixed=True)


class VProfileCopy5G(_VProfileCopy):
    n_profiles: ClassVar[Literal[5]] = 5

    strength_scale = Parameter(default=1, bounds=(0, None))

    strength_1 = Parameter(default=1, fixed=True)
    fwhm_v_1 = Parameter(default=1e-3, fixed=True)
    v_off_1 = Parameter(default=0, fixed=True)

    strength_2 = Parameter(default=1, fixed=True)
    fwhm_v_2 = Parameter(default=1e-3, fixed=True)
    v_off_2 = Parameter(default=0, fixed=True)

    strength_3 = Parameter(default=1, fixed=True)
    fwhm_v_3 = Parameter(default=1e-3, fixed=True)
    v_off_3 = Parameter(default=0, fixed=True)

    strength_4 = Parameter(default=1, fixed=True)
    fwhm_v_4 = Parameter(default=1e-3, fixed=True)
    v_off_4 = Parameter(default=0, fixed=True)

    strength_5 = Parameter(default=1, fixed=True)
    fwhm_v_5 = Parameter(default=1e-3, fixed=True)
    v_off_5 = Parameter(default=0, fixed=True)
