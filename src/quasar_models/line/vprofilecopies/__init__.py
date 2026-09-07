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

    strength_scale = Parameter(
        description="Integrated flux scaling factor",
        default=1, 
        bounds=(0, None),
    )
    strength_1 = Parameter(
        description="Integrated flux of the first Gaussian component",
        default=1,
    )
    fwhm_v_1 = Parameter(
        description="Intrinsic FWHM (km/s) of the first Gaussian component",
        default=1000,
    )
    v_off_1 = Parameter(
        description="Velocity offset (km/s) of the first Gaussian component",
        default=0,
    )


class VProfileCopy2G(_VProfileCopy):
    n_profiles: ClassVar[Literal[2]] = 2

    strength_scale = Parameter(
        description="Integrated flux scaling factor",
        default=1,
        bounds=(0, None),
    )
    strength_1 = Parameter(
        description="Integrated flux of the first Gaussian component",
        default=1,
    )
    fwhm_v_1 = Parameter(
        description="Intrinsic FWHM (km/s) of the first Gaussian component",
        default=1000,
    )
    v_off_1 = Parameter(
        description="Velocity offset (km/s) of the first Gaussian component",
        default=0,
    )
    strength_2 = Parameter(
        description="Integrated flux of the second Gaussian component",
        default=1,
    )
    fwhm_v_2 = Parameter(
        description="Intrinsic FWHM (km/s) of the second Gaussian component",
        default=1000,
    )
    v_off_2 = Parameter(
        description="Velocity offset (km/s) of the second Gaussian component",
        default=0,
    )


class VProfileCopy3G(_VProfileCopy):
    n_profiles: ClassVar[Literal[3]] = 3

    strength_scale = Parameter(
        description="Integrated flux scaling factor",
        default=1,
        bounds=(0, None),
    )
    strength_1 = Parameter(
        description="Integrated flux of the first Gaussian component",
        default=1,
    )
    fwhm_v_1 = Parameter(
        description="Intrinsic FWHM (km/s) of the first Gaussian component",
        default=1000,
    )
    v_off_1 = Parameter(
        description="Velocity offset (km/s) of the first Gaussian component",
        default=0,
    )
    strength_2 = Parameter(
        description="Integrated flux of the second Gaussian component",
        default=1,
    )
    fwhm_v_2 = Parameter(
        description="Intrinsic FWHM (km/s) of the second Gaussian component",
        default=1000,
    )
    v_off_2 = Parameter(
        description="Velocity offset (km/s) of the second Gaussian component",
        default=0,
    )
    strength_3 = Parameter(
        description="Integrated flux of the third Gaussian component",
        default=1,
    )
    fwhm_v_3 = Parameter(
        description="Intrinsic FWHM (km/s) of the third Gaussian component",
        default=1000,
    )
    v_off_3 = Parameter(
        description="Velocity offset (km/s) of the third Gaussian component",
        default=0,
    )


class VProfileCopy4G(_VProfileCopy):
    n_profiles: ClassVar[Literal[4]] = 4

    strength_scale = Parameter(
        description="Integrated flux scaling factor",
        default=1,
        bounds=(0, None),
    )
    strength_1 = Parameter(
        description="Integrated flux of the first Gaussian component",
        default=1,
    )
    fwhm_v_1 = Parameter(
        description="Intrinsic FWHM (km/s) of the first Gaussian component",
        default=1000,
    )
    v_off_1 = Parameter(
        description="Velocity offset (km/s) of the first Gaussian component",
        default=0,
    )
    strength_2 = Parameter(
        description="Integrated flux of the second Gaussian component",
        default=1,
    )
    fwhm_v_2 = Parameter(
        description="Intrinsic FWHM (km/s) of the second Gaussian component",
        default=1000,
    )
    v_off_2 = Parameter(
        description="Velocity offset (km/s) of the second Gaussian component",
        default=0,
    )
    strength_3 = Parameter(
        description="Integrated flux of the third Gaussian component",
        default=1,
    )
    fwhm_v_3 = Parameter(
        description="Intrinsic FWHM (km/s) of the third Gaussian component",
        default=1000,
    )
    v_off_3 = Parameter(
        description="Velocity offset (km/s) of the third Gaussian component",
        default=0,
    )
    strength_4 = Parameter(
        description="Integrated flux of the fourth Gaussian component",
        default=1,
    )
    fwhm_v_4 = Parameter(
        description="Intrinsic FWHM (km/s) of the fourth Gaussian component",
        default=1000,
    )
    v_off_4 = Parameter(
        description="Velocity offset (km/s) of the fourth Gaussian component",
        default=0,
    )


class VProfileCopy5G(_VProfileCopy):
    n_profiles: ClassVar[Literal[5]] = 5

    strength_scale = Parameter(
        description="Integrated flux scaling factor",
        default=1,
        bounds=(0, None),
    )
    strength_1 = Parameter(
        description="Integrated flux of the first Gaussian component",
        default=1,
    )
    fwhm_v_1 = Parameter(
        description="Intrinsic FWHM (km/s) of the first Gaussian component",
        default=1000,
    )
    v_off_1 = Parameter(
        description="Velocity offset (km/s) of the first Gaussian component",
        default=0,
    )
    strength_2 = Parameter(
        description="Integrated flux of the second Gaussian component",
        default=1,
    )
    fwhm_v_2 = Parameter(
        description="Intrinsic FWHM (km/s) of the second Gaussian component",
        default=1000,
    )
    v_off_2 = Parameter(
        description="Velocity offset (km/s) of the second Gaussian component",
        default=0,
    )
    strength_3 = Parameter(
        description="Integrated flux of the third Gaussian component",
        default=1,
    )
    fwhm_v_3 = Parameter(
        description="Intrinsic FWHM (km/s) of the third Gaussian component",
        default=1000,
    )
    v_off_3 = Parameter(
        description="Velocity offset (km/s) of the third Gaussian component",
        default=0,
    )
    strength_4 = Parameter(
        description="Integrated flux of the fourth Gaussian component",
        default=1,
    )
    fwhm_v_4 = Parameter(
        description="Intrinsic FWHM (km/s) of the fourth Gaussian component",
        default=1000,
    )
    v_off_4 = Parameter(
        description="Velocity offset (km/s) of the fourth Gaussian component",
        default=0,
    )
    strength_5 = Parameter(
        description="Integrated flux of the fifth Gaussian component",
        default=1,
    )
    fwhm_v_5 = Parameter(
        description="Intrinsic FWHM (km/s) of the fifth Gaussian component",
        default=1000,
    )
    v_off_5 = Parameter(
        description="Velocity offset (km/s) of the fifth Gaussian component",
        default=0,
    )
