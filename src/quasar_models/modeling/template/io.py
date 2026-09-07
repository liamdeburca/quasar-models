__all__ = ["drop_nonfinite", "drop_nonpos", "get_table_data"]

from typing import Protocol

from astropy.io import fits
from astropy.units import Quantity, Unit
from numpy import isfinite
from quasar_typing.numpy import FloatMatrix, FloatVector, SortedFloatVector
from quasar_typing.pathlib import AbsoluteFITSPath
from quasar_typing.scipy import csr_matrix_
from quasar_utils.setup import Info


class BaseTemplateProtocol(Protocol):
    fwhm: SortedFloatVector
    x: SortedFloatVector
    data: FloatMatrix

    is_logspace: bool
    sigma_res: float | None
    name: str

    path: AbsoluteFITSPath

    _alpha_matrix: csr_matrix_
    _beta_matrix: csr_matrix_
    _xn: SortedFloatVector

    x_norm: float
    fwhm_norm: float
    normalisation: float


def drop_nonpos(arr: FloatVector) -> FloatVector:
    return arr[arr > 0]


def drop_nonfinite(arr: FloatVector) -> FloatVector:
    return arr[isfinite(arr)]


def get_table_data(table: fits.BinTableHDU, cname: str) -> FloatVector | Quantity:
    data = table.data[cname]
    col_index = table.columns.names.index(cname)
    col = table.columns[col_index]
    n = col.array.shape[0]
    return data[:n]


def _save(
    *,
    template: BaseTemplateProtocol,
    info: Info,
) -> fits.HDUList:
    """
    Create a FITS HDUList from a template object.

    Notes
    -----
    The FWHM array is stored in km/s.
    """

    x_unit: str = info.units.wavelength_unit.to_string()
    f_unit: str = info.units.flux_unit.to_string()

    hdul = fits.HDUList()

    hdu: fits.PrimaryHDU = fits.PrimaryHDU(data=template.data)
    hdu.header["NAME"] = template.name
    hdu.header["CTYPE1"] = ("fwhm", "fwhm axis")
    hdu.header["CTYPE2"] = ("x", "spectral axis")
    hdu.header["BUNIT"] = (f_unit, "flux unit")

    if template.is_logspace:
        hdu.header["LOGSPACE"] = "y"
        hdu.header["V_RES"] = (template.sigma_res, "velocity resolution in c")
    else:
        hdu.header["LOGSPACE"] = "n"
        hdu.header["V_RES"] = (None, "velocity resolution in c")

    hdul.append(hdu)

    col_fwhm: fits.Column = fits.Column(
        name="fwhm",
        format="D",
        unit="km/s",
        array=template.fwhm,
    )
    col_x: fits.Column = fits.Column(
        name="x",
        format="D",
        unit=x_unit,
        array=template.x,
    )
    hdul.append(fits.BinTableHDU.from_columns([col_fwhm, col_x]))

    if getattr(template, "_alpha_matrix", None) is not None:
        col_xn = fits.Column(
            name="xn",
            format="D",
            unit=x_unit,
            array=template._xn,
        )
        col_alpha_data = fits.Column(
            name="alpha_data",
            format="D",
            array=template._alpha_matrix.data,
        )
        col_alpha_indices = fits.Column(
            name="alpha_indices",
            format="K",
            array=template._alpha_matrix.indices,
        )
        col_alpha_indptr = fits.Column(
            name="alpha_indptr",
            format="K",
            array=template._alpha_matrix.indptr,
        )
        col_beta_data = fits.Column(
            name="beta_data",
            format="D",
            array=template._beta_matrix.data,
        )
        col_beta_indices = fits.Column(
            name="beta_indices",
            format="K",
            array=template._beta_matrix.indices,
        )
        col_beta_indptr = fits.Column(
            name="beta_indptr",
            format="K",
            array=template._beta_matrix.indptr,
        )
        hdu: fits.BinTableHDU = fits.BinTableHDU.from_columns(
            [
                col_xn,
                col_alpha_data,
                col_alpha_indices,
                col_alpha_indptr,
                col_beta_data,
                col_beta_indices,
                col_beta_indptr,
            ]
        )

        hdr = hdu.header

        # no. of _xn values
        hdr["XN_VAL"] = template._xn.size
        # Alpha-matrix
        hdr["ASHAPE"] = "{}/{}".format(*template._alpha_matrix.shape)
        # no. of alpha-matric values
        hdr["A_VAL"] = template._alpha_matrix.data.size
        # no. of alpha-matrix indices
        hdr["A_IND"] = template._alpha_matrix.indices.size
        # no. of alpha-matrix index pointers
        hdr["A_PTR"] = template._alpha_matrix.indptr.size
        # Beta-matrix
        hdr["BSHAPE"] = "{}/{}".format(*template._beta_matrix.shape)
        # no. of beta-matrix values
        hdr["B_VAL"] = template._beta_matrix.data.size
        # no. of beta-matrix indices
        hdr["B_IND"] = template._beta_matrix.indices.size
        # no. of beta-matrix index pointers
        hdr["B_PTR"] = template._beta_matrix.indptr.size

        hdul.append(hdu)

    return hdul


def _load(
    *,
    path: AbsoluteFITSPath,
    info: Info,
) -> dict:
    """
    Create a dictionary of template attributes from a FITS file.
    """
    kwargs = {
        "n_scales": info.convolution.n_scales,
        "fwhm": None,
        "x": None,
        "data": None,
        "is_logspace": None,
        "sigma_res": None,
        "name": None,
        "path": path,
        "_alpha_matrix": None,
        "_beta_matrix": None,
        "_xn": None,
    }
    with fits.open(path) as hdul:
        v_unit = Unit(hdul[1].columns[0].unit)
        x_unit = Unit(hdul[1].columns[1].unit)
        f_unit = Unit(hdul[0].header["BUNIT"])

        def transform_kms(arr: FloatVector) -> FloatVector:
            return info.units.getKMS(arr * v_unit)

        def transform_wavelength(arr: FloatVector) -> FloatVector:
            return info.units.getWavelength(arr * x_unit)

        def transform_flux(arr: FloatVector) -> FloatVector:
            return info.units.getFlux(arr * f_unit)

        kwargs["data"] = data = transform_flux(hdul[0].data)
        kwargs["fwhm"] = transform_kms(hdul[1].data["fwhm"])[: data.shape[0]]
        kwargs["x"] = transform_wavelength(hdul[1].data["x"])[: data.shape[1]]

        kwargs["name"] = hdul[0].header["NAME"]

        if hdul[0].header["LOGSPACE"].lower() == "y":
            kwargs["is_logspace"] = True
            kwargs["sigma_res"] = float(hdul[0].header["V_RES"])
        else:
            kwargs["is_logspace"] = False
            kwargs["sigma_res"] = None

        if len(hdul) > 2:
            hdu2: fits.BinTableHDU = hdul[2]

            kwargs["_xn"] = transform_wavelength(hdu2.data["xn"])
            kwargs["_alpha_matrix"] = csr_matrix_(
                (
                    hdu2.data["alpha_data"],
                    hdu2.data["alpha_indices"],
                    hdu2.data["alpha_indptr"],
                ),
                shape=tuple(map(int, hdu2.header["ASHAPE"].strip().split("/"))),
            )
            kwargs["_beta_matrix"] = csr_matrix_(
                (
                    hdu2.data["beta_data"],
                    hdu2.data["beta_indices"],
                    hdu2.data["beta_indptr"],
                ),
                shape=tuple(map(int, hdu2.header["BSHAPE"].strip().split("/"))),
            )

    return kwargs
