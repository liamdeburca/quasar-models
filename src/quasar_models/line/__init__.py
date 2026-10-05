__all__ = [
    "GaussianModel",
    "VProfileCopy",
    "VProfileCopy1G",
    "VProfileCopy2G",
    "VProfileCopy3G",
    "VProfileCopy4G",
    "VProfileCopy5G",
    "VProfileCopyDict",
]
from collections import Counter, defaultdict
from typing import TypeVar, Union

from astropy.modeling import Parameter
from quasar_typing.astropy import CompoundModel_
from quasar_typing.bounds import AstropyBounds

from ..modeling.utils import LinearTie
from ..utils.astropy import apply_bounds
from .gaussian import GaussianModel
from .vprofilecopies import (
    VProfileCopy1G,
    VProfileCopy2G,
    VProfileCopy3G,
    VProfileCopy4G,
    VProfileCopy5G,
    _VProfileCopy,
)

VProfileCopy = Union[
    VProfileCopy1G,
    VProfileCopy2G,
    VProfileCopy3G,
    VProfileCopy4G,
    VProfileCopy5G,
]
VProfileCopyDict: dict[int, VProfileCopy] = {
    1: VProfileCopy1G,
    2: VProfileCopy2G,
    3: VProfileCopy3G,
    4: VProfileCopy4G,
    5: VProfileCopy5G,
}

V = TypeVar("V", bound=VProfileCopy)

###

def create_vprofilecopy_from_gaussian(
    model: Union[GaussianModel, CompoundModel_[GaussianModel]],
    name: str,
    *,
    wave: float,
    strength_scale_value: float = 1.0,
    strength_scale_bounds: AstropyBounds | None = None,
    strength_scale_fixed: bool | None = None,
) -> VProfileCopy:
    ms = (model,) if model.n_submodels == 1 else tuple(model)

    assert all(isinstance(m, GaussianModel) for m in ms)
    assert len(ms) in VProfileCopyDict
    assert len({m.pure_name for m in ms}) == 1

    return VProfileCopyDict[len(ms)].create(
        wave, name, ms[0].linetype, *ms,
        strength_scale_value=strength_scale_value,
        strength_scale_bounds=strength_scale_bounds,
        strength_scale_fixed=strength_scale_fixed,
        sigma_res=ms[0].sigma_res,
        master_name=ms[0].pure_name,
        n_sigmas=ms[0].n_sigmas,
        adapt=True,
    )

def create_vprofilecopy_from_vprofilecopy[V](
    model: V,
    name: str,
    *,
    wave: float,
    strength_scale_value: float = 1.0,
    strength_scale_bounds: AstropyBounds | None = None,
    strength_scale_fixed: bool | None = None,
) -> V:
    meta = model.meta.copy()
    meta["wave"] = wave
    meta["master"] = model.pure_name

    out = model.copy()
    out.name = name
    out.wave = wave
    out.master_name = model.pure_name

    out.strength_scale.value = strength_scale_value
    if strength_scale_bounds is not None:
        out.strength_scale.bounds = strength_scale_bounds
        out.strength_scale.value = apply_bounds.__wrapped__(
            out.strength_scale.value, 
            out.strength_scale.bounds,
        )

    if strength_scale_fixed is not None:
        out.strength_scale.fixed = strength_scale_fixed

    for pname in (
        prefix + f"_{i+1}"
        for prefix in ("strength", "fwhm_v", "v_off")
        for i in range(out.n_profiles)
    ):
        param: Parameter = getattr(out, pname)
        master_param: Parameter = getattr(model, pname)

        # Set value, bounds and fixed status from the master model
        param.value = master_param.value
        param.bounds = master_param.bounds
        param.fixed = master_param.fixed
        # Tie the parameter to the corresponding parameter
        param.tied = LinearTie(
            a=1.0, 
            b=0.0, 
            model_name=model.pure_name, 
            parameter_name=pname,
        )

    return out

def create_vprofilecopy(
    model: Union[GaussianModel, CompoundModel_[GaussianModel], VProfileCopy],
    name: str,
    *,
    wave: float,
    strength_scale_value: float = 1.0,
    strength_scale_bounds: AstropyBounds | None = None,
    strength_scale_fixed: bool | None = None,
) -> VProfileCopy:
    func = create_vprofilecopy_from_vprofilecopy \
        if model.n_submodels == 1 and isinstance(model, _VProfileCopy) \
        else create_vprofilecopy_from_gaussian
    return func(
        model, name,
        wave=wave,
        strength_scale_value=strength_scale_value,
        strength_scale_bounds=strength_scale_bounds,
        strength_scale_fixed=strength_scale_fixed,
    )

###

def rename_gaussians(
    model: Union[
        GaussianModel, 
        VProfileCopy, 
        CompoundModel_[Union[GaussianModel, VProfileCopy]],
    ],
    ) -> None:
    ms: list[GaussianModel] = sorted(
        [
            m
            for m in ((model,) if model.n_submodels == 1 else model)
            if isinstance(m, GaussianModel)
        ],
        key=lambda m: m.sorting_key,
    )
    model_counts: dict[str, int] = Counter(m.pure_name for m in ms)
    curr_count: dict[str, int] = defaultdict(int)
    for m in ms:
        if model_counts[m.pure_name] == 1:
            m.name = m.pure_name
        else:
            curr_count[m.pure_name] += 1
            m.name = f"{m.pure_name}#{curr_count[m.pure_name]}"