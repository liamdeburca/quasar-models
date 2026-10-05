from pathlib import Path

import click
from quasar_utils.setup import Info
from quasar_models.host import io as host_io
from quasar_models.host.setup import BC2003
from quasar_models.iron import io as iron_io
from quasar_models.iron.setup import BW, V2003, VW2001
from quasar_models.balmer.continuum import io as balmer_cont_io
from quasar_models.balmer.continuum.setup import QSFIT

INFO = Info()

DEFAULT_IRON_TEMPLATES: tuple[str, ...] = (
    "vw2001", "vw2001_blue", "vw2001_red",
    "bw",
    "v2003", "v2003_blue", "v2003_red",
)

DEFAULT_HOST_TEMPLATES: tuple[tuple[str, int], ...] = tuple(
    ("bc2003", BC2003.get_age_from_path(path) / 1e9)
    for path in BC2003.get_all_paths()
)

DEFAULT_BALMER_CONTINUUM_TEMPLATES: tuple[tuple[str, int, float, float], ...] = (
    ("qsfit", temp, tau, scale)
    for temp, tau, scale in zip(QSFIT.TEMPS, QSFIT.TAUS, QSFIT.SCALES)
)

def _check_path(path: Path, force: bool, verbose: bool, fail_fast: bool) -> None:
    if path.exists():
        if verbose:
            click.echo(f">>> Template already exists at {path!s}!")
        if not force:
            if verbose:
                click.echo(">>> Use -f or --force to overwrite.")
            if not fail_fast:
                return
            raise FileExistsError(path)

def _check_method(cls_: type, method: str, verbose: bool, fail_fast: bool) -> None:
    if not hasattr(cls_, method):
        if verbose:
            click.echo(f">>> Method {method} does not exist on class {cls_.__name__}")
        if not fail_fast:
            return
        raise RuntimeError(f"Method {method} does not exist on class {cls_.__name__}")

###

def _iron_init_helper(
    template: str, 
    force: bool, 
    verbose: bool,
    fail_fast: bool,
) -> None:
    if template not in DEFAULT_IRON_TEMPLATES:
        if verbose:
            click.echo(f">>> Unknown template '{template}'")
        if fail_fast:
            raise ValueError(f"Unknown template '{template}'")
        return 

    if verbose:
        click.echo(f"Initializing template: {template}")

    path = iron_io.convert_path(template)
    _check_path(path, force, verbose, fail_fast)
    
    cls_: type[VW2001 | BW | V2003] = {
        "vw2001": VW2001,
        "bw": BW,
        "v2003": V2003,
    }[template.split("_")[0]]
    cls_name = cls_.__name__

    if template.endswith("_blue"):
        method = "init_blue"
    elif template.endswith("_red"):
        method = "init_red"
    else:
        method = "init_basic"

    _check_method(cls_, method, verbose, fail_fast)

    if verbose:
        click.echo(f">>> Calling method: {cls_name}::{method}")

    try:
        getattr(cls_, method)()
    except Exception as exc:
        if verbose:
            click.echo(f">>> Failed to initialize template with error: {exc=}")
        if not fail_fast:
            return
        raise

    if verbose:
        click.echo(">>> Successfully initialized template!")

def _host_init_helper(
    source: str,
    age: int,
    force: bool, 
    verbose: bool,
    fail_fast: bool,   
) -> None:
    if (source, age) not in DEFAULT_HOST_TEMPLATES:
        if verbose:
            click.echo(f">>> Unknown host template: {source=} {age=} (Gyr)")
        if fail_fast:
            raise ValueError(f"Unknown host template: {source=} {age=} (Gyr)")
        return
    
    if verbose:
        click.echo(f"Initializing host template: {source=} {age=} (Gyr)")

    path = host_io.convert_params_to_path(source, int(1e9 * age))
    _check_path(path, force, verbose, fail_fast)

    cls_: type[BC2003] = {
        "bc2003": BC2003,
    }[source]
    cls_name = cls_.__name__
    method = "init_basic"

    _check_method(cls_, method, verbose, fail_fast)

    if verbose:
        click.echo(f">>> Calling method: {cls_name}::{method}")

    try:
        getattr(cls_, method)()
    except Exception as exc:
        if verbose:
            click.echo(f">>> Failed to initialize template with error: {exc=}")
        if not fail_fast:
            return
        raise

    if verbose:
        click.echo(">>> Successfully initialized template!")

def _balmer_cont_init_helper(
    name: str,
    temp: int,
    tau: float,
    scale: float,
    force: bool,
    verbose: bool,
    fail_fast: bool,
) -> None:
    if verbose:
        click.echo(
            "Initializing Balmer continuum template: "
            f"{name=} {temp=} {tau=:.1f} {scale=:.1f}"
        )

    path = balmer_cont_io.convert_params_to_path(
        temp=temp, 
        tau=tau, 
        scale=scale, 
        info=INFO,
    )
    _check_path(path, force, verbose, fail_fast)

    cls_ = QSFIT
    cls_name = cls_.__name__
    method = "create_template"

    _check_method(cls_, method, verbose, fail_fast)

    if verbose:
        click.echo(f">>> Calling method: {cls_name}::{method}")

    try:
        getattr(cls_, method)()
    except Exception as exc:
        if verbose:
            click.echo(f">>> Failed to initialize template with error: {exc=}")
        if not fail_fast:
            return
        raise

    if verbose:
        click.echo(">>> Successfully initialized template!")
