from pathlib import Path
from subprocess import run

directory = Path(__file__).resolve().parent / "src/quasar_models"


def _run_script(script_path: Path, plot: bool = False):
    assert script_path.is_file()
    cmd = ["python", str(script_path)]
    if plot:
        cmd.append("--plot")
    run(cmd, check=True)


def setup_iron(plot: bool = False):
    setup_file = directory / "iron/setup.py"
    _run_script(setup_file, plot)


def setup_balmer_series(plot: bool = False):
    setup_file = directory / "balmer/series/setup.py"
    _run_script(setup_file, plot)


def setup_balmer_continuum(plot: bool = False):
    setup_file = directory / "balmer/continuum/setup.py"
    _run_script(setup_file, plot)


def setup_host(plot: bool = False):
    setup_file = directory / "host/setup.py"
    _run_script(setup_file, plot)


def main(plot: bool = False):
    setup_iron(plot)
    setup_balmer_series(plot)
    setup_balmer_continuum(plot)
    setup_host(plot)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(plot=args.plot)