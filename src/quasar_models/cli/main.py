import click

from . import utils


@click.group()
def cli():
    pass

### Initialisation of iron templates

@cli.group("iron")
def iron():
    pass

@iron.command("init")
def iron_init(
    templates: tuple[str, ...],
    # Flags
    force: bool,
    verbose: bool,
    fail_fast: bool,
):
    if "*" in templates:
        templates = utils.DEFAULT_IRON_TEMPLATES
    else:
        templates = tuple(t.strip().lower() for t in templates)

    for template in templates:
        utils._iron_init_helper(template, force, verbose, fail_fast)
        
if __name__ == "__main__":
    cli()