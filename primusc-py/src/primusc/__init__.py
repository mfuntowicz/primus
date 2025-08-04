from pathlib import Path

import click

from .compiler import Compiler
from .devices import Device
from .llvm import initialize_llvm


@click.group()
@click.option('-v', '--verbose', count=True)
def cli(verbose: int):
    # logger.remove(0)
    #
    # match verbose:
    #     case 0:
    #         logger.add(sys.stdin, level="INFO")
    #     case 1:
    #         logger.add(sys.stdin, level="DEBUG")
    #     case _:
    #         logger.add(sys.stdin, level="TRACE")
    pass

@cli.command()
def arch():
    from llvmlite.binding import Target as LlvmTarget

    initialize_llvm()
    print(LlvmTarget.from_default_triple().triple)


@cli.command()
@click.argument("arch", type=str, required=True)
def supports(arch: str):
    from llvmlite.binding import Target as LlvmTarget

    initialize_llvm()
    target = LlvmTarget.from_triple(arch)
    print(f"{target.triple} ({target.description}): ✅")

@cli.command()
@click.option("--device", type=Device, required=True, help="Device to compile for")
@click.option("--output", type=click.Path(exists=False, writable=True, dir_okay=False, path_type=Path), required=False, default=None, help="Output file")
@click.argument("module", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
def compile(device: Device, module: Path, output: Path = None):
    if not output:
        from importlib.machinery import EXTENSION_SUFFIXES
        output = module.parent / f"{module.stem}.{EXTENSION_SUFFIXES[0][1:]}"  # [1:] to remove the heading "."

    primus = Compiler.from_file(module.parent, module.name)
    primus.compile(output, device=device)


@cli.command()
@click.argument("module", type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path))
def dump(module: Path):
    print(module)
    with open(module, "r") as f:
        for line in f.readlines():
            print(line[:-1])


def main() -> None:
    cli()
