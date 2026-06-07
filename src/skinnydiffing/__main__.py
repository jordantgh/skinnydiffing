"""CLI entry point for skinnydiffing. Use `skinnydiff --help` for usage instructions."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from .api import diff as diff_api
from .config import load_config
from .normalisers import REGISTRY as NORMALISER_REGISTRY
from .runner import run_comparison, run_config

app = typer.Typer(
    name="skinnydiff",
    help="Efficient cell-level table diffing.",
    no_args_is_help=True,
)


class FileFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    readstat = "readstat"


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None

    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def parse_value(value: str) -> Any:
    lowered = value.lower()

    if lowered in {"true", "yes", "y"}:
        return True
    if lowered in {"false", "no", "n"}:
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        return value


def parse_reader_options(items: list[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}

    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"Expected KEY=VALUE, got {item!r}")

        key, value = item.split("=", 1)
        key = key.strip()

        if not key:
            raise typer.BadParameter(f"Expected non-empty KEY in {item!r}")

        parsed[key] = parse_value(value.strip())

    return parsed


def print_result_summary(
    name: str, output_dir: Path, written: dict[str, Path], n_diffs: int
) -> None:
    typer.echo(f"{name}: {n_diffs} cell difference(s)")

    if written:
        typer.echo(f"wrote {len(written)} file(s) to {output_dir}")
    else:
        typer.echo("no output files written")


@app.command("diff")
def diff_command(
    source: Annotated[
        Path,
        typer.Argument(help="Source table path: file or directory."),
    ],
    target: Annotated[
        Path,
        typer.Argument(help="Target table path: file or directory."),
    ],
    keys: Annotated[
        str | None,
        typer.Option(
            "--keys",
            "-k",
            help="Comma-separated key column(s), e.g. --keys id,period.",
        ),
    ] = None,
    positional: Annotated[
        bool,
        typer.Option(
            "--positional",
            help="Compare rows by strict row position instead of key columns.",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Directory to write diff CSV outputs.",
        ),
    ] = Path("./diff_output"),
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Basename/prefix for output files.",
        ),
    ] = "diff",
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            "-c",
            help="Comma-separated columns to compare. Defaults to all shared non-key columns.",
        ),
    ] = None,
    exclude: Annotated[
        str | None,
        typer.Option(
            "--exclude",
            "-x",
            help="Comma-separated columns to ignore.",
        ),
    ] = None,
    normalise: Annotated[
        str | None,
        typer.Option(
            "--normalise",
            help=(
                "Comma-separated normalisers to apply before diffing. "
                f"Available: {', '.join(sorted(NORMALISER_REGISTRY))}"
            ),
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Number of columns per comparison batch. Use 0 to disable batching.",
        ),
    ] = 50,
    check_keys: Annotated[
        bool,
        typer.Option(
            "--check-keys/--no-check-keys",
            help="Validate key uniqueness before joining.",
        ),
    ] = True,
    file_format: Annotated[
        FileFormat | None,
        typer.Option(
            "--format",
            help="Force input format instead of inferring from extension.",
        ),
    ] = None,
    glob: Annotated[
        str | None,
        typer.Option(
            "--glob",
            help='Glob used when SOURCE/TARGET are directories, e.g. "*.parquet".',
        ),
    ] = None,
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            help="Reader option applied to both sides, as KEY=VALUE. Repeatable.",
        ),
    ] = None,
    source_option: Annotated[
        list[str] | None,
        typer.Option(
            "--source-option",
            help="Reader option applied only to SOURCE, as KEY=VALUE. Repeatable.",
        ),
    ] = None,
    target_option: Annotated[
        list[str] | None,
        typer.Option(
            "--target-option",
            help="Reader option applied only to TARGET, as KEY=VALUE. Repeatable.",
        ),
    ] = None,
    collect_lazy_inputs: Annotated[
        bool,
        typer.Option(
            "--collect-lazy-inputs",
            help="Allow non-Polars lazy dataframe inputs to be collected.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Diff two tables directly."""

    configure_logging(verbose)

    key_list = None if positional else split_csv(keys)

    if not positional and not key_list:
        raise typer.BadParameter("Pass --keys id[,id2,...] or use --positional.")

    compare_cols = split_csv(compare)
    exclude_cols = split_csv(exclude)
    normalisers = split_csv(normalise)

    shared_options = parse_reader_options(option)
    source_options = {
        **shared_options,
        **parse_reader_options(source_option),
    }
    target_options = {
        **shared_options,
        **parse_reader_options(target_option),
    }

    if file_format is not None:
        source_options["file_format"] = file_format.value
        target_options["file_format"] = file_format.value

    if glob is not None:
        source_options["glob"] = glob
        target_options["glob"] = glob

    result = diff_api(
        source,
        target,
        keys=key_list,
        compare=compare_cols,
        exclude=exclude_cols,
        normalise=normalisers,
        batch_size=None if batch_size <= 0 else batch_size,
        check_keys=check_keys,
        name=name,
        source_options=source_options,
        target_options=target_options,
        collect_lazy_inputs=collect_lazy_inputs,
    )

    written = result.write(out, basename=name)
    print_result_summary(name, out, written, result.n_diffs)


@app.command("run")
def run_command(
    config: Annotated[
        Path,
        typer.Argument(help="YAML config file containing one or more comparisons."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Override output_dir for every comparison in the config.",
        ),
    ] = None,
    collect_lazy_inputs: Annotated[
        bool | None,
        typer.Option(
            "--collect-lazy-inputs/--no-collect-lazy-inputs",
            help="Override collect_lazy_inputs for every comparison.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Run one or more comparisons from a YAML config file."""

    configure_logging(verbose)

    if out is None:
        results = run_config(config, collect_lazy_inputs=collect_lazy_inputs)
        typer.echo(f"completed {len(results)} comparison(s)")
        return

    results = []
    for job in load_config(config):
        job["output_dir"] = str(out)

        result = run_comparison(
            job,
            collect_lazy_inputs=collect_lazy_inputs,
        )
        result.write(out)
        results.append(result)

    typer.echo(f"completed {len(results)} comparison(s)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
