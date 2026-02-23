import argparse
import sys
from pathlib import Path
from typing import List

from madminer_dag.parse_utils import Args, parse_create, parse_redo


def parse_args(args: List[str]) -> Args:
    parser = argparse.ArgumentParser(prog="madminer_dag")

    parser.add_argument(
        "-s",
        "--samples",
        default="features",
        type=str,
        choices=["features", "particles", "both"],
        help="Generate feature-level data, particle-level data, or both simultaneously",
    )

    subparsers = parser.add_subparsers(title="commands")

    create = subparsers.add_parser("create")
    create.add_argument(
        "-c",
        "--config-dir",
        dest="config_dir",
        required=True,
        type=Path,
        help="Path to the experiment config folder.",
    )
    create.add_argument(
        "-v",
        "--vars",
        default="global.vars.dag",
        type=Path,
        help="Name of the file to store global macros",
    )
    create.add_argument(
        "-p",
        "--from-phase",
        dest="from_phase",
        type=str,
        default="generation",
        choices=("generation", "delphes", "analysis"),
        help="Start dag from this phase, skipping all earlier phases",
    )

    create.set_defaults(func=parse_create)

    redo = subparsers.add_parser("redo")
    redo.add_argument(
        "-e",
        "--experiment",
        type=Path,
        help="Experimet DAG folder with the required files",
        required=True,
    )
    redo.add_argument(
        "-p",
        "--from-phase",
        dest="from_phase",
        type=str,
        required=True,
        choices=("delphes", "analysis", "augmentation"),
        help="Phase to redo existing dag from",
    )
    redo.add_argument(
        "--rescue", type=int, default=1, help="Rescue number for the created file"
    )
    redo.set_defaults(func=parse_redo)

    arguments = parser.parse_args(args)

    try:
        return arguments.func(arguments)
    except Exception as ex:
        parser.error(f"{type(ex).__name__}: {ex}")
