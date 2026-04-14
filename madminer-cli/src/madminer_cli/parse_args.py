import argparse
import logging
import os
from pathlib import Path
from typing import List

from madminer_cli import __doc__ as PACKAGE_DOCSTRING
from madminer_cli import __version__
from madminer_cli.base import BASE_DELPHES, BASE_INFILE, BASE_SETUP
from madminer_cli.parse_cls import Args
from madminer_cli.parse_funs import (
    parse_analysis,
    parse_augmentation,
    parse_delphes,
    parse_gen,
    parse_plot,
    parse_setup,
)

__all__ = ["parse_args"]


# TODO: doesn't work for decorated functions ...
def doc(func):
    return func.__doc__.splitlines()[0] if func.__doc__ else ""


# TODO: Cast args to Path directly when possible


def parse_args(args: List[str]) -> Args:

    # Main parser
    parser = argparse.ArgumentParser(
        prog="madminer",
        description=PACKAGE_DOCSTRING,
    )
    subparsers = parser.add_subparsers(title="commands")

    # 0: Parent parser
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    parser.add_argument(
        "-V", "--verbose", default=0, action="count", help="increment verbosity"
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        default="logs/madminer_cli.log",
        type=Path,
        help="File to write logs to",
    )

    # 1: Setup parsing
    parser_setup = subparsers.add_parser(
        "run_setup",
        description="""
        To setup the miner. Input is expected to come from a .yaml file with the 
        neccessary fields (or stdin).
        """,
        help=doc(parse_setup),
        parents=[BASE_INFILE],
    )
    parser_setup.add_argument(
        "outfile",
        type=str,
        help="Output filepath to write setup file to",
    )
    parser_setup.set_defaults(arg_handler=parse_setup)

    # 2: Event Gen parsing
    parser_gen = subparsers.add_parser(
        "run_generation",
        description="""
        For event generation.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=doc(parse_gen),
        parents=[BASE_SETUP],
    )
    parser_gen.add_argument(
        "cards_dir",
        type=str,
        help="Dir with the cards for diagram generation and event simulation",
    )
    parser_gen.add_argument(
        "proc_dir",
        help="Name of the dir to be created for storing diagrams and events",
    )
    parser_gen.add_argument(
        "--mg-dir",
        type=str,
        default=os.getenv("MG_FOLDER_PATH"),
        help="""MadGraph root dir.""",
    )
    parser_gen.add_argument(
        "--proc-card",
        type=str,
        default="proc_card_mg5.dat",
        help="Name of process card to be found in the cards dir",
    )
    parser_gen.add_argument(
        "--param-card",
        type=str,
        default="param_card.dat",
        help="Name of param card to be found in the cards dir",
    )
    parser_gen.add_argument(
        "--run-card",
        type=str,
        default="run_card.dat",
        help="Name of run card to be found in the cards dir",
    )
    parser_gen.add_argument(
        "--pythia-card",
        type=str,
        default="pythia8_card.dat",  # should be None?
        help="Name of the Pythia8 card to be found in the cards dir",
    )
    # TODO: Allowing different benchmarks for one process dir is disadvised
    # remove
    parser_gen.add_argument(
        "-b",
        "--benchmarks",
        nargs="+",
        type=str,
        default=["sm"],
        help="Benchmark keys to sample from",
    )
    parser_gen.add_argument(
        "--mg-config-file",
        type=str,
        default="~/.mg5/mg5_configuration.txt",
        help="MadGraph configuration file",
    )
    parser_gen.add_argument(
        "--is_background",
        action="store_true",
        help="Specify this if sampling from background",
    )
    parser_gen.add_argument(
        "--now",
        action="store_true",
        help="Specify this if you want to run generation (not only preparation scripts)",
    )
    parser_gen.set_defaults(arg_handler=parse_gen)

    # 3. Delphes parsing
    parser_delphes = subparsers.add_parser(
        "run_delphes",
        description="Run Delphes on the showered events.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BASE_DELPHES],
        help=doc(parse_delphes),
    )
    parser_delphes.add_argument(
        "--delphes-card",
        type=str,
        required=True,
        help="The delphes configuration card path",
    )
    parser_delphes.add_argument(
        "--delphes-dir",
        type=str,
        default=f"{os.getenv('MG_FOLDER_PATH', '.')}/Delphes",
        help="The base directory of the Delphes program.",
    )
    parser_delphes.set_defaults(arg_handler=parse_delphes)

    # 4. Analysis parsing
    parser_analysis = subparsers.add_parser(
        "run_analysis",
        description="""
        Run analysis. Observables are expected to come from the `infile` .yaml file
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BASE_INFILE, BASE_SETUP, BASE_DELPHES],
        help=doc(parse_analysis),
    )
    parser_analysis.add_argument(
        "outfile",
        type=str,
        help="The name of the madminer .h5 file after running analysis and cuts",
    )
    parser_analysis.add_argument(
        "-b",
        "--benchmark",
        type=str,
        required=True,
        help="Specify the benchmark from which the event samples come from.",
    )
    parser_analysis.add_argument(
        "--weights",
        default="lhe",
        choices=("lhe", "delphes"),
        help="Where to get the sample weights from",
    )
    parser_analysis.add_argument(
        "--is_background",
        action="store_true",
        help="Specify this if sampling from background",
    )

    parser_analysis.set_defaults(arg_handler=parse_analysis)

    # 5. Augmentation parsing
    parser_augmentation = subparsers.add_parser(
        "run_augmentation",
        description="""
        Run augmentation
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=doc(parse_augmentation),
    )
    parser_augmentation.add_argument(
        "events_file",
        type=str,
        help="Path to the madminer .h5 file with the analyzed samples",
    )
    parser_augmentation.add_argument(
        "outdir",
        type=str,
        help="The name of the dir under the data directory",
    )
    parser_augmentation.add_argument(
        "--theta0",
        type=str,
        default="sampling.random_morphing_points(1000, [('gaussian', 0.0, 0.5)])",
        help="Python expressoin to sample the numerator points from",
    )
    parser_augmentation.add_argument(
        "--theta1",
        type=str,
        default="sampling.benchmark('sm')",
        help="Python expression to sample the denominator points",
    )
    parser_augmentation.add_argument(
        "--theta-test",
        type=str,
        default="sampling.benchmark('sm')",
        help="Python expression to sample theta test",
    )
    parser_augmentation.add_argument(
        "--n-samples", type=int, default=500_000, help="Number of samples for training"
    )
    parser_augmentation.add_argument(
        "--n-samples-test", type=int, default=1000, help="Number of samples for testing"
    )
    parser_augmentation.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of cores for parallel sampling. -1 for all available",
    )
    parser_augmentation.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        dest="validation_split",
        help="Fraction of training samples to use for validation",
    )
    parser_augmentation.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        dest="test_split",
        help="Fraction of events to reserve for the test partition",
    )
    parser_augmentation.set_defaults(arg_handler=parse_augmentation)

    # 6. Plot parsing
    parser_plot = subparsers.add_parser(
        "run_plot",
        description="""
        Plot observable distributions for one or more analyzed files.
        One composite figure per input file is written to <outdir>/<dataset>/<stem>.png.
        Supported inputs are analyzed MadMiner .h5 files and morphed x_*.npy files.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=doc(parse_plot),
    )
    parser_plot.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="One or more analyzed .h5 files or morphed x_*.npy files",
    )
    parser_plot.add_argument(
        "--outdir",
        type=str,
        default="images",
        help="Base output dir for the images",
    )
    parser_plot.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset label used as a subdir (e.g. '1d', '3d', 'experiment_so_cht')",
    )
    parser_plot.add_argument(
        "--level",
        type=str,
        required=True,
        choices=("particle", "feature"),
        help="Whether the input .h5 is particle-level or feature-level; becomes a subdir",
    )
    parser_plot.add_argument(
        "--observables",
        nargs="+",
        default=None,
        help="Subset of observable names to plot (default: all in the file)",
    )
    parser_plot.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "For .h5: benchmarks to overlay (default: all). "
            "For x_*.npy + y_*.npy: optional labels for the class overlays."
        ),
    )
    parser_plot.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize distributions to the total cross section",
    )
    parser_plot.add_argument(
        "--log",
        action="store_true",
        help="Draw y axes on a log scale",
    )
    parser_plot.add_argument(
        "--n-bins",
        dest="n_bins",
        type=int,
        default=50,
        help="Number of histogram bins",
    )
    parser_plot.add_argument(
        "--n-cols",
        dest="n_cols",
        type=int,
        default=3,
        help="Number of observable columns in the composite figure",
    )
    parser_plot.add_argument(
        "--uncertainties",
        choices=("nuisance", "none"),
        default="none",
        help="Whether to draw nuisance uncertainty bands",
    )
    parser_plot.add_argument(
        "--convert",
        dest="convert_to_ptetaphi",
        action="store_true",
        help=(
            "Replace each <prefix>_{E,px,py,pz} quartet with <prefix>_{pt,eta,phi} "
            "before plotting. --observables (if given) must use the converted names."
        ),
    )
    parser_plot.add_argument(
        "--plot-jet",
        dest="plot_jet_multiplicity",
        action="store_true",
        help=(
            "Also write <stem>_njets.png with the weighted distribution of the per-event "
            "jet count, inferred from finite j<i>_E (or j<i>_pt) columns."
        ),
    )
    parser_plot.set_defaults(arg_handler=parse_plot)

    # parse args
    arguments = parser.parse_args(args)

    if not getattr(arguments, "arg_handler", None):
        parser.error("Too few arguments provided")

    log_level = logging.INFO
    if arguments.verbose:
        log_level = max(logging.DEBUG, logging.ERROR - 10 * arguments.verbose)

    # Ensure log directory exists
    os.makedirs(arguments.log_file.parent, exist_ok=True)

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    logging.basicConfig(
        format="%(asctime)s %(name)-20.20s [%(levelname)s] %(message)s",
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=arguments.log_file, mode="w"),
        ],
    )

    try:
        return arguments.arg_handler(arguments)
    except Exception as ex:
        # parser.error(f"{type(ex).__name__}: {ex}")
        raise
