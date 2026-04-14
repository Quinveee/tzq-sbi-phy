import yaml

from madminer_cli.decorators import pack, validate_paths
from madminer_cli.parse_cls import (
    AnalysisArgs,
    AnalysisSample,
    AugmentationArgs,
    DelphesArgs,
    DelphesSample,
    GenArgs,
    PlotArgs,
    SetupArgs,
)
from madminer_cli.schemas import Benchmark, Cut, MorphingSetup, Observable, Parameter
from madminer_cli.utils import get_delphes_sample


@pack(SetupArgs)
def parse_setup(args):
    yaml_config = yaml.safe_load(args.infile)
    args.parameters = [Parameter(**p) for p in yaml_config["parameters"]]
    args.benchmarks = [Benchmark(**p) for p in yaml_config["benchmarks"]]
    args.morphing_setup = MorphingSetup(**yaml_config["morphing"])
    return args


@pack(GenArgs)
@validate_paths(
    "setup_file",
    "mg_dir",
    "mg_config_file",
    "cards_dir",
    "proc_card",
    "param_card",
    "run_card",
)
def parse_gen(args):
    args.pythia_card = args.cards_dir / args.pythia_card if args.pythia_card else None
    for attr in ("proc_card", "param_card", "run_card"):
        setattr(args, attr, args.cards_dir / getattr(args, attr))
    return args


# FIXME: Arguments for delphes and analysis are similar but not the same,
# and enforcing the same structure does not make sense


@pack(DelphesArgs)
@validate_paths("delphes_card", "delphes_dir", "proc_dir")
def parse_delphes(args):
    args.sample = get_delphes_sample(args)
    return args


@pack(AnalysisArgs)
@validate_paths("setup_file", "proc_dir")
def parse_analysis(args):
    yaml_config = yaml.safe_load(args.infile)
    observables = [] or yaml_config["observables"]
    cuts = [] or yaml_config["cuts"]

    args.observables = [Observable(**o) for o in observables]
    args.cuts = [Cut(name="CUT", **c) for c in cuts]
    args.outfile = args.outfile.format(args.proc_dir.name)

    delphes_sample = get_delphes_sample(args)
    args.sample = AnalysisSample(
        hepmc_filename=delphes_sample.hepmc_filename,
        lhe_filename=delphes_sample.lhe_filename,
        delphes_filename=delphes_sample.delphes_filename,
        sampled_from_benchmark=args.benchmark,
        is_background=args.is_background,
        weights=args.weights,
        # TODO: Add following to parse args
        # k_factor=args.k_factor,
        # systematics=args.systematics,
    )
    return args


@pack(PlotArgs)
def parse_plot(args):
    """Plot observable distributions for analyzed .h5 files or x_*.npy samples."""
    from pathlib import Path

    resolved: list = []
    for f in args.input_files:
        p = Path(f).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() not in {".h5", ".npy"}:
            raise ValueError(
                f"Unsupported input file type for plotting: {p} (expected .h5 or .npy)"
            )
        resolved.append(p)
    args.input_files = resolved
    args.outdir = Path(args.outdir).expanduser()
    return args


@pack(AugmentationArgs)
@validate_paths("events_file")
def parse_augmentation(args):
    import madminer.sampling as sampling  # don't remove, used for eval below

    args.nproc = args.nproc if args.nproc > 0 else None
    args.theta0 = eval(args.theta0)
    args.theta1 = eval(args.theta1)
    args.theta_test = eval(args.theta_test)

    return args
