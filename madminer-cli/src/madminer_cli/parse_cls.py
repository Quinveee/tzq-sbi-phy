from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from madminer_cli.schemas import (
    AnalysisSample,
    Benchmark,
    Cut,
    DelphesSample,
    MorphingSetup,
    Observable,
    Parameter,
)


@dataclass
class SetupArgs:
    parameters: List[Parameter]
    benchmarks: List[Benchmark]
    morphing_setup: MorphingSetup
    outfile: str


@dataclass
class GenArgs:
    setup_file: Path
    log_file: Path
    mg_dir: Path
    proc_card: Path
    param_card: Path
    run_card: Path
    pythia_card: Optional[Path]
    benchmarks: List[str]
    mg_config_file: Path
    proc_dir: str
    is_background: bool
    now: bool


@dataclass
class DelphesArgs:
    delphes_card: Path
    delphes_dir: Path
    sample: DelphesSample
    log_file: Path


@dataclass
class AnalysisArgs:
    setup_file: Path
    sample: AnalysisSample
    observables: List[Observable]
    cuts: List[Cut]
    outfile: str


@dataclass
class AugmentationArgs:
    events_file: Path
    outdir: str
    theta0: ...
    theta1: ...
    theta_test: ...
    n_samples: int
    n_samples_test: int
    nproc: Optional[int]
    validation_split: float = 0.0
    test_split: float = 0.2


Args = Union[SetupArgs, GenArgs, DelphesArgs, AugmentationArgs, AnalysisArgs]
