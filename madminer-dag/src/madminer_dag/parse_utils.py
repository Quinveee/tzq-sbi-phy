import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from madminer_dag.schemas import PhPhases

_CREATE_STR2PHASE = {
    "generation": PhPhases.PREPARE_GENERATION,
    "delphes": PhPhases.RUN_DELPHES,
    "analysis": PhPhases.RUN_ANALYSIS,
}


@dataclass
class ConfigDir:
    conf_yml: Dict[str, Any]
    dag_conf: Path


@dataclass
class ExperimentDir:
    name: Path
    status_file: Path
    dag_file: Path


@dataclass
class CreateArgs:
    conf: Dict[str, Any]
    name: Path
    dag_conf: Path
    gvars: Path
    samples: str
    from_phase: PhPhases


@dataclass
class RedoArgs:
    dirname: Path
    status_lines: List[str]
    phase: PhPhases
    rescue: int


Args = Union[CreateArgs, RedoArgs]


def ensure_config_dir(config_dir: Path) -> ConfigDir:
    dag_yml_path = config_dir / "dag.yml"
    dag_conf_path = config_dir / "dag.conf"

    files = (
        config_dir / "benchmarks.yml",
        dag_yml_path,
        dag_conf_path,
        config_dir / "observables.yml",
    )

    for f in files:
        if not (f).exists():
            raise FileNotFoundError(f)

    with open(dag_yml_path, "r") as f:
        dag_yml = yaml.safe_load(f)

    return ConfigDir(dag_yml, dag_conf_path)


def parse_create(arguments: argparse.Namespace) -> CreateArgs:
    config_dir = Path(arguments.config_dir)
    if not config_dir.exists() or not config_dir.is_dir():
        raise ValueError(f"Invalid experiment dir {config_dir}")

    config = ensure_config_dir(config_dir)

    # Embed samples type in dag dir so different sample runs don't clobber each other
    dag_stem = f"{config_dir.stem}_{arguments.samples}"
    return CreateArgs(
        conf=config.conf_yml,
        name=Path("dag", dag_stem, dag_stem + ".dag"),
        dag_conf=config.dag_conf,
        gvars=arguments.vars,
        samples=arguments.samples,
        from_phase=_CREATE_STR2PHASE[arguments.from_phase],
    )


def ensure_experiment_dir(experiment_dir: Path):
    status_file = experiment_dir / (experiment_dir.stem + ".dag.status")
    dag_file = experiment_dir / (experiment_dir.stem + ".dag")
    if not status_file.exists() or not dag_file.exists():
        raise FileNotFoundError(f"Missing either {status_file} or {dag_file}")
    return ExperimentDir(experiment_dir, status_file, dag_file)


str2phase = {
    "delphes": PhPhases.RUN_DELPHES,
    "analysis": PhPhases.RUN_ANALYSIS,
    "augmentation": PhPhases.RUN_AUGMENTATION,
}


def parse_redo(arguments: argparse.Namespace) -> RedoArgs:
    experimet_dir = ensure_experiment_dir(arguments.experiment)
    with open(experimet_dir.status_file, "r") as f:
        status_lines = f.readlines()
    return RedoArgs(
        experimet_dir.name,
        status_lines,
        str2phase[arguments.from_phase],
        arguments.rescue,
    )
