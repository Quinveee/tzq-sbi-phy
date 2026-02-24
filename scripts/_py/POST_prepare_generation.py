#!/bin/env python3

"""
After sucessful generation of process directories, some cards under
the `<PROCESS_DIR>/madminer` directory need post-processing.
    1. Change process command on top of the reweight cards
    2. Fill in the RND_SEED placeholder in the run cards
Additionally, helper files with the path to the process dir and
the benchmark identifier are created
"""

import argparse
import glob
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

parser = argparse.ArgumentParser(prog="POST_prepare_generation")
pexists = os.path.exists


@dataclass
class Args:
    n_gen: int
    job_id: str
    proc_dir: str
    n_subprocesses: int
    rwg_card: str
    tmp_dir: str
    benchmark: str
    is_background: bool = False


def parse_args(args: List[str]) -> Args:
    parser.add_argument(
        "n_gen", type=int, help="Identifier for process number directory"
    )
    parser.add_argument(
        "job_id", type=str, help="ID of the job to create the random seed for event gen"
    )
    parser.add_argument(
        "proc_dir", help="Madgraph process directory with the madminer folder"
    )
    parser.add_argument(
        "n_subprocesses", type=int, help="Number of subprocesses to consider"
    )
    parser.add_argument("benchmark", type=str, help="Morphing benchmark identifier")
    parser.add_argument(
        "--is-background",
        action="store_true",
        dest="is_background",
        help="Mark this process as a background process (no morphing)",
    )
    parser.add_argument(
        "--rwg-card",
        type=str,
        help="Part of the rewewight card to be inserted",
        default=None,
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        help="Path to temporary directory",
        default=tempfile.gettempdir(),
    )
    arguments = parser.parse_args(args)

    proc_dir = arguments.proc_dir + "." + arguments.job_id
    if not pexists(proc_dir):
        parser.error(f"Process directory {proc_dir} not found!")

    arguments.rwg_card = arguments.rwg_card if arguments.rwg_card != "None" else None
    if arguments.rwg_card is not None:
        if not pexists(arguments.rwg_card):
            parser.error(
                f"Reweight insert card provided {arguments.rwg_card} not found!"
            )

    return Args(
        n_gen=arguments.n_gen,
        n_subprocesses=arguments.n_subprocesses,
        job_id=arguments.job_id,
        proc_dir=proc_dir,
        rwg_card=arguments.rwg_card,
        tmp_dir=arguments.tmp_dir,
        benchmark=arguments.benchmark,
        is_background=arguments.is_background,
    )


def edit_rwg_cards(rwg_card_insert: Optional[str], proc_dir: str) -> None:

    if rwg_card_insert is None:
        print(f"No reweight card insert given")
        return

    print(f"Found card: {rwg_card_insert} in cards dir")

    with open(rwg_card_insert) as f:
        rwg_insert_str = [line.rstrip("\n") for line in f.readlines()]

    rwg_insert_str.insert(0, f"# Contents pasted from card {rwg_card_insert}\n")

    glob_pattern = str(Path(proc_dir) / "madminer" / "cards" / "reweight_card_*.dat")
    rwg_cards = glob.glob(glob_pattern)

    if not rwg_cards:
        raise FileNotFoundError(
            f"No reweight cards found using glob pattern: {glob_pattern}"
        )

    for rwg_card in rwg_cards:
        with open(rwg_card, "r") as f:
            rwg_contents_str = [line.rstrip("\n") for line in f.readlines()]

        with open(rwg_card, "w") as f:
            f.writelines([l + "\n" for l in (rwg_insert_str + rwg_contents_str)])

        print(f"Written contents of {rwg_card_insert} to {rwg_card}")


def fill_rnd_seed_runcards(
    n_gen: int, job_id: str, proc_dir: str, n_subprocesses: int
) -> None:
    # Line below follows recommendations here: https://answers.launchpad.net/mg5amcnlo/+question/254698
    seed = int(float(job_id)) + (n_gen * n_subprocesses)

    glob_pattern = str(Path(proc_dir) / "madminer" / "cards" / "run_card*")
    run_cards = glob.glob(glob_pattern)

    if not run_cards:
        raise FileNotFoundError(f"No cards found using glob pattern: {glob_pattern}")

    for run_card in run_cards:
        print(f"Found run card at {run_card}")
        with open(run_card) as f:
            contents = "\n".join(
                [l.rstrip("\n").replace("RND_SEED", str(seed)) for l in f.readlines()]
            )

        print(f"Replacing random seed placeholder by seed {seed}")
        with open(run_card, "w") as f:
            f.writelines([l + "\n" for l in contents.split("\n")])


def write_procdir_to_file(
    n_gen: int, proc_dir: str, benchmark: str, tmp_dir: str, is_background: bool = False
) -> None:
    fn = Path(tmp_dir) / f"procdir.{n_gen}.tmp"
    print(f"Filename to write process directory to: {fn}")

    with open(fn, "w") as f:
        f.write(f"{proc_dir} {benchmark} {str(is_background).lower()}")

    print(
        f"Process directory {proc_dir}, benchmark {benchmark}, is_background {is_background} written to file: {fn}"
    )


def main(arguments: Args) -> None:
    print(f"Running with parser arguments: {arguments}")

    edit_rwg_cards(
        rwg_card_insert=arguments.rwg_card,
        proc_dir=arguments.proc_dir,
    )

    fill_rnd_seed_runcards(
        n_gen=arguments.n_gen,
        job_id=arguments.job_id,
        proc_dir=arguments.proc_dir,
        n_subprocesses=arguments.n_subprocesses,
    )

    write_procdir_to_file(
        n_gen=arguments.n_gen,
        proc_dir=arguments.proc_dir,
        benchmark=arguments.benchmark,
        tmp_dir=arguments.tmp_dir,
        is_background=arguments.is_background,
    )


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
