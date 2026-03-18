from madminer_cli.schemas import DelphesSample


def get_delphes_sample(args) -> DelphesSample:

    hepmc_filename = (
        args.proc_dir / "Events" / f"run_01" / "tag_1_pythia8_events.hepmc.gz"
    )
    lhe_filename = args.proc_dir / "Events" / f"run_01" / "unweighted_events.lhe.gz"

    if args.root_files_dir:
        prefix = args.root_files_dir
    else:
        prefix = args.proc_dir / "Events" / f"run_01"

    delphes_filename = prefix / "tag_1_pythia8_events_delphes.root"

    delphes_filename.parent.mkdir(parents=True, exist_ok=True)

    return DelphesSample(
        hepmc_filename=hepmc_filename,
        lhe_filename=lhe_filename,
        delphes_filename=delphes_filename,
    )

    # samples = []
    # for run, bm in enumerate(args.benchmarks, start=1):

    #     hepmc_filename = (
    #         args.proc_dir / "Events" / f"run_0{run}" / "tag_1_pythia8_events.hepmc.gz"
    #     )
    #     lhe_filename = (
    #         args.proc_dir / "Events" / f"run_0{run}" / "unweighted_events.lhe.gz"
    #     )

    #     if args.root_files_dir:
    #         prefix = args.root_files_dir / args.proc_dir.name
    #     else:
    #         prefix = args.proc_dir / "Events" / f"run_0{run}"

    #     delphes_filename = prefix / "tag_1_pythia8_events_delphes.root"

    #     samples.append(
    #         DelphesSample(
    #             hepmc_filename=hepmc_filename,
    #             sampled_from_benchmark=bm,
    #             # lhe_filename=lhe_filename if lhe_filename.exists() else None,
    #             lhe_filename=lhe_filename,
    #             # delphes_filename=(
    #             #     delphes_filename if delphes_filename.exists() else None
    #             # ),
    #             delphes_filename=delphes_filename,
    #             is_background=args.background,
    #             weights=args.weights,
    #             # k_factor=args.k_factor,  # Not added in `parse_args.py` yet
    #             # systematics=args.systematics,  # Not added in `parse_args.py` yet
    #         )
    #     )
    # return samples
