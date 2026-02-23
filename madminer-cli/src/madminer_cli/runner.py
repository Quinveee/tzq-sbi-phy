from __future__ import annotations

import importlib
import logging
import os
import tempfile
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any, Type

from madminer_cli import LOGGER
from madminer_cli.parse_cls import (
    AnalysisArgs,
    Args,
    AugmentationArgs,
    DelphesArgs,
    GenArgs,
    SetupArgs,
)

if TYPE_CHECKING:
    from madminer import DelphesReader, MadMiner, SampleAugmenter


class Runner:

    @property
    def miner(self) -> Type[MadMiner]:
        return self._lazy_import("_miner", "madminer", "MadMiner")

    @property
    def delphes_reader(self) -> Type[DelphesReader]:
        return self._lazy_import("_delphes_reader", "madminer", "DelphesReader")

    @property
    def sample_augmenter(self) -> Type[SampleAugmenter]:
        return self._lazy_import("_sample_augmenter", "madminer", "SampleAugmenter")

    def __init__(self, args: Args) -> None:

        self.logger = LOGGER.getChild(f"{__name__}.{self.__class__.__name__}")
        self.arguments = args
        self._miner = None
        self._delphes_reader = None
        self._sample_augmenter = None

        self.run_args_map = {
            SetupArgs: self.run_setup,
            GenArgs: self.run_generate,
            DelphesArgs: self.run_delphes,
            AnalysisArgs: self.run_analysis,
            AugmentationArgs: self.run_augmentation,
        }

    def _lazy_import(
        self, attr_name: str, import_path: str, class_name: str
    ) -> Type[Any]:
        if getattr(self, attr_name) is None:
            module = importlib.import_module(import_path, class_name)

            self._reset_logging()
            setattr(self, attr_name, getattr(module, class_name))
        return getattr(self, attr_name)

    def _reset_logging(self) -> None:
        for key in logging.Logger.manager.loggerDict:
            if "madminer" not in key:
                logging.getLogger(key).setLevel(logging.WARNING)

    def run_setup(self, arguments: SetupArgs) -> None:
        miner = self.miner()
        for param in arguments.parameters:
            miner.add_parameter(
                lha_block=param.lha_block,
                lha_id=param.lha_id,
                parameter_name=param.parameter_name,
                param_card_transform=param.param_card_transform,
                morphing_max_power=param.morphing_max_power,
                parameter_range=param.parameter_range,
            )

        # Set benchmarks
        for bm in arguments.benchmarks:
            miner.add_benchmark(
                parameter_values=bm.parameter_values,
                benchmark_name=bm.benchmark_name,  # type: ignore (madminer mistake)
                # verbose=bm.verbose,
            )

        # Morphing
        morphing = arguments.morphing_setup
        miner.set_morphing(
            max_overall_power=morphing.max_overall_power,
            n_bases=morphing.n_bases,
            include_existing_benchmarks=morphing.include_existing_benchmarks,
            n_trials=morphing.n_trials,
            n_test_thetas=morphing.n_test_thetas,
        )
        miner.save(filename=arguments.outfile)

    def run_generate(self, arguments: GenArgs) -> None:
        miner = self.miner()
        miner.load(arguments.setup_file)

        # TODO: fix `only_prepare_script` to be `False` when `now`
        miner.run_multiple(
            mg_directory=str(arguments.mg_dir),
            proc_card_file=arguments.proc_card,
            param_card_template_file=arguments.param_card,
            run_card_files=[arguments.run_card],
            mg_process_directory=arguments.proc_dir,
            pythia8_card_file=arguments.pythia_card,
            configuration_file=arguments.mg_config_file,
            sample_benchmarks=arguments.benchmarks,
            is_background=arguments.is_background,
            only_prepare_script=False if arguments.now else True,
            log_directory=str(
                arguments.log_file.parent / Path(arguments.proc_dir).name
            ),
            temp_directory=tempfile.mkdtemp(),
            order="LO",
            # initial_command=arguments.initial_command,
            # ufo_model_directory=arguments.ufo_model_directory,
            # systematics=arguments.systematics,
            # python_executable=arguments.python_executable,
        )

        # TODO
        if arguments.now:
            cmd = os.path.abspath(
                os.path.join(arguments.proc_dir, "madminer", "run.sh")
            )
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            out, err = proc.communicate()
            exitcode = proc.returncode

            if exitcode != 0:
                raise RuntimeError(
                    f"Calling command {cmd} returned exit code {exitcode}.\n\nStd output: {out}\n\nError output: {err}\n\n"
                )
            return

    def run_delphes(self, arguments: DelphesArgs) -> None:

        # TODO: Add delphes_filename here below so that .root files
        # go to /dcache
        from madminer.utils.interfaces.delphes import run_delphes

        # Just one sample at a time
        sample = arguments.sample
        run_delphes(
            delphes_directory=arguments.delphes_dir,
            delphes_card_filename=arguments.delphes_card,
            hepmc_sample_filename=sample.hepmc_filename,
            delphes_sample_filename=sample.delphes_filename,
            log_file=arguments.log_file.parent / "Delphes.log",
            initial_command=None,
            overwrite_existing_delphes_root_file=True,
            delete_unzipped_file=True,
        )

        # delphes_reader = self.delphes_reader(arguments.setup_file)
        # for sample in arguments.samples:
        #     delphes_reader.add_sample(
        #         hepmc_filename=str(sample.hepmc_filename),
        #         lhe_filename=str(sample.lhe_filename),
        #         sampled_from_benchmark=sample.sampled_from_benchmark,
        #         is_background=sample.is_background,
        #         k_factor=sample.k_factor,
        #         weights=sample.weights,
        #     )

        # delphes_reader.run_delphes(
        #     delphes_directory=arguments.delphes_dir,
        #     delphes_card=arguments.delphes_card,
        #     log_file=arguments.log_file.parent / "Delphes.log",
        # )

    def run_analysis(self, arguments: AnalysisArgs) -> None:
        delphes_reader = self.delphes_reader(arguments.setup_file)

        # Just one sample at a time
        sample = arguments.sample
        delphes_reader.add_sample(
            hepmc_filename=str(sample.hepmc_filename),
            delphes_filename=str(sample.delphes_filename),
            # TODO: This below could be none depending on the value
            # of `sample.weights`
            lhe_filename=str(sample.lhe_filename),
            is_background=sample.is_background,
            sampled_from_benchmark=sample.sampled_from_benchmark,
            k_factor=sample.k_factor,
            weights=sample.weights,
        )

        for obs in arguments.observables:
            delphes_reader.add_observable(
                name=obs.name,
                definition=obs.val_expression,
                required=obs.is_required,
                default=obs.val_default,
            )

        for cut in arguments.cuts:
            delphes_reader.add_cut(
                definition=cut.val_expression, required=cut.is_required
            )

        delphes_reader.analyse_delphes_samples()
        delphes_reader.save(arguments.outfile)

    def run_augmentation(self, arguments: AugmentationArgs, strategy: str = "ratio") -> None:

        # TODO: Add support for other sampling strategies
        validation_split = arguments.validation_split
        test_split = arguments.test_split
        theta_0 = arguments.theta0
        theta_1 = arguments.theta1
        theta_test = arguments.theta_test
        n_samples = arguments.n_samples
        n_test_samples = arguments.n_samples_test
        outdir = arguments.outdir
        processes = arguments.nproc


        sampler = self.sample_augmenter(filename=arguments.events_file)


        # Generate Training and Testing Ratios for the two thetas of interest (theta_1 and theta_test)
        _ = sampler.sample_train_ratio(
            theta0=theta_0,
            theta1=theta_1,
            n_samples=n_samples,
            folder=outdir,
            filename="train_ratio",
            sample_only_from_closest_benchmark=True,
            return_individual_n_effective=True,
            n_processes=processes,
            validation_split=validation_split,
            test_split=test_split,
        )

        _ = sampler.sample_train_ratio(
            theta0=theta_0,
            theta1=theta_test,
            n_samples=n_test_samples,
            folder=outdir,
            filename="test_ratio",
            sample_only_from_closest_benchmark=True,
            return_individual_n_effective=True,
            n_processes=processes,
            validation_split=validation_split,
            test_split=test_split,
            partition="test",
        )

        # Generate Training and Testing Scores for the two thetas of interest (theta_1 and theta_test)
        # TODO: Add theta score argument, for now using denominator
        # of sampling ratios (sm)
        _ = sampler.sample_train_local(
            theta=theta_1,
            n_samples=n_samples,
            folder=outdir,
            filename="train_score",
            sample_only_from_closest_benchmark=True,
            validation_split=validation_split,
            test_split=test_split,
        )

        _ = sampler.sample_train_local(
            theta=theta_test,
            n_samples=n_test_samples,
            folder=outdir,
            filename="test_score",
            sample_only_from_closest_benchmark=True,
            validation_split=validation_split,
            test_split=test_split,
            partition="test",
        )

        # _ = sampler.sample_test(
        #     theta=arguments.theta_test,
        #     n_samples=arguments.n_samples_test,
        #     folder=arguments.outdir,
        #     filename="test",
        #     n_processes=arguments.nproc,  # type: ignore
        #     validation_split=validation_split,
        #     test_split=test_split,
        # )

    def run(self) -> None:

        self.logger.debug(f"Parsed parameters: {str(self.arguments)}")

        args_cls = type(self.arguments)
        run_fun = self.run_args_map.get(args_cls)

        if not run_fun:
            raise ValueError(f"Invalid argument type: {args_cls}")

        run_fun(self.arguments)
