from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from madminer_dag.dag import DAG
from madminer_dag.node import Node
from madminer_dag.schemas import PhPhases
from madminer_dag.typing import PathLike

__all__ = ["PhMetaDAG"]


class PhDAG(DAG):
    def __init__(self, id: int, dirname: PathLike, **kwds):
        super().__init__(Path(dirname) / f"{id}.dag", **kwds)
        self.id = id
        self.phases = {
            PhPhases.PREPARE_GENERATION: self.add_prepare_generation,
            PhPhases.RUN_GENERATION: self.add_run_generation,
            PhPhases.RUN_DELPHES: self.add_run_delphes,
            PhPhases.RUN_ANALYSIS: self.add_run_analysis,
        }

    def add_prepare_generation(
        self, parent_node: Optional[Node] = None, **kwds
    ) -> Node:
        node = Node(
            name=f"PREPARE_GENERATION_{self.id}", script="submit/prepare_generation.sub"
        )
        job_vars = [
            "cards_dir",
            "proc_card",
            "run_card",
            "param_card",
            "pythia_card",
            "benchmark",
            "proc_dir",
            "is_background",
        ]
        vars_to_add = {k: v for k, v in kwds.items() if k in job_vars}
        # Default is_background to false if not specified; normalize to lowercase string
        is_bg = vars_to_add.get("is_background", False)
        vars_to_add["is_background"] = "true" if is_bg else "false"
        node.add_vars(vars_to_add)
        node.add_post(
            script="scripts/POST_prepare_generation",
            args=[
                self.id,
                "$JOBID",
                kwds["proc_dir"],
                kwds["n_subprocesses"],
                kwds["benchmark"],
                kwds["reweight_card_insert"],
                kwds["tmp_dir"],
                self.filename.parent,
                str(kwds.get("is_background", False)).lower(),
            ],
        )
        self.add_node(node, from_parent=parent_node)
        return node

    def add_run_generation(self, parent_node: Optional[Node] = None, **kwds) -> Node:
        node = Node(
            name=f"RUN_GENERATION_{self.id}", script="submit/run_generation.sub"
        )
        node.add_vars({"ngen": self.id})
        self.add_node(node, from_parent=parent_node)
        return node

    def add_run_delphes(self, parent_node: Optional[Node] = None, **kwds) -> Node:
        node = Node(name=f"RUN_DELPHES_{self.id}", script="submit/run_delphes.sub")
        node.add_vars({"ngen": self.id})
        self.add_node(node, from_parent=parent_node)
        return node

    def add_run_analysis(
        self,
        parent_node: Optional[Node] = None,
        observables_override: Optional[str] = None,
        h5_dir_override: Optional[str] = None,
        suffix: str = "",
        **kwds,
    ) -> Node:
        name = f"RUN_ANALYSIS{suffix}_{self.id}"
        node = Node(name=name, script="submit/run_analysis.sub")
        vars = {"ngen": self.id}
        if observables_override:
            vars["OBSERVABLES"] = observables_override
        if h5_dir_override:
            vars["H5_DIR"] = h5_dir_override
        node.add_vars(vars)
        self.add_node(node, from_parent=parent_node)
        return node

    def add_run_analysis_both(
        self,
        parent_node: Optional[Node],
        observables_features: str,
        h5_dir_features: str,
        observables_particles: str,
        h5_dir_particles: str,
    ) -> List[Node]:
        """Add two parallel analysis nodes (features + particles) after parent."""
        features_node = self.add_run_analysis(
            parent_node=parent_node,
            observables_override=observables_features,
            h5_dir_override=h5_dir_features,
            suffix="_FEATURES",
        )
        particles_node = self.add_run_analysis(
            parent_node=parent_node,
            observables_override=observables_particles,
            h5_dir_override=h5_dir_particles,
            suffix="_PARTICLES",
        )
        return [features_node, particles_node]

    def add_from_phase(self, phase: PhPhases, both_conf: Optional[Dict] = None, **kwds) -> None:
        if phase not in self.phases:
            raise ValueError(
                f"Invalid phase: {phase}. Valid phases are: {self.phases.keys()}"
            )
        # Run all phases up to (but not including) RUN_ANALYSIS normally
        parent_node = self.phases[phase](parent_node=None, **kwds)
        for i in range(phase + 1, PhPhases.RUN_ANALYSIS):
            node = self.phases[i](parent_node=parent_node, **kwds)  # type: ignore
            parent_node = node

        # Handle analysis phase
        if both_conf is not None:
            self.add_run_analysis_both(
                parent_node=parent_node,
                **both_conf,
            )
        else:
            self.phases[PhPhases.RUN_ANALYSIS](parent_node=parent_node, **kwds)


class PhMetaDAG(DAG):
    def __init__(self, filename: PathLike, conf: Dict[str, Any], samples: str = "features", from_phase: int = PhPhases.PREPARE_GENERATION, **kwds) -> None:
        super().__init__(filename, **kwds)
        self._conf = self.preprocess_conf(conf, samples)
        self._from_phase = from_phase
        self.gvars_filename = None
        self.gvars = {}

    @staticmethod
    def preprocess_conf(conf: Dict[str, Any], samples: str = "features") -> Dict[str, Any]:
        import copy
        conf["setup_file"] = str(Path(conf["setup_dir"]) / conf["setup_file"])
        obs_base = Path(conf["observables"]).parent
        h5_base = Path(conf["h5_dir"])
        aug_outdir_base = Path(conf["augmentation"]["outdir"]).parent

        if samples == "both":
            # Store per-sample sub-configs; keep base h5_dir for PRE_run_setup
            conf["_both"] = {
                s: {
                    "observables": str(obs_base / "observables.d" / f"{s}.yml"),
                    "h5_dir": str(h5_base / s),
                    "augmentation": {**copy.deepcopy(conf["augmentation"]), "outdir": str(aug_outdir_base / s)},
                }
                for s in ("features", "particles")
            }
        else:
            conf["observables"] = str(obs_base / "observables.d" / f"{samples}.yml")
            conf["h5_dir"] = str(h5_base / samples)
            conf["augmentation"]["outdir"] = str(aug_outdir_base / samples)
        return conf

    @staticmethod
    def get_proc_dir(base_dir: PathLike, cards_dir: PathLike, benchmark: str) -> Path:
        cdir = str(Path(cards_dir).name)
        return Path(base_dir) / (cdir + "_" + benchmark)

    def init_directory(self) -> None:
        if self.dirname.exists() and self.dirname.is_dir():
            shutil.rmtree(self.dirname)
        self.dirname.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        gvars_filename: str = "global.vars.dag",
        dag_conf: Optional[PathLike] = None,
    ) -> None:
        self.init_directory()

        # 1. Add Config
        if dag_conf:
            self.add(f"CONFIG {dag_conf}")

        # 2. Add global variables (across all DAGs)
        # NOTE: DON'T include them in MetaDAG (naming collisions)
        self.gvars_filename = self.dirname / gvars_filename
        gvars_subdag = DAG(filename=self.gvars_filename)

        # gvars_keys = [
        #     "setup_file",
        #     "setup_conf",
        #     "mg_dir",
        #     "delphes_dir",
        #     "ld_library_path",
        #     "tmp_dir",
        #     "observables",
        #     "h5_dir",
        #     "delphes_card",
        #     "root_files_dir",
        # ]

        self.gvars = {k: v for k, v in self._conf.items() if isinstance(v, str)}
        gvars_subdag.add_global_vars(self.gvars)
        self.add_subdag(gvars_subdag)

        # 3. Add physics subdags
        self.add_ph_subdags()

        # 4. Add additional output files
        status_filename = self.dirname / (self.filename.name + ".status")
        self.add(f"NODE_STATUS_FILE {status_filename} 45")
        dot_filename = str(self.filename).replace(".dag", ".dot")
        self.add(f"DOT {dot_filename}")

        self.compile()
        self.write()

    def _make_augment_node(self, name: str, h5_dir: str, aug_conf: Dict, log_dir: str) -> Node:
        node = Node(name=name, script="submit/run_augmentation.sub")
        experiment_dir = Path(h5_dir).parent.parent
        samples_name = Path(h5_dir).name
        vars = dict(aug_conf)
        vars.update({
            "events_file": experiment_dir / samples_name / (experiment_dir.name + ".h5"),
            "log_dir": log_dir,
        })
        node.add_vars(vars)
        node.add_pre(script="scripts/PRE_run_augmentation", args=[h5_dir, log_dir])
        return node

    def _add_augmentation_nodes(self, parents_str: Optional[str] = None) -> None:
        """Add augmentation node(s), optionally chained after the given parents."""
        is_both = "_both" in self._conf
        log_dir = self.gvars.get("log_dir", self._conf.get("log_dir", ""))
        if is_both:
            for s in ("features", "particles"):
                node = self._make_augment_node(
                    name=f"RUN_AUGMENTATION_{s.upper()}",
                    h5_dir=self._conf["_both"][s]["h5_dir"],
                    aug_conf=self._conf["_both"][s]["augmentation"],
                    log_dir=log_dir,
                )
                self.add_node(node)
                if parents_str:
                    self.add(f"PARENT {parents_str} CHILD {node.name}")
        else:
            h5_dir = self.gvars.get("h5_dir", self._conf.get("h5_dir", ""))
            node = self._make_augment_node(
                name="RUN_AUGMENTATION",
                h5_dir=h5_dir,
                aug_conf=self._conf["augmentation"],
                log_dir=log_dir,
            )
            self.add_node(node)
            if parents_str:
                self.add(f"PARENT {parents_str} CHILD {node.name}")

    def add_ph_subdags(self) -> None:
        is_both = "_both" in self._conf

        # If starting from augmentation, skip setup and all physics subdags
        if self._from_phase >= PhPhases.RUN_AUGMENTATION:
            self._add_augmentation_nodes(parents_str=None)
            return

        # 1. Add setup step
        setup_node = Node(name="RUN_SETUP", script="submit/run_setup.sub")
        setup_vars = ["setup_file", "setup_conf", "log_dir"]
        setup_node.add_vars({k: v for k, v in self._conf.items() if k in setup_vars})
        # For 'both', PRE_run_setup gets the base h5_dir so it clears the parent,
        # wiping both h5/features and h5/particles on re-runs
        pre_setup_vars = ["setup_dir", "tmp_dir", "h5_dir", "processes_dir", "log_dir"]
        setup_node.add_pre(
            script="scripts/PRE_run_setup",
            args=[self._conf[v] for v in pre_setup_vars],
        )
        self.add_node(setup_node)

        # 2. Add per-process subdags
        c = 1
        ph_subdags_names = []
        both_conf = None
        if is_both:
            both_conf = {
                "observables_features": self._conf["_both"]["features"]["observables"],
                "h5_dir_features": self._conf["_both"]["features"]["h5_dir"],
                "observables_particles": self._conf["_both"]["particles"]["observables"],
                "h5_dir_particles": self._conf["_both"]["particles"]["h5_dir"],
            }
        for process in self._conf["processes"]:
            proc_dir = self.get_proc_dir(
                base_dir=self._conf["processes_dir"],
                cards_dir=process["cards_dir"],
                benchmark=process["benchmark"],
            )
            process.update({"proc_dir": proc_dir, "tmp_dir": self._conf["tmp_dir"]})
            for _ in range(int(process["runs"])):
                ph_subdag = PhDAG(id=c, dirname=self.dirname / str(c), name=f"PH_{c}")
                if self.gvars_filename is not None:
                    ph_subdag.add(f"INCLUDE {self.gvars_filename}")
                ph_subdag.add_global_vars({"log_dir": ph_subdag.dirname})
                ph_subdag.add_from_phase(self._from_phase, both_conf=both_conf, **process)
                self.add_subdag(ph_subdag, is_splice=True, from_parent=setup_node)
                ph_subdags_names.append(ph_subdag.name)
                c += 1

        # 3. Run augmentation (one node per sample type)
        parents_str = " ".join(ph_subdags_names)
        self._add_augmentation_nodes(parents_str=parents_str)
