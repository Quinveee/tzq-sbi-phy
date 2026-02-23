from pathlib import Path

from madminer_dag.node_parser import NodeStatusParser
from madminer_dag.parse_utils import Args, CreateArgs, RedoArgs
from madminer_dag.ph_dag import PhMetaDAG


def create(args: CreateArgs):
    PhMetaDAG(filename=args.name, conf=args.conf, samples=args.samples, from_phase=args.from_phase).run(
        gvars_filename=str(args.gvars), dag_conf=args.dag_conf
    )


def redo(args: RedoArgs):
    node_parser = NodeStatusParser(status_lines=args.status_lines, phase=args.phase)
    set_done = [
        node
        for node in node_parser.phase_nodes()
        if node.phase < node_parser.from_phase
    ]

    dag_file = (args.dirname / args.dirname.stem).with_suffix(".dag")
    rescue_file = Path(str(dag_file) + f".rescue{args.rescue:03d}")
    with open(rescue_file, "w") as f:
        f.writelines(f"DONE {node.name}\n" for node in set_done)

    print(
        f"Run \ncondor_submit_dag -DoRescueFrom {args.rescue} {dag_file}\nto redo DAG"
    )


args2fun = {CreateArgs: create, RedoArgs: redo}


def run(args: Args) -> None:
    args_cls = type(args)

    func = args2fun.get(args_cls)

    assert func, f"Invalid arguments class: {args_cls}"

    func(args)
