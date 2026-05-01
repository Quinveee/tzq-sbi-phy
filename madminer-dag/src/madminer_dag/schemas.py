from enum import Enum, IntEnum


class PhPhases(IntEnum):
    SETUP = 0
    PREPARE_GENERATION = 1
    RUN_GENERATION = 2  # merged event generation + Delphes
    RUN_ANALYSIS = 3
    RUN_AUGMENTATION = 4


class ScriptType(Enum):
    POST = "POST"
    PRE = "PRE"


class NodeType(Enum):
    JOB = "JOB"
    SPLICE = "SPLICE"


class NodeStatus(IntEnum):
    NOT_READY = 0
    READY = 1
    PRERUN = 2
    SUBMITTED = 3
    POSTRUN = 4
    DONE = 5
    ERROR = 6
    FUTILE = 7


class StatusNodeType:
    STATUS_NODE = "NodeStatus"
    STATUS_DAG = "DagStatus"
