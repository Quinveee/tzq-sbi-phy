import re
from dataclasses import dataclass
from typing import List, Optional

from madminer_dag.schemas import NodeStatus, PhPhases, StatusNodeType

str2phase = {
    "RUN_SETUP": PhPhases.SETUP,
    "PREPARE_GENERATION": PhPhases.PREPARE_GENERATION,
    "RUN_GENERATION": PhPhases.RUN_GENERATION,
    "RUN_ANALYSIS": PhPhases.RUN_ANALYSIS,
    "RUN_AUGMENTATION": PhPhases.RUN_AUGMENTATION,
}


@dataclass
class Node:
    name: str
    status: Optional[NodeStatus]

    @property
    def id(self) -> int:
        return int(self.name.rsplit("_", 1)[-1])

    @property
    def phase(self) -> Optional[int]:
        for key, value in str2phase.items():
            if key in self.name.upper():
                return value
        return None


@dataclass
class PhaseNode:
    name: str
    status: NodeStatus
    phase: int


class NodeStatusParser:

    BLOCK_RGX = re.compile(r"\[(.*?)\]", flags=re.DOTALL)
    TYPE_RGX = re.compile(r"Type = \"(\w*)\"")
    STATUS_RGX = re.compile(r"NodeStatus = (\d)")
    NAME_RGX = re.compile(r"Node = \"(.*)\"")

    def __init__(self, status_lines: List[str], phase: int) -> None:
        self._status_lines = status_lines
        self.from_phase = phase

    @staticmethod
    def get_match(regex: re.Pattern, string: str) -> str:
        m = re.search(regex, string)
        if not m:
            raise ValueError
        return m.group(1)

    @staticmethod
    def parse_lines(lines: List[str]) -> str:
        return "\n".join(l.partition(";")[0].rstrip() for l in lines)

    @classmethod
    def parse_status_blocks(cls, status_str: str) -> List[str]:
        return re.findall(cls.BLOCK_RGX, status_str)

    @classmethod
    def parse_block(cls, block: str) -> Node:
        node_type = cls.get_match(cls.TYPE_RGX, block)

        if node_type != StatusNodeType.STATUS_NODE:
            return Node("", None)

        node_name = cls.get_match(cls.NAME_RGX, block)
        node_status = int(cls.get_match(cls.STATUS_RGX, block))
        return Node(node_name, node_status)  # type: ignore

    def all_nodes(self) -> List[Node]:
        return [
            self.parse_block(b)
            for b in self.parse_status_blocks(self.parse_lines(self._status_lines))
        ]

    def phase_nodes(self) -> List[PhaseNode]:
        return [
            PhaseNode(name=n.name, status=n.status, phase=n.phase)
            for n in self.all_nodes()
            if n.status is not None and n.phase is not None
        ]
