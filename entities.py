from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List
from qiskit.circuit.quantumcircuit import QubitSpecifier, Gate, QuantumRegister


class FSMMode(Enum):
    FPM = "FPM"
    FFP = "FFP"
    SFSC = "SFSC"


@dataclass
class FSMGateControl:
    qubit: QubitSpecifier
    reverse: bool


@dataclass
class FSMGate:
    op: Callable[..., Gate]
    regs: List[QuantumRegister]
    controls: List[FSMGateControl] = field(default_factory=list)
    params: dict = field(default_factory=dict)
