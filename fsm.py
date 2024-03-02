from qiskit.circuit.quantumcircuit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    AncillaRegister,
    Qubit,
)
from typing import Dict, List
from numpy import floor, log2
from .entities import FSMGate, FSMGateControl, FSMMode
from .gates import (
    bitwise_cand,
    extend,
    match,
    unary_or,
    rccopy,
    reverse,
    rot,
)


class FSM:
    def __init__(self, x: str, y: str, d: int, mode: FSMMode, **kwargs):
        """Initializes a fixed substring matching algorithm.

        Args:
            `x`, `y` (str): Non-empty input strings in binary format, equally sized.
            `d` (int): Fixed length of the common substring(s) to search.
        """
        if not x or not y:
            raise ValueError("Input strings cannot be empty")
        if len(x) != len(y):
            raise ValueError("Input strings lengths do not match")
        if d < 2:
            raise ValueError("Substring length must be at least 2")
        try:
            int(f"0b{x}", base=0)
            int(f"0b{y}", base=0)
        except:
            raise TypeError("Input strings are not in binary format")

        self._mode = mode
        if self._mode == FSMMode.FFP.value:
            _pos = kwargs.get("starting_pos", None)
            if not _pos:
                raise ValueError(
                    "Starting position is requested when initializing the FFP problem. Please specify the starting position using the keyword argument starting_pos"
                )
            self._j = _pos
            print(_pos)
        else:
            self._j = None
        self._d = d
        self._n = len(x)
        self._logn = floor(log2(self._n)).astype(int)
        self._nd = floor(log2(self._d)).astype(int) + 1

        print("  Creating input registries...")
        self._rd = QuantumRegister(self._nd, "d")

        self._x = x
        self._y = y
        # input registers

        self._rx = QuantumRegister(self._n, "X")
        self._ry = QuantumRegister(self._n, "Y")
        print(f"    - d: {self._nd} qubits")
        print(f"    - x: {self._n} qubits")
        print(f"    - y: {self._n} qubits")

        print(f"  Creating {self._logn+1} λ bitvectors registries...")

        # lambda registers
        self._rli: List[QuantumRegister] = [
            QuantumRegister(self._n, f"\lambda{i}") for i in range(self._logn + 1)
        ]

        print(f"  Creating {self._logn+2} D bitvectors registries...")

        # D registers
        self._rddinit = QuantumRegister(self._n + 1, "D-1")
        self._rddi: List[QuantumRegister] = [
            QuantumRegister(self._n + 1, f"D{i}") for i in range(self._logn)
        ]

        # ancillae register
        # n/2 * log(n) for the rotation gates + n * log(n) for the conjunction gate
        print(f"  Creating {3 * int(self._n / 2) * self._logn} ancillae qubits...")

        self._ranc = AncillaRegister(3 * int(self._n / 2) * self._logn, "anc")
        # result qubit
        self._rout = QuantumRegister(1, "out")
        self._cout_outcome = ClassicalRegister(1, "found")
        self._cout_pos = ClassicalRegister(self._n + 1, "begins")

    @property
    def input(self):
        """FSM algorithm inputs"""
        return {
            "x": self._x,
            "y": self._y,
            "d": self._d,
            "mode": self._mode,
            "from_pos": self._j,
        }

    @property
    def regs(self) -> Dict[str, QuantumRegister | List[QuantumRegister]]:
        """FSM algorithm quantum registers"""
        return {
            "x": self._rx,
            "y": self._ry,
            "d": self._rd,
            "li": self._rli,
            "ddinit": self._rddinit,
            "ddi": self._rddi,
            "anc": self._ranc,
            "out": self._rout,
        }

    @property
    def cregs(self) -> Dict[str, ClassicalRegister]:
        """FSM algorithm classical registers"""
        return {"found": self._cout_outcome, "begins": self._cout_pos}

    def instantiate(self):
        return _FSMInstance(self)


class _FSMInstance:
    def __init__(self, fsm: FSM):
        print("  Setting FSM instance up...")
        self._ready = False
        self._measured = False
        self._fsm = fsm
        _regs: List[QuantumRegister] = []
        for reg in self._fsm.regs.values():
            if isinstance(reg, list):
                _regs.extend(reg)
            else:
                _regs.append(reg)

        print("  Creating circuit...")
        self._qc = QuantumCircuit(*_regs, *list(self._fsm.cregs.values()))

        print("  Initializing input registers...")
        for i, bit in enumerate(bin(self.d)[2:]):
            if bit == "1":
                self._qc.x(self.di[i])

        _inx, _iny = (self._fsm.input["x"], self._fsm.input["y"])
        _regx, _regy = self.xy
        for i, bit in enumerate(_inx):
            if bit == "1":
                self._qc.x(_regx[i])
        for i, bit in enumerate(_iny):
            if bit == "1":
                self._qc.x(_regy[i])

        if self.mode == FSMMode.FPM.value:
            _ddinit = "".join(["1", "0" * self.n])
        elif self.mode == FSMMode.FFP.value:
            _ddinit = "".join(
                ["0" * self.from_pos, "1", "0" * (self.n - self.from_pos)]
            )
        else:
            _ddinit = "1" * (self.n + 1)
        for i, bit in enumerate(_ddinit):
            if bit == "1":
                self._qc.x(self.ddinit[i])
        print("FSM instance initialization successful.")

    @property
    def ready(self) -> bool:
        """Returns True if the circuit is ready to be executed)"""
        return self._ready

    @property
    def measured(self) -> bool:
        """Returns True if the circuit output was already measured"""
        return self._measured

    @property
    def mode(self) -> FSMMode:
        """Circuit mode (used algorithm)"""
        return self._fsm.input["mode"]

    @property
    def from_pos(self) -> int:
        """Position index from which to search for common substrings.

        Raises:
            - `KeyError` if circuit mode is not FFP"""
        if not self._fsm.input["from_pos"]:
            raise KeyError("FSM instance is not in FFP mode")
        return self._fsm.input["from_pos"]

    @property
    def xy(self) -> tuple[QuantumRegister]:
        """Input registers"""
        return self._fsm.regs["x"], self._fsm.regs["y"]

    @property
    def n(self) -> int:
        """Length of input strings"""
        return self._fsm.regs["x"].size

    @property
    def d(self) -> QuantumRegister:
        """Fixed length of the common substring to search"""
        return self._fsm.input["d"]

    @property
    def dsize(self) -> int:
        """Length of binary representation of d"""
        return floor(log2(self.d)).astype(int) + 1

    @property
    def di(self) -> QuantumRegister:
        """Reversed binary representation of d"""
        return self._fsm.regs["d"]

    @property
    def qc(self) -> QuantumCircuit:
        """Quantum circuit instance"""
        return self._qc

    @property
    def li(self) -> List[QuantumRegister]:
        """λ bitvectors"""
        return self._fsm.regs["li"]

    @property
    def ddinit(self) -> QuantumRegister:
        """D-1 bitvector used in the initialization phase"""
        return self._fsm.regs["ddinit"]

    @property
    def ddi(self) -> List[QuantumRegister]:
        """Di bitvectors"""
        return self._fsm.regs["ddi"]

    @property
    def ancillae(self) -> AncillaRegister:
        """Ancillae register to achieve parallelism in rotation"""
        return self._fsm.regs["anc"]

    @property
    def out(self) -> QuantumRegister:
        """Final output register"""
        return self._fsm.regs["out"]

    @property
    def result(self) -> tuple[ClassicalRegister]:
        """Measurement results.

        Returns:
            - `found`: algorithm outcome (match was found or not)
            - `begins`: start position of the common substring"""
        if not self.measured:
            raise RuntimeError(
                "The output qubits were not measured yet. Please execute the algorithm before looking for results."
            )
        return self._fsm.cregs["found"], self._fsm.cregs["begins"]

    def build(self):
        REV = FSMGate(reverse, [self.di])
        M = FSMGate(match, [*self.xy, self.li[0]])

        self.apply(REV, M)

        print("  Applying extension gates to λ registries...")
        for i in range(len(self.li) - 1):
            EXT = FSMGate(extend, [self.li[i], self.li[i + 1]], params={"i": i + 1})
            self.apply(EXT)

        # repeat for each Di register, with i = 1...log(n)
        print(
            "  Applying controlled bitwise AND, rotation and reverse-controlled copy gates to D registries..."
        )
        _ddall = [self.ddinit, *self.ddi]
        for i in range(len(self.ddi)):
            # for each "block" of 3n/2 ancillae, the first n are used for the conjunction gate and the last n/2 ones for the rotation gates
            # this process of subdividing ancillae register is done exactly log(n) times
            first_anc_conj = i * 3 * int(self.n / 2)
            first_anc_rot = first_anc_conj + self.n

            ctrl = FSMGateControl(self.di[i], False)
            rctrl = FSMGateControl(self.di[i], True)
            AND = FSMGate(
                bitwise_cand,
                [
                    self.di[i],
                    self.li[i],
                    _ddall[i],
                    AncillaRegister(
                        name="anc_cj",
                        bits=self.ancillae[first_anc_conj : first_anc_conj + self.n],
                    ),
                    _ddall[i + 1],
                ],
            )
            ROT = FSMGate(
                rot,
                [
                    QuantumRegister(name=f"D{i}", bits=_ddall[i + 1][1:]),
                    AncillaRegister(
                        name="anc_rot",
                        bits=self.ancillae[
                            first_anc_rot : first_anc_rot + int(self.n / 2)
                        ],
                    ),
                ],
                [ctrl],
                {"k": 2**i},
            )
            CRC = FSMGate(
                rccopy,
                [_ddall[i], _ddall[i + 1]],
                [rctrl],
            )
            self.apply(AND, ROT, CRC)
        print(f"  Applying disjunction to D{len(self.ddi)-1}...")
        OR = FSMGate(unary_or, [self.ddi[len(self.ddi) - 1], self.out])
        self.apply(OR)
        self._qc.draw(filename="FSM", fold=-1, output="mpl", initial_state=True)
        print(f"Circuit building successful. Qubits: {len(self._qc.qubits)}")
        self._ready = True
        return self

    def apply(self, *gates: FSMGate):
        for gate in gates:
            gate_qubits = []
            for reg in gate.regs:
                if isinstance(reg, Qubit):
                    gate_qubits.append(reg)
                else:
                    gate_qubits.extend(reg)
            if gate.controls:
                ctrl_qubits = [ctrl.qubit for ctrl in gate.controls]
                # duplicated_qubits = (set(*gate_qubits) + set(*ctrl_qubits)) - (set(*gate_qubits) - set(*ctrl_qubits))
                # if len(duplicated_qubits) != len(gate_qubits) + len(ctrl_qubits):
                #     qubits = self._remove_dup_qubits([*ctrl_qubits, *gate_qubits])
                self._qc = self._qc.compose(
                    gate.op(*gate.regs, **gate.params).control(
                        len(gate.controls),
                        ctrl_state="".join(
                            [("0" if ctrl.reverse else "1") for ctrl in gate.controls]
                        ),
                    ),
                    [*ctrl_qubits, *gate_qubits],
                )
            else:
                self._qc = self._qc.compose(
                    gate.op(*gate.regs, **gate.params), gate_qubits
                )
        return self

    def revert(self):
        self._qc = QuantumCircuit(list(self._fsm.regs.values()))

    def execute(self, token: str, iterations=1):
        if not self.ready:
            raise RuntimeError(
                "The circuit was not initialized yet. Use instance.build() before executing to initialize the algorithm circuit"
            )
        self._qc.measure(self.out, self._fsm.cregs["found"])
        self._qc.measure(self.ddi[len(self.ddi) - 1], self._fsm.cregs["begins"])

        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Options

        print("token:", token)
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            instance="ibm-q/open/main",
            token="148653106263ee232f0b7d905569ab477aa935e3bf0e178547dbcecc5df67e60d00d61bad9207789c66c8a0c7df5ab906429dec717f3541bb04f2d27c544ab97",
            overwrite=True,
            # token=token,
        )
        service = QiskitRuntimeService()
        backend = service.backend("ibm_kyoto")

        # options = Options()
        # options.resilience_level = 1
        # options.optimization_level = 2

        job = Sampler(backend).run(self._qc, shots=iterations)
        # print(f"Running job {job.job_id()} on IBM Quantum Platform...")
        result = job.result()
        print(result.quasi_dists[0].binary_probabilities())
        self._measured = True