# from math import ceil, log
# from typing import List
# import numpy as np
# from .tools import (
#     calc_bitvec_cell,
#     add_padding,
#     bin_prefix_sum,
#     get_positive_idxs,
#     calc_bitvectors,
#     get_reverse_bitstr,
# )


# class FSM:
#     def __init__(self, d: int):
#         """Initializes a fixed substring matching algorithm.

#         Args:
#             d (int): length of the common substring to search
#         """
#         self._d = d
#         self._n = 0
#         self._x = None
#         self._y = None

#     @property
#     def xy(self):
#         """input strings"""
#         if not self._x or not self._y:
#             return None
#         return self._x, self._y

#     @xy.setter
#     def xy(self, x: str, y: str):
#         if not x or not y:
#             raise ValueError("Input strings cannot be empty")
#         if not len(x) == len(y):
#             raise ValueError("Input strings lengths do not match")
#         self._x = add_padding(x, "$")
#         self._y = add_padding(y, "%")

#     @property
#     def d(self):
#         """length of the common substring to search"""
#         return self._d

#     @property
#     def rd(self):
#         """reversed bitstring representation of d"""
#         return get_reverse_bitstr(self.d)

#     @property
#     def nd(self):
#         """number of bits needed to represent d"""
#         return ceil(log(self.d))

#     @property
#     def n(self):
#         """number of bits needed to represent either x or y"""
#         return ceil(log(self.xy[0]))

#     @property
#     def bitvecs(self):
#         return self._bitvecs

#     @bitvecs.setter
#     def bitvecs(self):
#         if not self.xy:
#             raise KeyError("Input strings not set")
#         self._bitvecs = np.array(
#             [
#                 [calc_bitvec_cell(*self.xy, i, j) for j in range(self.n)]
#                 for i in range(self.nd + 1)
#             ],
#             dtype=int,
#         )

#     @property
#     def neg_dvec(self):
#         """D**-1 vector representation"""
#         if not self.n:
#             return np.array([])
#         return np.ones(self.n)

#     def dvec(self, order: int):
#         if order < 0:
#             return self.neg_dvec
#         if 2**order > self.n:
#             return np.zeros(self.n)

#         dvec = self.neg_dvec
#         for i in range(order + 1):
#             dvec = np.roll(np.logical_and(dvec, self.bitvecs[i]), 2**i)
#             dvec[: 2**i] = 0

#         return dvec

#     def match(self, x: str, y: str):
#         self.x = x, self.y = y
#         self.bitvecs = calc_bitvectors(x, y)
#         matches = []
#         for j in range(len(x)):
#             if all(
#                 self.bitvecs[i][j + bin_prefix_sum(self.rd, i - 1)]
#                 for i in get_positive_idxs(self.rd)
#             ):
#                 matches.append(j)
#         return matches


from math import ceil, log
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from typing import List
import numpy as np

from tools import add_padding
from .gates import (
    arccopy,
    bitwise_cand,
    extend,
    fanout,
    match,
    unary_or,
    rccopy,
    reverse,
    roll2m,
    rot,
)


class FSM:
    def __init__(self, x: str, y: str, d: int):
        """Initializes a fixed substring matching algorithm.

        Args:
            x, y (str): Non-empty input strings in binary format, equally sized.
            d (int): Fixed length of the common substring(s) to search.
        """
        if not x or not y:
            raise ValueError("Input strings cannot be empty")
        if len(x) != len(y):
            raise ValueError("Input strings lengths do not match")
        if d < 2:
            raise ValueError("Substring length must be at least 2")
        if not np.can_cast(f"0b{x}", int) or not np.can_cast(f"0b{y}", int):
            raise TypeError("Input strings are not in binary format")

        self._d = d
        self._rd = QuantumRegister((bin(d))[2:], name="d")
        self._n = len(x)

        self._x = x
        self._y = y
        # register of the binary representation of d
        # input registers
        self._rx = QuantumRegister(self._x, name="X")
        self._ry = QuantumRegister(self._y, name="Y")
        # lambda registers
        self._rli: List[QuantumRegister] = [
            QuantumRegister("0" * self._n, name=f"\lambda{i}")
            for i in range(np.ceil(np.log2(self._n)))
        ]
        # D registers
        self._rddneg = QuantumRegister("1" * self._n, name="D-1")
        self._rddi: List[QuantumRegister] = [
            QuantumRegister("1" * (self._n + 1), name=f"D-1"),
            *[
                QuantumRegister("0" * (self._n + 1), name=f"D{i}")
                for i in range(np.ceil(np.log2(self._n)))
            ],
        ]
        # ancillae register
        self._ranc = AncillaRegister("0" * self._rd.size, name="anc")
        # result qubit
        self._rout = QuantumRegister(1, name="out")

    @property
    def regs(self):
        return (
            self._rd,
            self._rx,
            self._ry,
            self._rli,
            self._rddneg,
            self._rddi,
            self._ranc,
            self._rout,
        )

    def instantiate(self):
        return FSMInstance(self)


class FSMInstance:
    def __init__(self, fsm: FSM):
        self._fsm = fsm
        self._qc = QuantumCircuit(fsm.regs)

    @property
    def xy(self):
        """Input registers"""
        return self._fsm._rx, self._fsm._ry

    @property
    def n(self):
        """Length of input strings"""
        return self._fsm._rx.size

    @property
    def nd(self):
        """Fixed length of the common substring to search"""
        return self._fsm._d

    @property
    def ndbits(self):
        """Length of binary representation of d"""
        return self._fsm._rd.size

    @property
    def d(self):
        """Reversed binary representation of d"""
        return self._fsm._rd

    @property
    def qc(self):
        """Quantum circuit instance"""
        return self._qc

    @property
    def rli(self):
        """λ bitvectors"""
        return self._fsm._rli

    @property
    def ddreg_init(self):
        """D-1 bitvector used in the initialization phase"""
        return self._fsm._rddneg

    @property
    def ddregs(self):
        """Di bitvectors"""
        return self._fsm._rddi

    def build(self):
        self.apply(reverse, self.d).apply((match, (*self.xy, self.rli[0]), None)),

        for i in range(len(self.rli)):
            self.apply((extend, (self.rli[i], self.rli[i + 1])), {"i": i + 1})

        # repeat for each Di register, with i = 1...log(n)
        for i, ddreg in enumerate(self.ddregs):
            self.apply(
                (
                    bitwise_cand,
                    (self.d[i], self.rli[i], ddreg, self.ddregs[i + 1]),
                    None,
                )
            )
            self.apply(
                (rot, (self.ddregs[i + 1], self.d), {"i": 2**i}), controls=[self.d[i]]
            )
            self.apply(())

            # qc = (
            #     qc.compose(rot(dd[i + 1], d, 2**i).control(1), [d[i], ddreg])
            qc.compose(rccopy(d[i], ddreg, dd[i + 1]), [d[i], ddreg, dd[i + 1]])
            # )

        qc = qc.compose(unary_or(dd[ceil(log(n))], out), [dd[ceil(log(n))], out])
        qc.draw(filename="qc", fold=n, output="mpl", vertical_compression="high")

    def apply(self, *gates, controls=[]) -> QuantumCircuit:
        for gate in gates:
            op, regs, params = gate
            unpacked_regs = []
            for reg in regs:
                unpacked_regs.append(*reg)
            if controls:
                self._qc = self._qc.compose(
                    op(*regs, **params).control(len(controls)),
                    [*controls, *unpacked_regs],
                )
            else:
                self._qc = self._qc.compose(op(*regs, **params), unpacked_regs)
        return self._qc
