from math import ceil, log
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from .gates import (
    fanout,
    match,
    rccopy,
    reverse,
    extend,
    bitwise_cand,
    arccopy,
    rot,
    unary_or,
)

n = 8  # length of input strings
nd = 4  # length of desired common substring

# c = QuantumRegister(1, name="c")
# input strings
x = QuantumRegister(n, name="x")
y = QuantumRegister(n, name="y")

# bit representation of nd
d = QuantumRegister(len((bin(nd))[2:]), name="d")

# lambda bitvectors
li = [QuantumRegister(n, name=f"\lambda{i}") for i in range(ceil(log(n)))]

# D registers
dd = [QuantumRegister(n + 1, name=f"D{i-1}") for i in range(ceil(log(n)) + 1)]
out = QuantumRegister(1, name="out")
anc = AncillaRegister(n, name="a")


qc = QuantumCircuit(d, x, y, *li, *dd, out)

# reverse d bitstring
qc = qc.compose(reverse(d), d)
# negate D-1 registry
qc.x([*dd[0]])

# bitvector 0
qc = qc.compose(match(x, y, li[0]), [*x, *y, *li[0]])

# bitvectors 1 to ceil(log(n))
for i in range(len(li) - 1):
    qc = qc.compose(extend(li[i], li[i + 1], i + 1), [*li[i], *li[i + 1]])

# repeat for each Di register, with i = 1...log(n)
for i, ddreg in enumerate(dd[1:]):
    qc = (
        qc.compose(
            bitwise_cand(d[i], li[i], ddreg, dd[i + 1]), [d[i], li[i], ddreg, dd[i + 1]]
        )
        .compose(rot(dd[i + 1], d, 2**i).control(1), [d[i], ddreg])
        .compose(rccopy(d[i], ddreg, dd[i + 1]), [d[i], ddreg, dd[i + 1]])
    )

qc = qc.compose(unary_or(dd[ceil(log(n))], out), [dd[ceil(log(n))], out])
qc.draw(filename="qc", fold=n, output="mpl", vertical_compression="high")
