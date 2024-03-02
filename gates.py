# The operators we use in the construction of our circuit are as follows:
# – the circular shift operator (ROT)
# – the matching operator (M) for the initialization of the function λi
# – the extension operator (EXTi) of the function λi
# – the register reversal operator (+)
# – the controlled bitwise conjunction operator (∧) and
# – the register disjunction operator (∨)
# – the copy operator with reverse ctrl (C)

from numpy import ceil, floor, log2, rint
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    AncillaRegister,
    Qubit,
)
from qiskit.circuit.quantumcircuit import QubitSpecifier


def fanout(src: Qubit, x: QuantumRegister):
    """fanouts single qubit to each qubit of a register"""
    if not isinstance(src, Qubit):
        raise NotImplementedError(
            "Fanout operator with more than one source is not supported"
        )
    qc = QuantumCircuit([src], x)
    qc.cx(src, x[0])
    n = x.size

    for j in (2**x for x in range(ceil(log2(n)).astype(int))):
        # print(f"j={j}")
        for i in range(j):
            # print(f"(x[{i}], x[{j+i}])")
            qc.cx(x[i], x[j + i])

    qc.draw(filename="fanout", output="mpl")
    return qc.to_gate(label="FAN")


def match(x: QuantumRegister, y: QuantumRegister, result: QuantumRegister):
    """initializes λ0 bitvector"""
    if x.size != y.size:
        raise ValueError("Registers size don't match")
    #    result = QuantumRegister(x.size, name="λ0")
    qc = QuantumCircuit(x, y, result)

    # n parallel Toffoli gates
    for i in range(result.size):
        qc.ccx(x[i], y[i], result[i])

    # 2n parallel X gates, one for each input qubit
    for i in range(x.size):
        qc.x(x[i])
        qc.x(y[i])

    # n parallel Toffoli gates
    for i in range(result.size):
        qc.ccx(x[i], y[i], result[i])

    # 2n parallel X gates, one for each input qubit
    for i in range(x.size):
        qc.x(x[i])
        qc.x(y[i])
    qc.draw(filename="match", output="mpl")
    return qc.to_gate(label="M(x,y)")


def extend(bitvec: QuantumRegister, result: QuantumRegister, i: int = 1):
    """extends λi-1 bitvector to λi"""
    #    result = QuantumRegister(bitvec.size, name=f"λ{i}")
    qc = QuantumCircuit(bitvec, result)
    if i != 1:
        pos_list = range(result.size)
    else:
        # first apply Toffoli for even positions, then apply to odd positions
        pos_list = [
            *[idx for idx in range(result.size) if not idx % 2],
            *[idx for idx in range(result.size) if idx % 2],
        ]
    for j in pos_list:
        if j + 2 ** (i - 1) >= result.size:
            continue
        qc.ccx(bitvec[j], bitvec[j + 2 ** (i - 1)], result[j])

    qc.draw(filename=f"extend{i}", output="mpl")
    return qc.to_gate(label=f"EXT{i}")


def reverse(x: QuantumRegister):
    """reverses the given register"""
    qc = QuantumCircuit(x)
    for i in range(floor(x.size / 2).astype(int)):
        qc.swap(x[i], x[x.size - 1 - i])
    qc.draw(filename="reverse", output="mpl")
    return qc.to_gate(label="REV")


def bitwise_cand(
    ctrl: QuantumRegister,
    x: QuantumRegister,
    y: QuantumRegister,
    anc: AncillaRegister,
    result: QuantumRegister,
):
    """controlled AND between bits of two registers"""
    if anc.size != x.size or anc.size != y.size - 1:
        raise ValueError("Ancillae and input register sizes are inconsistent")
    qc = QuantumCircuit([ctrl], anc, x, y, result)
    qc = qc.compose(fanout(ctrl, anc), [ctrl, *anc])
    for i in range(x.size):
        qc.mcx([anc[i], x[i], y[i]], result[i + 1])
    qc.draw(filename="bitwise_cand", output="mpl")
    return qc.to_gate(label="CAND")


def unary_or(x: QuantumRegister, r: QuantumRegister):
    """puts disjunction result of x in the r qubit"""
    qc = QuantumCircuit(x, r)
    for bit in x:
        qc.x(bit)
    qc.mcx([*x], r[0])
    for bit in x:
        qc.x(bit)
    qc.x(r[0])
    qc.draw(filename="unary_or", output="mpl")
    return qc.to_gate(label="OR")


def rccopy(x: QuantumRegister, result: QuantumRegister):
    """copies x qubits in result register with reversal control"""
    qc = QuantumCircuit(x, result)
    for i in range(result.size):
        qc.cx(x[i], result[i])
    qc.draw(filename="rcopy", output="mpl")
    return qc.to_gate(label="CRC")


# def arccopy(
#     ctrl: QuantumRegister,
#     x: QuantumRegister,
#     anc: AncillaRegister,
#     result: QuantumRegister,
# ):
#     """copies x qubits in result register with reversal control using ancillae qubits"""
#     qc = QuantumCircuit(ctrl, x, result)
#     qc.x(ctrl[0])
#     qc.cx(ctrl[0], [bit for bit in anc])
#     for i in range(result.size):
#         qc.cx(anc[i], result[i])
#     qc.x([ctrl[0], *[bit for bit in anc]])
#     qc.draw(filename="acopy", output="mpl")
#     return qc.to_gate(label="ACRC")


def rot(x: QuantumRegister, anc: AncillaRegister, k: int = 1):
    """Cyclically rotates the input register of k positions, with k power of 2,
    only using swap operations, each controlled by an ancillae qubit leveraged to achieve parallelism.

    Args:
        - `x` Register to rotate
        - `k` Number of positions to rotate
        - `anc` Register of ancillae qubits

    Raises:
        - `ValueError` if `anc` size is not exactly `(x.size * log2(x.size)) / 2`

    Complexity:
        - volume: `nlog2^3(n)/2` i.e. `O(nlog2^3(n))`
        - depth: `Θ(log2^3(n))`
    """
    if 2 ** (log2(k).astype(int)) != k:
        raise NotImplementedError(
            "Cyclic rotation is implemented for powers of two only"
        )
    if anc.size != int(x.size / 2):
        raise ValueError(
            "Ancillae register must be half the size of the input register x"
        )

    qc = QuantumCircuit(anc, x)
    next_anc = 0
    for i in range(1, ceil(log2(x.size)).astype(int) - ceil(log2(k)).astype(int) + 1):
        for j in range(int(x.size / (2**i * k))):
            for q in range(j * k * 2**i, k * (j * 2**i + 1)):
                qc.cswap(anc[next_anc % anc.size], x[q], x[q + (2 ** (i - 1)) * k])
                next_anc += 1

    qc.draw(filename=f"rot{k}", output="mpl")
    return qc.to_gate(label=f"ROT{k}")
