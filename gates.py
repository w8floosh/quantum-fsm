# The operators we use in the construction of our circuit are as follows:
# – the circular shift operator (ROT)
# – the matching operator (M) for the initialization of the function λi
# – the extension operator (EXTi) of the function λi
# – the register reversal operator (+)
# – the controlled bitwise conjunction operator (∧) and
# – the register disjunction operator (∨)
# – the copy operator with reverse ctrl (C)

from numpy import ceil, floor, log2
from qiskit.circuit import QuantumCircuit, QuantumRegister, QuantumRegister, Qubit


def fanout(src: Qubit, x: QuantumRegister):
    """Fanouts `src` qubit to each qubit of `x`.

    Args:
        - `src` (Qubit): copy source (this is often a control bit)
        - `x` (QuantumRegister): copy destination

    Raises:
        - `NotImplementedError`: if `src` specifies more than one `Qubit` (i.e. it is not a `Qubit` instance)
    """
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
    """Checks which qubits match between `x` and `y` and puts the result into `result` register.

    In FSM algorithm, it initializes λ0 bitvector.

    Args:
        - `x`, `y`: (`QuantumRegister`): input registers
        - `result`: (`QuantumRegister`): output register

    Raises:
        - `ValueError` if registers sizes are not equal
    """
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
    """Extends `bitvec` λ bitvector into `result` register to make it a λ bitvector of order `i`.

    It is used to extend λ`i-1` into λ`i`, where λ`i` is a bitvector whose `j`-th bit is set to 1 if substrings of length `2**i` starting from position `j` are equal (i.e. `x[j : j + 2**i - 1] == y[j : j + 2**i - 1]` ) .

    Args:
        - `bitvec` (QuantumRegister): input bitvector
        - `result` (QuantumRegister): output bitvector
        - `i` (int): order of extension
    """
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
    """Reverses the qubit states of `x`.

    Args:
        - `x` (QuantumRegister): register to reverse"""
    qc = QuantumCircuit(x)
    for i in range(floor(x.size / 2).astype(int)):
        qc.swap(x[i], x[x.size - 1 - i])
    qc.draw(filename="reverse", output="mpl")
    return qc.to_gate(label="REV")


def bitwise_cand_anc(
    ctrl: Qubit,
    x: QuantumRegister,
    y: QuantumRegister,
    anc: QuantumRegister,
    result: QuantumRegister,
):
    """Computes bitwise AND between `x` and `y` if `ctrl` state is set to 1.

    Fanouts `ctrl` qubit into `anc` ancillae register to achieve parallelism when applying

    Args:
        - `ctrl` (Qubit): control qubit, used both for gate conditional application and in fanout gate
        - `x`, `y` (QuantumRegister): input registers
        - `anc` (QuantumRegister): ancillae register to copy `ctrl` state to
        - `result` (QuantumRegister): output register
    """
    if anc.size != x.size or anc.size != y.size - 1:
        raise ValueError("Ancillae and input register sizes are inconsistent")

    qc = QuantumCircuit([ctrl], anc, x, y, result)
    qc = qc.compose(fanout(ctrl, anc), [ctrl, *anc])
    for i in range(x.size):
        # qc = qc.compose(C3XGate(), [anc[i], x[i], y[i], result[i]])
        qc.mcx([anc[i], x[i], y[i]], result[i])  # problematic?
    qc.draw(filename="bitwise_cand", output="mpl")
    return qc.to_gate(label="CAND")

def bitwise_and(x: QuantumRegister, y: QuantumRegister, result: QuantumRegister):
    """Computes bitwise AND between `x` and `y` if `ctrl` state is set to 1.

    Args:
        - `x`, `y` (QuantumRegister): input registers
        - `result` (QuantumRegister): output register
    """
    qc = QuantumCircuit(x, y, result)
    for i in range(x.size):
        # qc = qc.compose(C3XGate(), [anc[i], x[i], y[i], result[i]])
        qc.mcx([x[i], y[i]], result[i])  # problematic?
    qc.draw(filename="bitwise_and", output="mpl")
    return qc.to_gate(label="AND")


def unary_or(x: QuantumRegister, r: Qubit):
    """Computes bitwise unary OR on `x` and puts the result in `r`.

    Args:
        - `x` (QuantumRegister): input register
        - `r` (Qubit): output qubit"""
    qc = QuantumCircuit(x, r)
    for bit in x:
        qc.x(bit)
    qc.mcx([*x], r)
    # qc = qc.compose(fanout(x[0], QuantumRegister(bits=x[1:])), [*x, r])
    for bit in x:
        qc.x(bit)
    qc.x(r)
    qc.draw(filename="unary_or", output="mpl")
    return qc.to_gate(label="OR")


def copy(x: QuantumRegister, result: QuantumRegister):
    """Copies `x` qubits in `result`.

    Args:
        - `x` (QuantumRegister): input register
        - `result` (QuantumRegister): output register"""
    qc = QuantumCircuit(x, result)
    for i in range(result.size):
        qc.cx(x[i], result[i])
    qc.draw(filename="rcopy", output="mpl")
    return qc.to_gate(label="CRC")


# def acopy(
#     ctrl: QuantumRegister,
#     x: QuantumRegister,
#     anc: QuantumRegister,
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


# def rot(x: QuantumRegister, anc: QuantumRegister, k: int = 1):
#     """Cyclically rotates the input register of k positions, with k power of 2,
#     only using swap operations, each controlled by an ancillae qubit leveraged to achieve parallelism.

#     Args:
#         - `x` Register to rotate
#         - `anc` Register of ancillae qubits
#         - `k` Number of positions to rotate

#     Raises:
#         - `ValueError` if `anc` size is not exactly `(x.size * log2(x.size)) / 2`

#     Complexity:
#         - volume: `nlog2^3(n)/2` i.e. `O(nlog2^3(n))`
#         - depth: `Θ(log2^3(n))`
#     """
#     if 2 ** (log2(k).astype(int)) != k:
#         raise NotImplementedError(
#             "Cyclic rotation is implemented for powers of two only"
#         )
#     if anc.size != int(x.size / 2):
#         raise ValueError(
#             "Ancillae register must be half the size of the input register x"
#         )

#     qc = QuantumCircuit(anc, x)
#     next_anc = 0
#     for i in range(1, ceil(log2(x.size)).astype(int) - ceil(log2(k)).astype(int) + 1):
#         for j in range(int(x.size / (2**i * k))):
#             for q in range(j * k * 2**i, k * (j * 2**i + 1)):
#                 qc.cswap(anc[next_anc % anc.size], x[q], x[q + (2 ** (i - 1)) * k])
#                 next_anc += 1

#                 # or try the following:

#                 # b0 = anc[next_anc % anc.size]
#                 # b1 = x[q]
#                 # b2 = x[q + (2 ** (i - 1)) * k]
#                 # qc.ccx(b0, b1, b2)
#                 # qc.ccx(b0, b2, b1)
#                 # qc.ccx(b0, b1, b2)

#     qc.draw(filename=f"rot{k}", output="mpl")
#     return qc.to_gate(label=f"ROT{k}")

def rot(x: QuantumRegister, k: int = 1):
    """Cyclically rotates the input register of k positions using the reflection method made with swap gates only.
    Args:
        - `x` Register to rotate
        - `k` Number of positions to rotate

    Complexity:
        - volume: `O(n)`
        - depth: `Θ(1)`
    """
    qc = QuantumCircuit(x)
    n = x.size
    for i in range(1, ceil(n/2).astype(int)):
        qc.swap(x[i], x[n-i])
    for j in range(1, ceil(n/2).astype(int)):
        qb1 = (ceil(k/2).astype(int)-j)%n
        qb2 = (floor(k/2).astype(int)+j)%n
        qc.swap(x[qb1], x[qb2])

    qc.draw(filename=f"rot{k}", output="mpl")
    return qc.to_gate(label=f"ROT{k}")
