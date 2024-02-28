# The operators we use in the construction of our circuit are as follows:
# – the circular shift operator (ROT)
# – the matching operator (M) for the initialization of the function λi
# – the extension operator (EXTi) of the function λi
# – the register reversal operator (+)
# – the controlled bitwise conjunction operator (∧) and
# – the register disjunction operator (∨)
# – the copy operator with reverse ctrl (C)

from math import ceil, floor, log
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    AncillaRegister,
)


def fanout(ctrl: QuantumRegister, x: QuantumRegister):
    # if ctrl.size != 1:
    #     raise NotImplementedError(
    #         "Fanout operator with more than 1 qubit is not supported"
    #     )
    qc = QuantumCircuit(ctrl, x)
    qc.cx(ctrl[0], x[0])
    n = x.size

    for j in (2**x for x in range(ceil(log(n)))):
        # print(f"j={j}")
        for i in range(j):
            # print(f"(x[{i}], x[{j+i}])")
            qc.cx(x[i], x[j + i])

    qc.draw(filename="fanout", output="mpl")
    return qc.to_gate(label="FAN")


def match(x: QuantumRegister, y: QuantumRegister, result: QuantumRegister):
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


def extend(bitvec: QuantumRegister, result: QuantumRegister, i: int):
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
    qc = QuantumCircuit(x)
    for i in range(floor(x.size / 2.0)):
        qc.swap(x[i], x[x.size - 1 - i])
    qc.draw(filename="reverse", output="mpl")
    return qc.to_gate(label="REV")


def bitwise_cand(
    ctrl: QuantumRegister,
    x: QuantumRegister,
    y: QuantumRegister,
    result: QuantumRegister,
):
    qc = QuantumCircuit(ctrl, x, y, result)
    for i in range(result.size):
        qc.mcx([ctrl[0], x[i], y[i]], result[i])
    qc.draw(filename="bitwise_cand", output="mpl")
    return qc.to_gate(label="CAND")


def unary_or(x: QuantumRegister, r: QuantumRegister):
    qc = QuantumCircuit(x, r)
    for bit in x:
        qc.x(bit)
    qc.mcx([*x], r[0])
    for bit in x:
        qc.x(bit)
    qc.x(r[0])
    qc.draw(filename="unary_or", output="mpl")
    return qc.to_gate(label="OR")


def rccopy(ctrl: QuantumRegister, x: QuantumRegister, result: QuantumRegister):
    qc = QuantumCircuit(ctrl, x, result)
    qc.x(ctrl[0])
    for i in range(result.size):
        qc.ccx(ctrl[0], x[i], result[i], ctrl_state="01")
    qc.x(ctrl[0])
    qc.draw(filename="rcopy", output="mpl")
    return qc.to_gate(label="CRC")


def arccopy(
    ctrl: QuantumRegister,
    x: QuantumRegister,
    anc: AncillaRegister,
    result: QuantumRegister,
):
    qc = QuantumCircuit(ctrl, x, result)
    qc.x(ctrl[0])
    qc.cx(ctrl[0], [bit for bit in anc])
    for i in range(result.size):
        qc.cx(anc[i], result[i])
    qc.x([ctrl[0], *[bit for bit in anc]])
    qc.draw(filename="acopy", output="mpl")
    return qc.to_gate(label="ACRC")


# def rot(n, k, block_size=1):
#     qc = QuantumCircuit(n, name=f"rot_k={k}")
#     stop = int(np.log2(n)) - int(np.log2(k * block_size)) + 2
#     for i in range(block_size, stop):
#         for j in range(0, int(n / (k * (2**i)))):
#             for x in range(j * k * (2**i), k * ((j * 2**i + 1))):
#                 for offset in range(block_size):
#                     inizio_swap = x + k * offset
#                     fine_swap = x + 2 ** (i - 1) * k + k * offset
#                     qc.swap(inizio_swap, fine_swap)

#     # qkt.draw_circuit(qc)
#     rot_gate = qc.to_gate(label="Rot_" + str(k))
#     return rot_gate


def rot(x: QuantumRegister, k: QuantumRegister, anc: AncillaRegister, i: int):
    if k.size != log(x.size):
        raise NotImplementedError(
            "Cyclic rotation of any number of positions not implemented"
        )
    if log(x.size) != anc.size:
        raise ValueError("Ancillae register must have exactly log(n) qubits")

    qc = QuantumCircuit(x, k, anc)
    # for i, ki in enumerate(k):
    qc = qc.compose(roll2m(x, i).control(1), [k[i], *anc, *x])
    return qc.to_gate(label=f"ROT{k}")


def roll2m(x: QuantumRegister, m: int):
    qc = QuantumCircuit(x)
    k = 2**m
    for i in range(1, log(x.size) - m + 1):
        for j in range(x.size / (2**i * k)):
            for q in range(j * k * 2**i, -1 + k * (j * 2**i + 1)):
                qc.swap(x[q], x[q + k * 2 ** (i - 1)])
    return qc.to_gate(label=f"R{m}")
