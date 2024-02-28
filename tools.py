from math import ceil, log
from typing import List


def pb_decompose(w: str):
    d = len(w)
    _d = get_reverse_bitstr(d)  # reverse bitstring
    S = []
    for i, bit in enumerate(_d):
        if int(bit):
            S.append(i)

    decomposition = []
    for i in S:
        lsum = bin_prefix_sum(_d, i - 1)
        rsum = bin_prefix_sum(_d, i) - 1
        decomposition.append(w[lsum:rsum])

    return decomposition


def bin_prefix_sum(x: str, end: int):
    sum = 0
    for i, bit in enumerate(x):
        if int(bit):
            sum += 2**i
        if i == end:
            break
    return sum


def add_padding(x: str, char):
    # take the smallest value p for which we have n < 2**p and concatenate the text with 2**p − n copies of the special character
    n = len(x)
    if not n % 2:
        return x
    p = ceil(log(len(x)))
    return x.ljust(2**p - n, char)


def get_reverse_bitstr(x: int):
    return (bin(x)[2:])[::-1]  # reverse bitstring


def calc_bitvec_cell(x: str, y: str, i: int, j: int):
    if i:
        return calc_bitvec_cell(x, y, i - 1, j) and calc_bitvec_cell(
            x, y, i - 1, j + 2 ** (i - 1)
        )
    else:
        return 1 if x[j] == y[j] else 0


def get_positive_idxs(x: str):
    S = []
    for i, bit in enumerate(x):
        if int(bit):
            S.append(i)
    return S


# def match_substr_fixed_idx(x: str, y: str, d: int):
#     bitvectors = calc_bitvectors(x, y)
#     _d = get_reverse_bitstr(d)
#     for j in range(len(x)):
#         if all(
#             bitvectors[i][j + bin_prefix_sum(_d, i - 1)] for i in get_positive_idxs(_d)
#         ):
#             return j
#     return None


# def odds(bits: list):
#     return [bit for i, bit in enumerate(bits) if not i % 2]


# def evens(bits: list):
#     return [bit for i, bit in enumerate(bits) if i % 2]
