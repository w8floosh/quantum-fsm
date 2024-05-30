# Quantum Fixed Substring Matching
 Quantum implementation of algorithms to solve fixed-length-substring matching problems using Qiskit Runtime to work with real quantum machines. 

 This project aims to implement what was discussed in the paper "Quantum Circuits for Fixed Substring Matching Problems" by Domenico Cantone, Simone Faro, Arianna Pavone and Caterina Viola, freely available at <https://arxiv.org/abs/2308.11758v1>.

## Requirements and dependencies
- Python 3.11 or greater (not tested in older versions)
* Graphviz
+ `matplotlib`
- `numpy`
* `qiskit[visualization]`
+ `qiskit_ibm_runtime`

## How to run
The software can be executed as a module as it is ready to use. 
An IBM Quantum Platform account is required to obtain a IBM API token to use when creating the job on the IBM backends.
The syntax to run the algorithm on IBM machines is the following:
```bash
python -m quantum_fsm [-h] [-l] [-m {FPM,FFP,SFSC}] [-p POSITION] -t TOKEN [-b BACKEND] x y length
```

To see what each argument is used for, please have a look to the command usage help:
```bash
python -m quantum_fsm -h
```

The software will execute the algorithm in `SFSC` mode and on the `ibm_kyoto` backend by default, and print the quasi-probabilities distribution of the possible outcomes when the job is completed. Be aware that the software will block until the request is fulfilled from IBM servers, which may be subject to high queue waiting time, depending on which backend was chosen.

## Capabilities
The algorithm can search for fixed-length common prefixes (`FPM` problem), common substrings starting at a certain position (`FFP` problem) or common substrings starting at any position $j$ (`SFSC` problem) inside bitstrings which length is a power of 2. 
The algorithms work only for fixed-lengths being powers of 2 and only support bitstrings as input.

Due to the number of qubits involved even in easy cases of use, local simulation is not supported. The software can only be executed on real quantum machines or simulators with a high number (127+) of qubits.
Furthermore, as of first release period, the algorithm won't work on IBM simulator backends because of a transpilation issue.

Every gate applied in the building phase (including the whole circuit) will be drawn through matplotlib and Graphviz and saved as an image in the current working directory.

## Performance
Each of the algorithm modes uses exactly $2(n+1)\log_2{}d + (\frac{13}{2})n\log_2{}n = \mathcal{O}(n\log_2{}n)$ qubits, where $d$ is the fixed length of the substring to search and $n$ is the size of any of the input registers (their size must be equal and a power of 2).

The maximum depth is $\mathcal{O}(\log_2^3{}n)$ as stated in the paper, but several tricks have been leveraged in order to reduce the depth and the computation time at the expense of the number of quantum lines, so that the quantum volume is $\mathcal{O}(n\log_2^4{}n)$ even with those improvements.

These algorithms are still under refining, as the current implementation of the full circuit does not give an actual result with acceptable error when running on real quantum machines.

## Legal information

This project was not endorsed by or funded by IBM Corporation and it was realized for academic purposes.
