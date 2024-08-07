from numpy import log2
from .fsm import FSM
from argparse import ArgumentParser

parser = ArgumentParser(prog="quantum_fsm")

parser.add_argument("x", help="First binary input string")
parser.add_argument("y", help="Second binary input string")
parser.add_argument("length", help="Length of the common substring to search", type=int)
parser.add_argument(
    "-m",
    "--mode",
    default="SFSC",
    help="Substring matching mode",
    choices=["FPM", "FFP", "SFSC"],
)
parser.add_argument(
    "-p",
    "--position",
    help="Position index from which to search a common substring (required when mode is 'FFP', else ignored)",
    type=int,
)
mxgroup = parser.add_mutually_exclusive_group(required=True)
mxgroup.add_argument("-t", "--token", help="IBM Quantum platform API token")
mxgroup.add_argument(
    "-T", "--tokenpath", help="Path to read IBM Quantum platform API token from"
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="Execute the algorithm locally using QASM simulator with matrix_product_state method",
)
parser.add_argument("-b", "--backend", default="ibm_sherbrooke", help="IBM backend name")

args = parser.parse_args()

if args.mode == "FFP" and not args.position:
    parser.error("specifying a starting position is required in FFP mode. See help")

if len(args.x) != len(args.y):
    parser.error("x and y inputs must have the same length")

if 2 ** (log2(len(args.x)).astype(int)) != len(args.x):
    parser.error("input strings length must be a power of 2")

# if 2 ** (log2(args.length).astype(int)) != args.length:
#     parser.error("length must be a power of 2")

if not args.token and not args.tokenpath:
    parser.error("IBM Quantum Platform API token is required")

_token = args.token or open(args.tokenpath, "r").read().strip()

if len(_token) != 128:
    parser.error(
        "invalid IBM Quantum Platform API token: (must be a 128 hex digit number)"
    )
try:
    int(f"0x{_token}", base=16)
except:
    parser.error("invalid IBM Quantum Platform API token (must be convertible to hex)")

print(
    f"Instantiating FSM with mode {args.mode} and inputs x = {args.x}, y = {args.y}, length = {args.length}"
)

if args.mode == "FFP":
    fsm = FSM(
        args.x, args.y, args.length, args.mode, args.backend, starting_pos=args.position
    ).instantiate()
else:
    fsm = FSM(args.x, args.y, args.length, args.mode, args.backend).instantiate()

print(f"Building {args.mode} algorithm circuit...")
fsm = fsm.build()

print(f"Executing {'locally' if args.local else 'remotely'}...")
fsm.execute(str(_token), iterations=10000, local=args.local)
