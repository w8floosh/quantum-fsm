from .entities import FSMMode
from .fsm import FSM

x = "10011100"  # length of input strings
y = "00011000"
nd = 4  # length of desired common substring
mode = FSMMode.SFSC

print(
    f"Instantiating FSM with mode {mode.value} and inputs x = {x}, y = {y}, length = {nd}"
)
fsm = FSM(x, y, nd, mode).instantiate()
print(f"Building {mode.value} algorithm circuit...")
fsm = fsm.build()
print("Executing...")
fsm.execute()
