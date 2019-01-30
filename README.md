# Description

Quasar: an ultralight python-2.7/python-3.X quantum  circuit simulator package.

# Simple Use

```python
import quasar
circuit = quasar.Circuit(N=4)
circuit.add_gate(T=0, key=0, gate=quasar.Gate.H)
circuit.add_gate(T=1, key=(0,1), gate=quasar.Gate.CNOT)
circuit.add_gate(T=2, key=(1,2), gate=quasar.Gate.CNOT)
circuit.add_gate(T=3, key=(2,3), gate=quasar.Gate.CNOT)
print(circuit)
```

# Rationale

There are a million quantum circuit simulation packages out there - so why
write another one? Three reasons:

 * As I got started in QIS, I was using `package-X` (an industry standard).
Then a week later I hit `conda update package-X`. And all my circuits
silently broke (ran but produced wrong answers). So I decided to make an
environment which is both hard to break and where I am to blame if it breaks.
 * A bit later on, I went to run an 18-qubit MC-VQE job with a simple cyclical
Ising model (now in both `package-X` and `package-Y`). And burned a nice
MacBook Pro up waiting for the state vector simulation result. So I decided to
make a really simple state vector simulator that relies on some standard
`np.einsum` magic and  circuit compression tricks to get the answer I needed in
less than 1/10th of the time.
 * Most important: I really enjoyed reading Jarrod McClean's great post on
using `numpy` for quantum circuit simulation:
https://jarrodmcclean.com/basic-quantum-circuit-simulation-in-python.  Jarrod
is totally right that `numpy` is all you need to do quantum circuit simulation - with a good `numpy` quantum circuit simulator, you can run deep circuits with
~32 qubits in no time on classical hardware (state vector memory is much more
of a bottleneck). If you need between ~34-40 qubits, you need a DOE
supercomputer, Google TPU pod, or similar for the required memory. And if you
need >48 qubits (being generous), you are SOL. So there is only a very narrow
window where post-`numpy` quantum circuit simulators are useful, and you should
probably instead be thinking about more-clever representations of your
wavefunction/density matrix in that limit anyways. All quasar is is a little
bit of sugar to help you build, simulate, and analyze circuits in `numpy` -
sort of Jarrod's tutorial extended to make life easy for someone who has to do
QIS work every day.

# Contents

* quasar - lightweight quantum simulator module
* test-quasar - unit tests for quasar
* demos - Jupyter notebooks with demos of quasar

# Contact

Rob Parrish - robparrish@gmail.com
