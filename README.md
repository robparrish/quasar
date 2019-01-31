# Description

Quasar: an ultralight python-2.7/python-3.X quantum  circuit simulator package.

# Sample Use

Build a *N*=4 GHZ circuit:
```python
import quasar
circuit = quasar.Circuit(N=4)
circuit.add_gate(T=0, key=0, gate=quasar.Gate.H)
circuit.add_gate(T=1, key=(0,1), gate=quasar.Gate.CNOT)
circuit.add_gate(T=2, key=(1,2), gate=quasar.Gate.CNOT)
circuit.add_gate(T=3, key=(2,3), gate=quasar.Gate.CNOT)
print(circuit)
```
```text
T   : |0|1|2|3|
               
|0> : -H-@-----
         |     
|1> : ---X-@---
           |   
|2> : -----X-@-
             | 
|3> : -------X-

T   : |0|1|2|3|
```
Simulate the evolution of the state vector through the circuit (aside - see `simulate_steps` to see the details of the state vector through time):
```python
wfn = circuit.simulate()
print(wfn)
```
```text
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```
Computing 1-qubit Pauli expectation values <I_A>, <X_A>, <Y_A>, and <Z_A> indicates that the individual qubits are equally likely to be observed in +Z or -Z: 
```python
PA = quasar.Circuit.compute_pauli_1(wfn=wfn, A=0)
print(PA)
```
```text
[1. 0. 0. 0.]
```
But, computing 2-qubit expectation values like <Z_A * Z_B> (the lower right entry below) indicates that observations between pairs of qubits are perfectly positively correlated:
```python
PAB = quasar.Circuit.compute_pauli_2(wfn=wfn, A=0, B=1)
print(PAB)
```
```text
[[1. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 1.]]
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

# Acknowledgements

Quasar was built as a weekend hobby project while I was getting started with QIS jointly with QC Ware Corp. and SLAC National Accelerator Laboratory. I am grateful for many useful discussions with Peter McMahon, Fabio Sanchez, Juan Ignacio Adame, Ed Hohenstein, and Todd Martinez on this topic.
