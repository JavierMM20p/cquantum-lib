
# CQuantum-lib

Simulation of quantum computing algorithms in architectures with vector extensions



## State vector data structure

```c
typedef struct {
    __m512 *state_vec;
    int n_qubits;
    int n_vectors;
} StateVector;
```
This structure represents the quantum state vector of a system. The state_vec is an array of 512-bit AVX-512 vector registers, each holding multiple complex amplitudes. n_qubits indicates the number of qubits in the system, while n_vectors is the number of AVX-512 vector blocks required to represent the full state.


## State vector initialization

```c
StateVector init_state_vector(int qubits);
```
This function initializes a quantum state vector for a given number of qubits. It allocates memory and sets the state to the initial |0⟩ state (i.e., the first amplitude is 1, all others are 0). The number of vectors required is calculated based on the number of qubits.

## Print state vector

```c
void state_vector_to_string(StateVector *state_vect);
```
This function prints the quantum state vector in its raw complex amplitude form. It outputs each amplitude as a complex number (real and imaginary parts), making it useful for debugging or understanding the current state of the system.

```c
void state_vector_probability_to_string(StateVector *state_vect);
```
This function prints the probability of each basis state in the quantum system. It computes the squared magnitude (|amplitude|²) of each complex number in the state vector, showing the probability of measuring the system in each state.

```c
void state_vector_polar_to_string(StateVector *state_vect);
```
This function prints each state vector entry in polar form (magnitude and phase angle). It is useful for visualizing how phase affects the quantum state, which is essential in quantum algorithms relying on interference.

## Free state vector

```c
void free_state_vector(StateVector *state_vect);
```
This function releases the dynamically allocated memory associated with a StateVector. It is crucial for avoiding memory leaks, especially in simulations involving multiple or large quantum systems.

## Single qubit gates

```c
void X_gate(StateVector *state_vect, int qubit);
```
Applies the Pauli-X gate (quantum NOT gate) to the specified qubit. This gate flips the state of the qubit from |0⟩ to |1⟩ and vice versa.

```c
void Y_gate(StateVector *state_vect, int qubit);
```
Applies the Pauli-Y gate to the specified qubit. This gate performs a bit-flip and phase-flip, rotating the qubit around the Y-axis of the Bloch sphere.

```c
void Z_gate(StateVector *state_vect, int qubit);
```
Applies the Pauli-Z gate to the specified qubit. This gate introduces a phase flip (sign inversion) to the |1⟩ component of the qubit.

```c
void H_gate(StateVector *state_vect, int qubit);
```
Applies the Hadamard gate to the specified qubit. This gate creates superposition by transforming |0⟩ into (|0⟩ + |1⟩)/√2 and |1⟩ into (|0⟩ - |1⟩)/√2.

```c
void S_gate(StateVector *state_vect, int qubit);
```
Applies the S gate (phase gate) to the specified qubit. This gate adds a π/2 phase shift to the |1⟩ state, useful in quantum phase operations.

```c
void T_gate(StateVector *state_vect, int qubit);
```
Applies the T gate (π/4 phase gate) to the specified qubit. It introduces a π/4 phase to the |1⟩ state, often used in quantum circuits requiring finer phase control.


## Double qubit gates

```c
void CNOT_gate(StateVector *state_vect, int control_qubit, int qubit);
```
Applies a Controlled-NOT gate, flipping the target qubit (qubit) if the control qubit (control_qubit) is in the |1⟩ state. This is a key entangling gate in quantum circuits.

```c
void SWAP_gate(StateVector *state_vect, int control_qubit, int qubit);
```
Swaps the quantum states of two qubits. This gate is useful for reordering qubits or preparing specific entangled states.

```c
void C_Phase_gate(StateVector *state_vect, int control_qubit, int qubit, float phi);
```
Applies a Controlled Phase gate with phase angle phi. It applies a phase shift of e^(i*phi) to the |11⟩ state of the control and target qubits, used for implementing quantum phase and interference logic.

## Measurement

```c
int Z_measure(StateVector *state_vect, int qubit);
```
Performs a projective measurement of the specified qubit in the computational (Z) basis. It collapses the qubit's state to |0⟩ or |1⟩ and returns the measured outcome. The state vector is updated to reflect this measurement collapse.

## QFT
```c
void general_QFT(StateVector *state_vect);
```
Applies the Quantum Fourier Transform (QFT) to the entire quantum state vector. QFT transforms the basis states into a superposition weighted by complex exponential coefficients, effectively mapping a quantum state from the time domain to the frequency domain. This operation is fundamental in many quantum algorithms, such as Shor’s algorithm for factoring and quantum phase estimation. The general_QFT function applies the standard QFT circuit consisting of Hadamard and controlled phase shift gates across all qubits in the system.
