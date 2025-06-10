
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
## Double qubit gates
## Measurement
