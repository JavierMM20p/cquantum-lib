# cquantum-lib
Simulation of quantum computing algorithms in architectures with vector extensions
---
📁 Header Files

    state_vector.h: Initialization and manipulation of quantum states.

    qgates.h: Quantum gate operations.

    circuits.h: Quantum circuits like QFT.

🧠 Data Structure

typedef struct {
    __m512 *state_vec;
    int n_qubits;
    int n_vectors;
} StateVector;

🧱 State Vector Functions
StateVector init_state_vector(int qubits);

Initializes a quantum state vector to |0⟩.

Example:

StateVector sv = init_state_vector(3);  // 3-qubit system

void state_vector_to_string(StateVector *state_vect);

Prints the real and imaginary components of the state vector.

Example:

state_vector_to_string(&sv);

void state_vector_probability_to_string(StateVector *state_vect);

Prints the measurement probabilities of each basis state.

Example:

state_vector_probability_to_string(&sv);

void state_vector_polar_to_string(StateVector *state_vect);

Prints the magnitude and phase of each amplitude in the state vector.

Example:

state_vector_polar_to_string(&sv);

void free_state_vector(StateVector *state_vect);

Frees allocated memory.

Example:

free_state_vector(&sv);

⚛️ Quantum Gate Functions
void X_gate(StateVector *state_vect, int qubit);

Applies the Pauli-X (NOT) gate.

Example:

X_gate(&sv, 0);

void Y_gate(StateVector *state_vect, int qubit);

Applies the Pauli-Y gate.

Example:

Y_gate(&sv, 1);

void Z_gate(StateVector *state_vect, int qubit);

Applies the Pauli-Z gate.

Example:

Z_gate(&sv, 2);

void H_gate(StateVector *state_vect, int qubit);

Applies the Hadamard gate.

Example:

H_gate(&sv, 0);

void S_gate(StateVector *state_vect, int qubit);

Applies the S phase gate.

Example:

S_gate(&sv, 1);

void T_gate(StateVector *state_vect, int qubit);

Applies the T phase gate.

Example:

T_gate(&sv, 2);

void CNOT_gate(StateVector *state_vect, int control_qubit, int qubit);

Applies a controlled-NOT gate.

Example:

CNOT_gate(&sv, 0, 1);

void SWAP_gate(StateVector *state_vect, int control_qubit, int qubit);

Swaps two qubits.

Example:

SWAP_gate(&sv, 1, 2);

void C_Phase_gate(StateVector *state_vect, int control_qubit, int qubit, float phi);

Applies a controlled phase gate with angle phi.

Example:

C_Phase_gate(&sv, 0, 2, 3.1415); // π

int Z_measure(StateVector *state_vect, int qubit);

Performs a measurement in the Z-basis. Returns 0 or 1.

Example:

int result = Z_measure(&sv, 1);
printf("Measurement result: %d\n", result);

void debug_gate(StateVector *state_vect, int qubit);

Debug/test function for applying a custom gate on a single qubit.

Example:

debug_gate(&sv, 0);

void debug_gate_d(StateVector *state_vect, int c_qubit, int qubit);

Debug/test function for a controlled operation between two qubits.

Example:

debug_gate_d(&sv, 0, 1);

🧮 Circuit Functions
void general_QFT(StateVector *state_vect);

Applies the Quantum Fourier Transform (QFT) on the state vector.

Example:

general_QFT(&sv);

🧪 Example Program

#include "state_vector.h"
#include "qgates.h"
#include "circuits.h"

int main() {
    StateVector sv = init_state_vector(2);

    H_gate(&sv, 0);
    CNOT_gate(&sv, 0, 1);

    state_vector_to_string(&sv);
    state_vector_probability_to_string(&sv);

    free_state_vector(&sv);
    return 0;
}
