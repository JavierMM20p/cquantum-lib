#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define QSTATE_SIZE(n) (1 << (n)) // 2^n for n qubits

typedef double complex qcomplex;

// Function to allocate and initialize a quantum state vector
qcomplex* create_quantum_state(int num_qubits) {
    int size = QSTATE_SIZE(num_qubits);
    qcomplex* state = (qcomplex*)calloc(size, sizeof(qcomplex));
    if (!state) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize to |0...0> state
    state[0] = 1.0 + 0.0 * I;
    return state;
}

// Function to apply a single-qubit gate to a quantum state
void apply_single_qubit_gate(qcomplex* state, int num_qubits, int target_qubit,
                             qcomplex gate[2][2]) {
    int size = QSTATE_SIZE(num_qubits);
    qcomplex* temp_state = malloc(sizeof(qcomplex) * size);
    if (!temp_state) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Copy original state
    for (int i = 0; i < size; i++) {
        temp_state[i] = state[i];
    }

    // Loop through all indices
    for (int i = 0; i < size; i++) {
        // Check if target qubit bit is 0
        if (((i >> target_qubit) & 1) == 0) {
            int i0 = i;
            int i1 = i0 | (1 << target_qubit);

            qcomplex s0 = temp_state[i0];
            qcomplex s1 = temp_state[i1];

            state[i0] = gate[0][0] * s0 + gate[0][1] * s1;
            state[i1] = gate[1][0] * s0 + gate[1][1] * s1;
        }
    }

    free(temp_state);
}


// Function to apply a two-qubit gate to a quantum state
void apply_two_qubit_gate(qcomplex* state, int num_qubits,
                          int qubit1, int qubit2,
                          qcomplex gate[4][4]) {
    if (qubit1 == qubit2) {
        fprintf(stderr, "Target qubits must be different\n");
        exit(EXIT_FAILURE);
    }

    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }

    int size = QSTATE_SIZE(num_qubits);

    for (int i = 0; i < size; i++) {
        // Extract bits
        int b1 = (i >> qubit1) & 1;
        int b2 = (i >> qubit2) & 1;

        // Work only on basis states where both bits are 0 to avoid double application
        if (b1 == 0 && b2 == 0) {
            // Create the 4 indices for the 2-qubit subsystem
            int base = i & ~( (1 << qubit1) | (1 << qubit2) );
            int idx[4];
            for (int j = 0; j < 4; ++j) {
                int bit1 = (j >> 1) & 1;
                int bit2 = j & 1;
                idx[j] = base | (bit1 << qubit1) | (bit2 << qubit2);
            }

            // Temporarily store the values to be updated
            qcomplex temp[4];
            for (int j = 0; j < 4; ++j) {
                temp[j] = state[idx[j]];
            }

            // Apply the gate
            for (int j = 0; j < 4; ++j) {
                state[idx[j]] = 0.0 + 0.0 * I;
                for (int k = 0; k < 4; ++k) {
                    state[idx[j]] += gate[j][k] * temp[k];
                }
            }
        }
    }
}


// Function to print a quantum state
void print_quantum_state(qcomplex* state, int num_qubits) {
    int size = QSTATE_SIZE(num_qubits);
    for (int i = 0; i < size; i++) {
        printf("|%0*d⟩: %.4f + %.4fi\n", num_qubits, i,
               creal(state[i]), cimag(state[i]));
    }
}

void apply_controlled_phase(qcomplex* state, int num_qubits, int control, int target, double theta) {
    int size = QSTATE_SIZE(num_qubits);
    for (int i = 0; i < size; ++i) {
        if (((i >> control) & 1) && ((i >> target) & 1)) {
            state[i] *= cexp(I * theta);
        }
    }
}

void apply_qft(qcomplex* state, int num_qubits) {
    qcomplex h_gate[2][2] = {
        { 1.0 / sqrt(2.0) + 0.0 * I,  1.0 / sqrt(2.0) + 0.0 * I },
        { 1.0 / sqrt(2.0) + 0.0 * I, -1.0 / sqrt(2.0) + 0.0 * I }
    };

    for (int i = 0; i < num_qubits; ++i) {
        // Apply Hadamard to qubit i
        apply_single_qubit_gate(state, num_qubits, i, h_gate);

        // Apply controlled phase shifts
        for (int j = 1; i + j < num_qubits; ++j) {
            double theta = 3.14159 / (1 << j);  // π / 2^j
            apply_controlled_phase(state, num_qubits, i + j, i, theta);
        }
    }

    // Swap qubits to reverse order
    for (int i = 0; i < num_qubits / 2; ++i) {
        int j = num_qubits - i - 1;
        for (int k = 0; k < QSTATE_SIZE(num_qubits); ++k) {
            int bi = (k >> i) & 1;
            int bj = (k >> j) & 1;
            if (bi != bj) {
                int swapped = k ^ ((1 << i) | (1 << j));
                if (k < swapped) {
                    qcomplex tmp = state[k];
                    state[k] = state[swapped];
                    state[swapped] = tmp;
                }
            }
        }
    }
}

int main() {

    qcomplex h_gate[2][2] = {
    { 1.0 / sqrt(2.0) + 0.0 * I,  1.0 / sqrt(2.0) + 0.0 * I },
    { 1.0 / sqrt(2.0) + 0.0 * I, -1.0 / sqrt(2.0) + 0.0 * I }
    };

    qcomplex cnot[4][4] = {
    {1.0 + 0.0 * I, 0.0 + 0.0 * I, 0.0 + 0.0 * I, 0.0 + 0.0 * I},
    {0.0 + 0.0 * I, 1.0 + 0.0 * I, 0.0 + 0.0 * I, 0.0 + 0.0 * I},
    {0.0 + 0.0 * I, 0.0 + 0.0 * I, 0.0 + 0.0 * I, 1.0 + 0.0 * I},
    {0.0 + 0.0 * I, 0.0 + 0.0 * I, 1.0 + 0.0 * I, 0.0 + 0.0 * I}
    };

    const int QBIT_N = 28;
    const int TEST_TIMES = 1000;
    
    struct timespec start, end;

    printf("[INFO] Starting tests!!!.\n");

    /**
    // TEST 1 H GATES
    //
    printf("[TEST] Test 1: starting...\n");
    qcomplex* s1= create_quantum_state(QBIT_N);

    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int i = 0; i<TEST_TIMES; i++){
        int j = i % QBIT_N;
            apply_single_qubit_gate(s1, QBIT_N, j, h_gate);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[TEST] Test 1: HGates took %f seconds to execute.\n", time_taken);

    //print_quantum_state(s1, QBIT_N);

    free(s1);
    // TEST 2 C GATES
    //
    printf("[TEST] Test 2: starting...\n");
    qcomplex* s2= create_quantum_state(QBIT_N);

    clock_gettime(CLOCK_MONOTONIC, &start);

    int j = 0;
    int c = 0;
    int app_gat = 0;
    while(app_gat < TEST_TIMES){
        for(j = 0; j<QBIT_N && (app_gat < TEST_TIMES); j++){
        for(c = 0; c<QBIT_N && (app_gat < TEST_TIMES); c++){
            if(c != j){
                    apply_two_qubit_gate(s2, QBIT_N, c, j, cnot);
                    app_gat++;
                }
        }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[TEST] Test 2: CGates took %f seconds to execute.\n", time_taken);

    //print_quantum_state(s2, QBIT_N);

    free(s2);

    printf("[INFO] Finished tests!!!.\n");

    */
    // TEST 3 QFT
    //
    printf("[TEST] Test 3: starting...\n");
    qcomplex* s3= create_quantum_state(QBIT_N);

    clock_gettime(CLOCK_MONOTONIC, &start);

    apply_qft(s3, QBIT_N);

    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[TEST] Test 2: CGates took %f seconds to execute.\n", time_taken);

    //print_quantum_state(s2, QBIT_N);

    free(s3);

    printf("[INFO] Finished tests!!!.\n");

    return 0;
}

