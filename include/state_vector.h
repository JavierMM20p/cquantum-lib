#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <immintrin.h>

typedef struct {
    __m512 *state_vec;
    int n_qubits;
    int n_vectors;
} StateVector;

StateVector init_state_vector(int qubits);
void state_vector_to_string(StateVector *state_vect);
void state_vector_probability_to_string(StateVector *state_vect);
void free_state_vector(StateVector *state_vect);
void state_vector_polar_to_string(StateVector *state_vect);

#endif