#ifndef QGATES_H
#define QGATES_H

#include <immintrin.h>
#include "state_vector.h"

void debug_gate(StateVector *state_vect, int qubit);
void debug_gate_d(StateVector *state_vect, int c_qubit, int qubit);

void X_gate(StateVector *state_vect, int qubit);
void Y_gate(StateVector *state_vect, int qubit);
void Z_gate(StateVector *state_vect, int qubit);
void H_gate(StateVector *state_vect, int qubit);
void S_gate(StateVector *state_vect, int qubit);
void T_gate(StateVector *state_vect, int qubit);
void CNOT_gate(StateVector *state_vect, int control_qubit, int qubit);
void SWAP_gate(StateVector *state_vect, int control_qubit, int qubit);
void C_Phase_gate(StateVector *state_vect, int control_qubit, int qubit, float phi);

int Z_measure(StateVector *state_vect, int qubit);

#endif