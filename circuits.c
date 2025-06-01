#include "state_vector.h"
#include "qgates.h"
#include <math.h>

void QFT_swap_end(StateVector *state_vect, int size){
    for(int i=0; i<size/2; i++){
        SWAP_gate(state_vect, i, size-i-1);
    }
}

void QFT_rot(StateVector *state_vect, int size){
    if(size == 0){
        return;
    }
    size -= 1;
    H_gate(state_vect, size);
    for(int i=0; i<size; i++){
        C_Phase_gate(state_vect, i, size, (3.14159/pow(2,(size-i))));
    }
    QFT_rot(state_vect,size);
}

void general_QFT(StateVector *state_vect){
    int size = state_vect->n_qubits;
    QFT_rot(state_vect, size);
    QFT_swap_end(state_vect, size);
}