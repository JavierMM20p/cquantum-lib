#include <immintrin.h>
#include <stdio.h>
#include "state_vector.h"
#include "qgates.h"
#include "circuits.h"
#include <stdlib.h>
#include <time.h>

void maxQubitTest(){
    const int QBIT_N = 28;
    StateVector s = init_state_vector(QBIT_N);
    
    X_gate(&s,1);

    char outst[512];
    outst[0] = '\0';

    state_vector_to_string(&s);

    printf("%s", outst);
}

void QFT3qubitTest(){
    const int QBIT_N = 3;
    StateVector s = init_state_vector(QBIT_N);
    
    X_gate(&s,0);
    X_gate(&s,2);
    general_QFT(&s);

    char outst[512];
    outst[0] = '\0';

    state_vector_to_string(&s);

    printf("%s", outst);
}

void speedTest10qubit(){
    const int QBIT_N = 20;
    StateVector s = init_state_vector(QBIT_N);
    
    for(int i=0; i<100; i++){
        for(int j=0; j<QBIT_N; j++){
            H_gate(&s,j);
        }
    }

    char outst[512];
    outst[0] = '\0';

    state_vector_to_string(&s);

    printf("%s", outst);
}