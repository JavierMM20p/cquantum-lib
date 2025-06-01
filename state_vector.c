#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>
#include "state_vector.h"
#include <malloc.h>

/** StateVector init_state_vector(qubits)
 *   Initialization for StateVector data structure, all registers to 0.0f except the first one with an 1.0.
 *   It will contain 1 vector if qubits <= 3, or (qubits - 3) vectors for all other cases.
 */
StateVector init_state_vector(int qubits){
    int n_vects = (qubits >= 3) ? (1 << (qubits - 3)) : 1;
    __m512 *avxArray = NULL;
    if (posix_memalign((void**)&avxArray, 64, 2* n_vects * sizeof(__m512)) != 0) {
        printf("Memory allocation failed!\n");
    }

    int reach = 8;
    if(qubits == 1) reach = 2;
    if(qubits == 2) reach = 4; 
    for(int i=0; i < n_vects; i++){
        float values[16];
        for (int j = 0; j < reach; j++){
            values[2*j] = 0.0;
            values[2*j + 1] = 0.0;
        }for (int j = reach; j < 8; j++){
            values[2*j] = 0.0;
            values[2*j + 1] = 0.0;
        }
        if(i==0){values[0] = 1.0;}
        avxArray[i] = _mm512_loadu_ps(values);
    }

    StateVector s;
    s.state_vec = avxArray;
    s.n_qubits = qubits;
    s.n_vectors = n_vects;
    return s;
}

/** void state_vector_to_string(state_vect, buffer)
 *   Print the state vector values, as complex numbers.
 */
void state_vector_to_string(StateVector *state_vect){
    int qubits = state_vect->n_qubits;
    int rows = 1 << qubits;
    int n_vec = state_vect->n_vectors;
    __m512 *avx_v = state_vect->state_vec;

    int reach = 8;
    if(qubits == 1) reach = 2;
    if(qubits == 2) reach = 4; 

    char *tmp_ptr;
    for(int reg = 0; reg < n_vec; reg++){
        float values[16];
        _mm512_store_ps(values, avx_v[reg]);
        for(int i=0; i < reach; i++){
            char buffer[256];
            buffer[0] = '\0';
            tmp_ptr = strcat(buffer, "|");
            int state_index = reg * 8 + i;
            for(int j=0; j < qubits; j++){
                int mask = 1 << (qubits - j -1);
                if(state_index & mask){
                    tmp_ptr = strcat(buffer, "1");
                }else{
                    tmp_ptr = strcat(buffer, "0");
                }
            }
            char tmp_extra[32];
            float real_val = values[2*i];
            float im_val = values[2*i+1];
            snprintf(tmp_extra, sizeof(tmp_extra),"> %f + %fi\n", real_val, im_val);
            strcat(buffer, tmp_extra);
            printf("%s",buffer);
            //if (strlen(buffer) >= buffer_size - 256) {
            //    fprintf(stderr, "Buffer size exceeded!\n");
            //    return;
            //}
        }
    }
}

/** void state_vector_probability_to_string(state_vect, buffer)
 *   Print the state vector values, as the probability of resulting in that state.
 */
void state_vector_probability_to_string(StateVector *state_vect){
        int qubits = state_vect->n_qubits;
    int rows = 1 << qubits;
    int n_vec = state_vect->n_vectors;
    __m512 *avx_v = state_vect->state_vec;

    int reach = 8;
    if(qubits == 1) reach = 2;
    if(qubits == 2) reach = 4; 

    char *tmp_ptr;
    for(int reg = 0; reg < n_vec; reg++){
        float values[16];
        _mm512_store_ps(values, avx_v[reg]);
        for(int i=0; i < reach; i++){
            char buffer[256];
            buffer[0] = '\0';
            tmp_ptr = strcat(buffer, "|");
            int state_index = reg * 8 + i;
            for(int j=0; j < qubits; j++){
                int mask = 1 << (qubits - j -1);
                if(state_index & mask){
                    tmp_ptr = strcat(buffer, "1");
                }else{
                    tmp_ptr = strcat(buffer, "0");
                }
            }
            char tmp_extra[32];
            float real_val = values[2*i];
            float im_val = values[2*i+1];
            float probability = real_val*real_val + im_val*im_val;
            snprintf(tmp_extra, sizeof(tmp_extra),"> %.3f %%\n", probability);
            strcat(buffer, tmp_extra);
            printf("%s",buffer);
            //if (strlen(buffer) >= buffer_size - 256) {
            //    fprintf(stderr, "Buffer size exceeded!\n");
            //    return;
            //}
        }
    }
}

/** void state_vector_probability_to_string(state_vect, buffer)
 *   Print the state vector values, as polar complex numbers.
 */
void state_vector_polar_to_string(StateVector *state_vect){
        int qubits = state_vect->n_qubits;
    int rows = 1 << qubits;
    int n_vec = state_vect->n_vectors;
    __m512 *avx_v = state_vect->state_vec;

    int reach = 8;
    if(qubits == 1) reach = 2;
    if(qubits == 2) reach = 4; 

    char *tmp_ptr;
    for(int reg = 0; reg < n_vec; reg++){
        float values[16];
        _mm512_store_ps(values, avx_v[reg]);
        for(int i=0; i < reach; i++){
            char buffer[256];
            buffer[0] = '\0';
            tmp_ptr = strcat(buffer, "|");
            int state_index = reg * 8 + i;
            for(int j=0; j < qubits; j++){
                int mask = 1 << (qubits - j -1);
                if(state_index & mask){
                    tmp_ptr = strcat(buffer, "1");
                }else{
                    tmp_ptr = strcat(buffer, "0");
                }
            }
            char tmp_extra[32];
            float real_val = values[2*i];
            float im_val = values[2*i+1];
            float magnitude = sqrt((real_val*real_val) + (im_val*im_val));
            float angle = 0.0;
            if(real_val > 0.01){
                angle = atan2(im_val, real_val);
            }else if(real_val < -0.01){
                angle = atan2(im_val, real_val) + 1;
            }else if(im_val >= 0.0){
                angle = 1/2;
            }else{
                angle = -1/2;
            }
            
            snprintf(tmp_extra, sizeof(tmp_extra),"> r = %.3f, t = %.3f\n", magnitude, angle);
            strcat(buffer, tmp_extra);
            printf("%s",buffer);

            //if (strlen(buffer) >= buffer_size - 256) {
            //    fprintf(stderr, "Buffer size exceeded!\n");
            //    return;
            //}
        }
    }
}

/** void free_state_vector(state_vect)
 *   free all memory allocated to state_vect
 */
void free_state_vector(StateVector *state_vect){
    if (state_vect == NULL) return;

    if (state_vect->state_vec != NULL) {
        free(state_vect->state_vec);
        state_vect->state_vec = NULL;
    }

    state_vect->n_qubits = 0;
    state_vect->n_vectors = 0;
}
