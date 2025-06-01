#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "state_vector.h"
#include "vector_masks.h"

/** void permute_vector(vec1, qubit)
 *   instead of modifying the gate matrix operator, it is faster to permute the state vector,
 *   this function allows us to quickly execute the most frequently used permutes.
 */
void permute_vector(__m512 *vec1, int qubit){
    __m512i mask;
    switch (qubit)
    {
    case 2:
        mask = get_mask(PERMUTE_V_1);
        *vec1 = _mm512_permutexvar_ps(mask, *vec1);
        break;
    case 3:
        mask = get_mask(PERMUTE_V_2);
        *vec1 = _mm512_permutexvar_ps(mask, *vec1);
        break;
    case 20:
        mask = get_mask(PERMUTE_V_3);
        *vec1 = _mm512_permutexvar_ps(mask, *vec1);
        break;
    case 21:
        mask = get_mask(PERMUTE_V_4);
        *vec1 = _mm512_permutexvar_ps(mask, *vec1);
        break;
    case -21:
        mask = get_mask(PERMUTE_V_5);
        *vec1 = _mm512_permutexvar_ps(mask, *vec1);
        break;
    default:
        break;
    }
}

/**  void swap_half(vec1, vec2)
 *    swaps the bottom half of vec1 for the top half of vec2
 *    this is used in non-local matrix operations
 */
void swap_half(__m512 *vec1, __m512 *vec2){
    __m512i mask = get_mask(HALF_SWAP_1);
    __m512i mask2 = get_mask(HALF_SWAP_2);
    __m512 tmp = _mm512_permutex2var_ps(*vec1, mask, *vec2);
    *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
    *vec2 = tmp;
}

/**  void swap_pos_qbit(vec1, vec2, qubit)
 *    swap elements from two vectors, the position of these elements
 *    is decided by the qubit on which the swap must be executed.
 */
void swap_pos_qbit(__m512 *vec1, __m512 *vec2, int qubit){
    __m512i mask2;
    __m512i mask;
    __m512 tmp;
    switch (qubit)
    {
    case 1:
        mask = get_mask(TWO_POS_SWAP_1);
        mask2 = get_mask(TWO_POS_SWAP_2);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    case 2:
        mask = get_mask(TWO_POS_SWAP_3);
        mask2 = get_mask(TWO_POS_SWAP_4);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    case 3:
        mask = get_mask(TWO_POS_SWAP_5);
        mask2 = get_mask(TWO_POS_SWAP_6);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    default:
        mask = get_mask(HALF_SWAP_1);
        mask2 = get_mask(HALF_SWAP_2);
        tmp = _mm512_permutex2var_ps(*vec1, mask, *vec2);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    }
}

/** void reverse_swap_pos_qbit(vec1, vec2, qubit)
 *   this function allows us to reverse the swap_pos_qbit operation.
 */
void reverse_swap_pos_qbit(__m512 *vec1, __m512 *vec2, int qubit){
    __m512i mask2;
    __m512i mask;
    __m512 tmp;
    switch (qubit)
    {
    case 1:
        mask = get_mask(TWO_POS_SWAP_7);
        mask2 = get_mask(TWO_POS_SWAP_8);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    case 2:
        mask = get_mask(TWO_POS_SWAP_9);
        mask2 = get_mask(TWO_POS_SWAP_10);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    case 3:
        mask = get_mask(TWO_POS_SWAP_11);
        mask2 = get_mask(TWO_POS_SWAP_12);
        tmp = _mm512_permutex2var_ps(*vec2, mask, *vec1);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    default:
        mask2 = get_mask(HALF_SWAP_1);
        mask = get_mask(HALF_SWAP_2);
        tmp = _mm512_permutex2var_ps(*vec1, mask, *vec2);
        *vec1 = _mm512_permutex2var_ps(*vec2, mask2, *vec1);
        *vec2 = tmp;
        break;
    }
}

/** void quad_swap(vec1, vec2, vec3, vec4)
 *   swaps elements from four different vector registers such as:
 * 
 *      1a 1b 1c 1d 1e 1f 1g 1h 1i 1j 1k 1l 1m 1n 1o 1p
 *      2a 2b 2c 2d 2e 2f 2g 2h 2i 2j 2k 2l 2m 2n 2o 2p
 *      3a 3b 3c 3d 3e 3f 3g 3h 3i 3j 3k 3l 3m 3n 3o 3p
 *      4a 4b 4c 4d 4e 4f 4g 4h 4i 4j 4k 4l 4m 4n 4o 4p
 * 
 *      vvvvv
 * 
 *      1a 1b 2a 2b 3a 3b 4a 4b 1i 1j 2i 2j 3i 3j 4i 4j
 *      1c 1d 2c 2d 3c 3d 4c 4d 1k 1l 2k 2l 3k 3l 4k 4l
 *      1e 1f 2e 2f 3e 3f 4e 4f 1m 1n 2m 2n 3m 3n 4m 4n
 *      1f 1g 2f 2g 3f 3g 4f 4g 1o 1p 2o 2p 3o 3p 4o 4p
 */
void quad_swap(__m512 *vec1, __m512 *vec2, __m512 *vec3, __m512 *vec4){

    __m512i mask_1 = get_mask(QUAD_SWAP_1);
    __m512i mask_2 = get_mask(QUAD_SWAP_2);
    __m512i mask_3 = get_mask(QUAD_SWAP_3);
    __m512 tmp_1 = _mm512_permutex2var_ps(*vec2, mask_1, *vec1);
    tmp_1 = _mm512_permutex2var_ps(*vec3, mask_2, tmp_1);
    tmp_1 = _mm512_permutex2var_ps(*vec4, mask_3, tmp_1);

    __m512i mask_4 = get_mask(QUAD_SWAP_4);
    __m512i mask_5 = get_mask(QUAD_SWAP_5);
    __m512i mask_6 = get_mask(QUAD_SWAP_6);
    __m512 tmp_2 = _mm512_permutex2var_ps(*vec2, mask_4, *vec1);
    tmp_2 = _mm512_permutex2var_ps(*vec3, mask_5, tmp_2);
    tmp_2 = _mm512_permutex2var_ps(*vec4, mask_6, tmp_2);

    __m512i mask_7 = get_mask(QUAD_SWAP_7);
    __m512i mask_8 = get_mask(QUAD_SWAP_8);
    __m512i mask_9 = get_mask(QUAD_SWAP_9);
    __m512 tmp_3 = _mm512_permutex2var_ps(*vec2, mask_7, *vec1);
    tmp_3 = _mm512_permutex2var_ps(*vec3, mask_8, tmp_3);
    tmp_3 = _mm512_permutex2var_ps(*vec4, mask_9, tmp_3);

    __m512i mask_10 = get_mask(QUAD_SWAP_10);
    __m512i mask_11 = get_mask(QUAD_SWAP_11);
    __m512i mask_12 = get_mask(QUAD_SWAP_12);
    __m512 tmp_4 = _mm512_permutex2var_ps(*vec2, mask_10, *vec1);
    tmp_4 = _mm512_permutex2var_ps(*vec3, mask_11, tmp_4);
    tmp_4 = _mm512_permutex2var_ps(*vec4, mask_12, tmp_4);

    *vec1 = tmp_1;
    *vec2 = tmp_2;
    *vec3 = tmp_3;
    *vec4 = tmp_4;
}

/** __m512 complex_matrix_vector_mul_avx512_4x4(st_vec, gate1, gate2, mask1... mask8)
 *   AVX512 implementation of complex number [4x4] * [4x1] matrix-vector multiplication.
 *   The operation matrix must be passed in two halves (gate1 and gate2).
 *   Masks are passed as arguments, they are needed to execute the operation, and passing them as arguments
 *   instead of creating them everytime this function is called results in a big reduction of runtime, as this
 *   operation is executed 2^(n-1) times per gate, for n qubits.
 */
__m512 complex_matrix_vector_mul_avx512_4x4(__m512 st_vec, __m512 *gate1, __m512 *gate2, __m512i mask1, __m512i mask2, __m512i mask3, __m512i mask4, __m512i mask5, __m512i mask6, __m512i mask7, __m512i mask8){
    __m512 l1_1 = _mm512_permutexvar_ps(mask1, st_vec);
    __m512 l2_1 = _mm512_permutexvar_ps(mask3, st_vec);
    __m512 l3_1 = _mm512_permutexvar_ps(mask5, st_vec);
    __m512 l4_1 = _mm512_permutexvar_ps(mask7, st_vec);
    __m512 l1_2 = _mm512_permutexvar_ps(mask2, st_vec);
    __m512 l2_2 = _mm512_permutexvar_ps(mask4, st_vec);
    __m512 l3_2 = _mm512_permutexvar_ps(mask6, st_vec);
    __m512 l4_2 = _mm512_permutexvar_ps(mask8, st_vec);

    __m512 mult1_1 = _mm512_mul_ps(l1_1, gate1[0]);
    __m512 mult2_1 = _mm512_mul_ps(l2_1, gate2[0]);
    __m512 mult3_1 = _mm512_mul_ps(l3_1, gate1[1]);
    __m512 mult4_1 = _mm512_mul_ps(l4_1, gate2[1]);
    __m512 mult1_2 = _mm512_mul_ps(l1_2, gate1[2]);
    __m512 mult2_2 = _mm512_mul_ps(l2_2, gate2[2]);
    __m512 mult3_2 = _mm512_mul_ps(l3_2, gate1[3]);
    __m512 mult4_2 = _mm512_mul_ps(l4_2, gate2[3]);

    __m512 row1 = _mm512_add_ps(mult1_1, mult1_2);
    __m512 row2 = _mm512_add_ps(mult2_1, mult2_2);
    __m512 row3 = _mm512_add_ps(mult3_1, mult3_2);
    __m512 row4 = _mm512_add_ps(mult4_1, mult4_2);

    __m512 result = _mm512_add_ps(_mm512_add_ps(row1, row2), _mm512_add_ps(row3, row4));
    return result;
}

/** __m512 complex_matrix_vector_mul_avx512_2x2(st_vec, gate, mask1... mask4)
 *   AVX512 implementation of complex number [2x2] * [2x1] matrix-vector multiplication
 *   Masks are passed as arguments, they are needed to execute the operation, and passing them as arguments
 *   instead of creating them everytime this function is called results in a big reduction of runtime, as this
 *   operation is executed 2^(n) times per gate, for n qubits.
 */
__m512 complex_matrix_vector_mul_avx512_2x2(__m512 st_vec, __m512 *gate, __m512i mask1, __m512i mask2, __m512i mask3, __m512i mask4){

    __m512 q1 = _mm512_permutexvar_ps(mask1, st_vec);
    __m512 q2 = _mm512_permutexvar_ps(mask2, st_vec);
    __m512 q3 = _mm512_permutexvar_ps(mask3, st_vec);
    __m512 q4 = _mm512_permutexvar_ps(mask4, st_vec);

    __m512 gate_row1 = gate[0];
    __m512 gate_row2 = gate[1];

    __m512 mult1 = _mm512_mul_ps(q1, gate_row1);
    __m512 mult2 = _mm512_mul_ps(q2, gate_row2);
    __m512 mult3 = _mm512_mul_ps(q3, gate_row1);
    __m512 mult4 = _mm512_mul_ps(q4, gate_row2);

    __m256 l1 = _mm512_castps512_ps256(mult1);
    __m256 h1 = _mm512_extractf32x8_ps(mult1,1);
    __m256 l2 = _mm512_castps512_ps256(mult2);
    __m256 h2 = _mm512_extractf32x8_ps(mult2,1);
    __m256 l3 = _mm512_castps512_ps256(mult3);
    __m256 h3 = _mm512_extractf32x8_ps(mult3,1);
    __m256 l4 = _mm512_castps512_ps256(mult4);
    __m256 h4 = _mm512_extractf32x8_ps(mult4,1);

    __m256 a1 = _mm256_hadd_ps(l1,h1);
    __m256 a2 = _mm256_hadd_ps(l2,h2);
    __m256 a3 = _mm256_hadd_ps(l3,h3);
    __m256 a4 = _mm256_hadd_ps(l4,h4);
    
    __m512 half1 = _mm512_insertf32x8(_mm512_castps256_ps512(a1), a3, 1);
    __m512 half2 = _mm512_insertf32x8(_mm512_castps256_ps512(a2), a4, 1);

    __m512 res = _mm512_add_ps(half1, half2);
    return res;
}

/** void non_local_apply_gate(st_vec, n_vec, qubit, gate)
 *   If the gate to be applied is non-local, ex: gate for a qubit larger than number 2, then this function is called.
 */
void non_local_apply_gate(__m512 *st_vec, int n_vec, int qubit, __m512 *gate){

    int factor = qubit - 3;
    __m512i mask1 = get_mask(MULT2X2_1);
    __m512i mask2 = get_mask(MULT2X2_2);
    __m512i mask3 = get_mask(MULT2X2_3);
    __m512i mask4 = get_mask(MULT2X2_4);

    int pointer = 0;
    while(pointer < n_vec){
        swap_half(&st_vec[pointer], &st_vec[pointer+factor]);
        permute_vector(&st_vec[pointer], 3);
        permute_vector(&st_vec[pointer + factor], 3);
        st_vec[pointer] = complex_matrix_vector_mul_avx512_2x2(st_vec[pointer], gate, mask1, mask2, mask3, mask4);
        st_vec[pointer+factor] = complex_matrix_vector_mul_avx512_2x2(st_vec[pointer+factor], gate, mask1, mask2, mask3, mask4);
        permute_vector(&st_vec[pointer], 3);
        permute_vector(&st_vec[pointer + factor], 3);
        swap_half(&st_vec[pointer], &st_vec[pointer+factor]);
        pointer++;
        if(pointer % factor == 0){
            pointer += factor;
        }
    }
}

/** void local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high)
 *   Local operation for double qubit gates, control_qubit and qubit.
 *   The parameter control_high indicates whether the control qubit has a
 *   larger value than the qubit on which the operation is applied.
 */
void local_d_gate(__m512 *st_vec, int n_vec, int control_qubit, int qubit, __m512 *gate1, __m512 *gate2, int control_high){

    __m512i mask1 = get_mask(MULT4X4_1);
    __m512i mask2 = get_mask(MULT4X4_2);
    __m512i mask3 = get_mask(MULT4X4_3);
    __m512i mask4 = get_mask(MULT4X4_4);
    __m512i mask5 = get_mask(MULT4X4_5);
    __m512i mask6 = get_mask(MULT4X4_6);
    __m512i mask7 = get_mask(MULT4X4_7);
    __m512i mask8 = get_mask(MULT4X4_8);

    for(int i = 0; i < n_vec; i++){
                if(control_qubit == 3 || qubit == 3){
                    if(control_qubit == 2 || qubit == 2){
                        permute_vector(&st_vec[i],21);
                        if(control_qubit < qubit){
                            permute_vector(&st_vec[i],2);
                            st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                            permute_vector(&st_vec[i],2);
                        }else{
                            st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                        }
                        permute_vector(&st_vec[i],-21);
                    }else{
                        permute_vector(&st_vec[i],20);
                        if(control_qubit < qubit){
                            permute_vector(&st_vec[i],2);
                            st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                            permute_vector(&st_vec[i],2);
                        }else{
                            st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                        }
                        permute_vector(&st_vec[i],20);
                    }
                }else{
                    if(control_qubit < qubit){
                        permute_vector(&st_vec[i],2);
                        st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                        permute_vector(&st_vec[i],2);
                    }else{
                        st_vec[i] = complex_matrix_vector_mul_avx512_4x4(st_vec[i], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                    }
                }
    }
}

/** void half_non_local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high)
 *   Implementation of double qbit gate for AVX.
 *   This function is called when the operation needs to be applied to both local and non-local registers.
 */
void half_non_local_d_gate(__m512 *st_vec, int n_vec, int control_qubit, int qubit, __m512 *gate1, __m512 *gate2, int control_high){

    __m512i mask1 = get_mask(MULT4X4_1);
    __m512i mask2 = get_mask(MULT4X4_2);
    __m512i mask3 = get_mask(MULT4X4_3);
    __m512i mask4 = get_mask(MULT4X4_4);
    __m512i mask5 = get_mask(MULT4X4_5);
    __m512i mask6 = get_mask(MULT4X4_6);
    __m512i mask7 = get_mask(MULT4X4_7);
    __m512i mask8 = get_mask(MULT4X4_8);

    int pointer = 0;
        while(pointer <= n_vec/2){
            int leap = (control_qubit > 3) ? 1 << (control_qubit - 4) : 1 << (qubit - 4);
            if(control_high){
                swap_pos_qbit(&st_vec[pointer], &st_vec[pointer+leap],qubit);
                st_vec[pointer] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                st_vec[pointer+leap] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                reverse_swap_pos_qbit(&st_vec[pointer], &st_vec[pointer+leap],qubit);
            }else{
                swap_pos_qbit(&st_vec[pointer], &st_vec[pointer+leap],control_qubit);
                permute_vector(&st_vec[pointer],2);
                permute_vector(&st_vec[pointer+leap],2);
                st_vec[pointer] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                st_vec[pointer+leap] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
                permute_vector(&st_vec[pointer],2);
                permute_vector(&st_vec[pointer+leap],2);
                reverse_swap_pos_qbit(&st_vec[pointer], &st_vec[pointer+leap],control_qubit);
            }
            pointer++;
            if(pointer % leap == 0){
                pointer += leap;
            }
        }
}

/** void fully_non_local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high)
 *   Implementation of double qbit gate for AVX. 
 *   This function is called when the operation needs to be applied to only non-local registers.
 */
void fully_non_local_d_gate(__m512 *st_vec, int n_vec, int control_qubit, int qubit, __m512 *gate1, __m512 *gate2, int control_high){
    
    __m512i mask1 = get_mask(MULT4X4_1);
    __m512i mask2 = get_mask(MULT4X4_2);
    __m512i mask3 = get_mask(MULT4X4_3);
    __m512i mask4 = get_mask(MULT4X4_4);
    __m512i mask5 = get_mask(MULT4X4_5);
    __m512i mask6 = get_mask(MULT4X4_6);
    __m512i mask7 = get_mask(MULT4X4_7);
    __m512i mask8 = get_mask(MULT4X4_8);
    
    int pointer = 0;
    while(pointer <= n_vec/4){
        int leap_1 = 1 << (control_qubit - 4);
        int leap_2 = 1 << (qubit - 4);
        int leap_3 = leap_1 + leap_2;

        quad_swap(&st_vec[pointer],&st_vec[pointer+leap_1],&st_vec[pointer+leap_2],&st_vec[pointer+leap_3]);
        if(control_high){
            st_vec[pointer] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_1] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_1], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_2] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_2], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_3] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_3], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
        }else{
            permute_vector(&st_vec[pointer],2);
            permute_vector(&st_vec[pointer+leap_1],2);
            permute_vector(&st_vec[pointer+leap_2],2);
            permute_vector(&st_vec[pointer+leap_3],2);
            st_vec[pointer] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_1] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_1], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_2] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_2], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            st_vec[pointer+leap_3] = complex_matrix_vector_mul_avx512_4x4(st_vec[pointer+leap_3], gate1, gate2, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8);
            permute_vector(&st_vec[pointer],2);
            permute_vector(&st_vec[pointer+leap_1],2);
            permute_vector(&st_vec[pointer+leap_2],2);
            permute_vector(&st_vec[pointer+leap_3],2);
        }
        quad_swap(&st_vec[pointer],&st_vec[pointer+leap_1],&st_vec[pointer+leap_2],&st_vec[pointer+leap_3]);

        pointer++;
        if(pointer % leap_1 == 0){
            pointer += leap_1;
        }
        if(pointer % leap_2 == 0){
            pointer += leap_2;
        }
        if(pointer % leap_3 == 0){
            pointer += leap_3;
        }
    }
}

/** void apply_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2)
 *   Implementation of double qbit gate for AVX. 
 *   This function determines whether the operation is local, non-local or both.
 */
void apply_d_gate(__m512 *st_vec, int n_vec, int control_qubit, int qubit, __m512 *gate1, __m512 *gate2){
    int control_high = (control_qubit > qubit);
    int num_vec_sw = 1;

    if (qubit > 3) num_vec_sw = num_vec_sw << 1;
    if (control_qubit > 3) num_vec_sw = num_vec_sw << 1;

    if(num_vec_sw == 4){
        fully_non_local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high);
    }else if(num_vec_sw == 2){
        half_non_local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high);
    }else{
        local_d_gate(st_vec, n_vec, control_qubit, qubit, gate1, gate2, control_high);
    }
}

/** void local_apply_gate(st_vec, n_vec, qubit, gate)
 *   Implementation of single qbit gate for AVX.
 */
void local_apply_gate(__m512 *st_vec, int n_vec, int qubit, __m512 *gate){
    __m512i mask1;
    __m512i mask2;
    __m512i mask3;
    __m512i mask4;
    // Máscaras para la permutación
    mask1 = get_mask(MULT2X2_1);
    mask2 = get_mask(MULT2X2_2);
    mask3 = get_mask(MULT2X2_3);
    mask4 = get_mask(MULT2X2_4);
    for(int i = 0; i < n_vec; i++){
        permute_vector(&st_vec[i], qubit);
        st_vec[i] = complex_matrix_vector_mul_avx512_2x2(st_vec[i], gate, mask1, mask2, mask3, mask4);
        permute_vector(&st_vec[i], qubit);
    }
}

/** void local_apply_gate(st_vec, n_vec, qubit, gate)
 *   Implementation of single qbit gate for AVX.
 */
void apply_single_qubit_gate(StateVector *state_vect, int qubit, float *mat){   //TODO: ESTA FUNCIÓN AHORA MISMO ES REDUNDANTE, EDITAR ESTA O APPLY_GATE
    int n_qubits_total = state_vect->n_qubits;
    int n_vects = state_vect->n_vectors;
    __m512 *s_vect = state_vect->state_vec;
    qubit = qubit+1;

    __m512 gate[2] = {
        _mm512_set_ps(mat[4], mat[5], -mat[5], mat[4], mat[4], mat[5], -mat[5], mat[4], mat[0], mat[1], -mat[1], mat[0], mat[0], mat[1], -mat[1], mat[0]),
        _mm512_set_ps(mat[6], mat[7], -mat[7], mat[6], mat[6], mat[7], -mat[7], mat[6], mat[2], mat[3], -mat[3], mat[2], mat[2], mat[3], -mat[3], mat[2])
    };

    if(qubit > n_qubits_total || qubit < 0){
        fprintf(stderr, "Single qubit gate error!\n");
        exit(EXIT_FAILURE);
    }
    if(qubit > 3){
        non_local_apply_gate(s_vect, n_vects, qubit, gate);
    }else{
        local_apply_gate(s_vect, n_vects, qubit, gate);
    }
}

/** void local_apply_gate(st_vec, n_vec, qubit, gate)
 *   Implementation of double qbit gate for AVX.
 */
void double_qubit_gate(StateVector *state_vect, int control_qubit, int qubit, float *mat){   //TODO: ESTA FUNCIÓN AHORA MISMO ES REDUNDANTE, EDITAR ESTA O APPLY_GATE
    int n_qubits_total = state_vect->n_qubits;
    int n_vects = state_vect->n_vectors;
    __m512 *s_vect = state_vect->state_vec;
    qubit = qubit+1;
    control_qubit = control_qubit+1;

    __m512 *gate_D1 = (__m512 *)_mm_malloc(2 * 4 * sizeof(__m512), 64);
    __m512 *gate_D2 = (__m512 *)_mm_malloc(2 * 4 * sizeof(__m512), 64);

    gate_D1[0] = _mm512_set_ps(mat[25], mat[24], mat[17], mat[16], mat[9], mat[8], mat[1], mat[0], mat[25], mat[24], mat[17], mat[16], mat[9], mat[8], mat[1], mat[0]);
    gate_D1[1] = _mm512_set_ps(mat[29], mat[28], mat[21], mat[20], mat[13], mat[12], mat[5], mat[4], mat[29], mat[28], mat[21], mat[20], mat[13], mat[12], mat[5], mat[4]);
    gate_D1[2] = _mm512_set_ps(mat[24], -mat[25], mat[16], -mat[17], mat[8], -mat[9], mat[0], -mat[1], mat[24], -mat[25], mat[16], -mat[17], mat[8], -mat[9], mat[0], -mat[1]);
    gate_D1[3] = _mm512_set_ps(mat[28], -mat[29], mat[20], -mat[21], mat[12], -mat[13], mat[4], -mat[5], mat[28], -mat[29], mat[20], -mat[21], mat[12], -mat[13], mat[4], -mat[5]);

    gate_D2[0] = _mm512_set_ps(mat[27], mat[26], mat[19], mat[18], mat[11], mat[10], mat[3], mat[2], mat[27], mat[26], mat[19], mat[18], mat[11], mat[10], mat[3], mat[2]);
    gate_D2[1] = _mm512_set_ps(mat[31], mat[30], mat[23], mat[22], mat[15], mat[14], mat[7], mat[6], mat[31], mat[30], mat[23], mat[22], mat[15], mat[14], mat[7], mat[6]);
    gate_D2[2] = _mm512_set_ps(mat[26], -mat[27], mat[18], -mat[19], mat[10], -mat[11], mat[2], -mat[3], mat[26], -mat[27], mat[18], -mat[19], mat[10], -mat[11], mat[2], -mat[3]);
    gate_D2[3] = _mm512_set_ps(mat[30], -mat[31], mat[22], -mat[23], mat[14], -mat[15], mat[6], -mat[7], mat[30], -mat[31], mat[22], -mat[23], mat[14], -mat[15], mat[6], -mat[7]);

    if(qubit > n_qubits_total || control_qubit > n_qubits_total){
        fprintf(stderr, "Double qubit gate error!\n");
        exit(EXIT_FAILURE);
    }

    apply_d_gate(s_vect, n_vects, control_qubit, qubit, gate_D1, gate_D2);
}

void X_gate(StateVector *state_vect, int qubit){

    float mat[8] = {0.0, 0.0,   1.0, 0.0,
                    1.0, 0.0,   0.0, 0.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void Y_gate(StateVector *state_vect, int qubit){

    float mat[8] = {0.0, 0.0,   0.0, -1.0,
                    0.0, 1.0,   0.0, 0.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void Z_gate(StateVector *state_vect, int qubit){

    float mat[8] = {1.0, 0.0,   0.0, 0.0,
                    0.0, 0.0,   -1.0, 0.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void H_gate(StateVector *state_vect, int qubit){

    float mat[8] = {0.7071, 0.0,   0.7071, 0.0,
                    0.7071, 0.0,   -0.7071, 0.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void CNOT_gate(StateVector *state_vect, int control_qubit, int qubit){

    float mat[32] = {1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f};

    double_qubit_gate(state_vect, control_qubit, qubit, mat);
}

void SWAP_gate(StateVector *state_vect, int control_qubit, int qubit){

    float mat[32] = {1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f};

    double_qubit_gate(state_vect, control_qubit, qubit, mat);
}

void C_Phase_gate(StateVector *state_vect, int control_qubit, int qubit, float phi){

    float mat[32] = {1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f};

    mat[30] = cosf(phi);
    mat[31] = sinf(phi);

    double_qubit_gate(state_vect, control_qubit, qubit, mat);
}

void S_gate(StateVector *state_vect, int qubit){

    float mat[8] = {1.0, 0.0,   0.0, 0.0,
                    0.0, 0.0,   0.0, 1.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void T_gate(StateVector *state_vect, int qubit){

    float mat[8] = {1.0, 0.0,   0.0, 0.0,
                    0.0, 0.0,   0.7071, 0.7071};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void debug_gate(StateVector *state_vect, int qubit){

    float mat[8] = {0.0, 0.0,   1.0, 0.0,
                    1.0, 0.0,   0.0, 0.0};

    apply_single_qubit_gate(state_vect, qubit, mat);
}

void debug_gate_d(StateVector *state_vect, int c_qubit, int qubit){

    float mat[32] = {0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     1.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,
                     0.0f, 0.0f,   0.0f, 0.0f,   1.0f, 0.0f,   0.0f, 0.0f};

    double_qubit_gate(state_vect, c_qubit, qubit, mat);
}

int Z_measure(StateVector *state_vect, int qubit){
    int size = state_vect->n_qubits;
    __m512 *sv = state_vect->state_vec;
    int jump = 1 << qubit;
    float prob = 0;
    for(int i = 0; i < (1<<(size-1)); i++){
        int pos = i % 8;
        int vec = i / 8;
        float values[16];
        _mm512_store_ps(values, sv[vec]);
        float real = values[2*pos];
        float im = values[2*pos + 1];
        prob += (real*real + im*im);
    }
    srand(time(NULL));
    double random_num = (double)rand() / RAND_MAX;

    if(random_num <= prob) return 1;
    else return 0;
}
