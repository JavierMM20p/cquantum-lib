#ifndef VECTOR_MASKS_H
#define VECTOR_MASKS_H

#include <immintrin.h>

#define MULT2X2_1 0
#define MULT2X2_2 1
#define MULT2X2_3 2
#define MULT2X2_4 3
#define MULT4X4_1 4
#define MULT4X4_2 5
#define MULT4X4_3 6
#define MULT4X4_4 7
#define MULT4X4_5 8
#define MULT4X4_6 9
#define MULT4X4_7 10
#define MULT4X4_8 11
#define QUAD_SWAP_1 12
#define QUAD_SWAP_2 13
#define QUAD_SWAP_3 14
#define QUAD_SWAP_4 15
#define QUAD_SWAP_5 16
#define QUAD_SWAP_6 17
#define QUAD_SWAP_7 18
#define QUAD_SWAP_8 19
#define QUAD_SWAP_9 20
#define QUAD_SWAP_10 21
#define QUAD_SWAP_11 22
#define QUAD_SWAP_12 23
#define HALF_SWAP_1 24
#define HALF_SWAP_2 25
#define PERMUTE_V_1 26
#define PERMUTE_V_2 27
#define PERMUTE_V_3 28
#define PERMUTE_V_4 29
#define PERMUTE_V_5 30
#define TWO_POS_SWAP_1 31
#define TWO_POS_SWAP_2 32
#define TWO_POS_SWAP_3 33
#define TWO_POS_SWAP_4 34
#define TWO_POS_SWAP_5 35
#define TWO_POS_SWAP_6 36
#define TWO_POS_SWAP_7 37
#define TWO_POS_SWAP_8 38
#define TWO_POS_SWAP_9 39
#define TWO_POS_SWAP_10 40
#define TWO_POS_SWAP_11 41
#define TWO_POS_SWAP_12 42

typedef struct AVXVectorMasks {
    __m512i mult2x2_1;
    __m512i mult2x2_2;
    __m512i mult2x2_3;
    __m512i mult2x2_4;
    __m512i mult4x4_1;
    __m512i mult4x4_2;
    __m512i mult4x4_3;
    __m512i mult4x4_4;
    __m512i mult4x4_5;
    __m512i mult4x4_6;
    __m512i mult4x4_7;
    __m512i mult4x4_8;
    __m512i quad_swap_1;
    __m512i quad_swap_2;
    __m512i quad_swap_3;
    __m512i quad_swap_4;
    __m512i quad_swap_5;
    __m512i quad_swap_6;
    __m512i quad_swap_7;
    __m512i quad_swap_8;
    __m512i quad_swap_9;
    __m512i quad_swap_10;
    __m512i quad_swap_11;
    __m512i quad_swap_12;
    __m512i half_swap_1;
    __m512i half_swap_2;
    __m512i permute_v_1;
    __m512i permute_v_2;
    __m512i permute_v_3;
    __m512i permute_v_4;
    __m512i permute_v_5;
    __m512i two_pos_swap_1;
    __m512i two_pos_swap_2;
    __m512i two_pos_swap_3;
    __m512i two_pos_swap_4;
    __m512i two_pos_swap_5;
    __m512i two_pos_swap_6;
    __m512i two_pos_swap_7;
    __m512i two_pos_swap_8;
    __m512i two_pos_swap_9;
    __m512i two_pos_swap_10;
    __m512i two_pos_swap_11;
    __m512i two_pos_swap_12;
} AVXVectorMasks;

__m512i get_mask(int mask);

#endif