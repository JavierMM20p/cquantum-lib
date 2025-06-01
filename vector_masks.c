#include "vector_masks.h"
#include <immintrin.h>

static AVXVectorMasks instance;
static int initialized = 0;

static void init_avx_masks(){
    if(!initialized){
        instance.mult2x2_1 = _mm512_set_epi32(
            5, 4, 5, 4, 1, 0, 1, 0, 5, 4, 5, 4, 1, 0, 1, 0);
        instance.mult2x2_2 = _mm512_set_epi32(
            7, 6, 7, 6, 3, 2, 3, 2, 7, 6, 7, 6, 3, 2, 3, 2);
        instance.mult2x2_3 = _mm512_set_epi32(
            13, 12, 13, 12, 9, 8, 9, 8, 13, 12, 13, 12, 9, 8, 9, 8);
        instance.mult2x2_4 = _mm512_set_epi32(
            15, 14, 15, 14, 11, 10, 11, 10, 15, 14, 15, 14, 11, 10, 11, 10);
        instance.mult4x4_1 = _mm512_set_epi32(
            8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0);
        instance.mult4x4_2 = _mm512_set_epi32(
            9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1);
        instance.mult4x4_3 = _mm512_set_epi32(
            10, 10, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2);
        instance.mult4x4_4 = _mm512_set_epi32(
            11, 11, 11, 11, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3);
        instance.mult4x4_5 = _mm512_set_epi32(
            12, 12, 12, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4);
        instance.mult4x4_6 = _mm512_set_epi32(
            13, 13, 13, 13, 13, 13, 13, 13, 5, 5, 5, 5, 5, 5, 5, 5);
        instance.mult4x4_7 = _mm512_set_epi32(
            14, 14, 14, 14, 14, 14, 14, 14, 6, 6, 6, 6, 6, 6, 6, 6);
        instance.mult4x4_8 = _mm512_set_epi32(
            15, 15, 15, 15, 15, 15, 15, 15, 7, 7, 7, 7, 7, 7, 7, 7);
        instance.quad_swap_1 = _mm512_set_epi32(
            31, 30, 29, 28, 9, 8, 25, 24, 23, 22, 21, 20, 1, 0, 17, 16);
        instance.quad_swap_2= _mm512_set_epi32(
            31, 30, 9, 8, 27, 26, 25, 24, 23, 22, 1, 0, 19, 18, 17, 16);
        instance.quad_swap_3= _mm512_set_epi32(
            9, 8, 29, 28, 27, 26, 25, 24, 1, 0, 21, 20, 19, 18, 17, 16);
        instance.quad_swap_4 = _mm512_set_epi32(
            31, 30, 29, 28, 11, 10, 27, 26, 23, 22, 21, 20, 3, 2, 19, 18);
        instance.quad_swap_5 = _mm512_set_epi32(
            31, 30, 11, 10, 27, 26, 25, 24, 23, 22, 3, 2, 19, 18, 17, 16);
        instance.quad_swap_6 = _mm512_set_epi32(
            11, 10, 29, 28, 27, 26, 25, 24, 3, 2, 21, 20, 19, 18, 17, 16);
        instance.quad_swap_7 = _mm512_set_epi32(
            31, 30, 29, 28, 13, 12, 29, 28, 23, 22, 21, 20, 5, 4, 21, 20);
        instance.quad_swap_8 = _mm512_set_epi32(
            31, 30, 13, 12, 27, 26, 25, 24, 23, 22, 5, 4, 19, 18, 17, 16);
        instance.quad_swap_9 = _mm512_set_epi32(
            13, 12, 29, 28, 27, 26, 25, 24, 5, 4, 21, 20, 19, 18, 17, 16);
        instance.quad_swap_10 = _mm512_set_epi32(
            31, 30, 29, 28, 15, 14, 31, 30, 23, 22, 21, 20, 7, 6, 23, 22);
        instance.quad_swap_11 = _mm512_set_epi32(
            31, 30, 15, 14, 27, 26, 25, 24, 23, 22, 7, 6, 19, 18, 17, 16);
        instance.quad_swap_12 = _mm512_set_epi32(
            15, 14, 29, 28, 27, 26, 25, 24, 7, 6, 21, 20, 19, 18, 17, 16);
        instance.half_swap_1 = _mm512_set_epi32(
                31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8);
        instance.half_swap_2 = _mm512_set_epi32(
                7, 6, 5, 4, 3, 2, 1, 0, 23, 22, 21, 20, 19, 18, 17, 16);
        instance.permute_v_1 = _mm512_set_epi32(
            15, 14, 11, 10, 13, 12, 9, 8, 7, 6, 3, 2, 5, 4, 1, 0);
        instance.permute_v_2 = _mm512_set_epi32(
            15, 14, 7, 6, 11, 10, 3, 2, 13, 12, 5, 4, 9, 8, 1, 0);
        instance.permute_v_3 = _mm512_set_epi32(
            15, 14, 13, 12, 7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 1, 0);
        instance.permute_v_4 = _mm512_set_epi32(
            15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
        instance.permute_v_5 = _mm512_set_epi32(
            15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
        instance.two_pos_swap_1 = _mm512_set_epi32(
                15, 14, 13, 12, 31, 30, 29, 28, 11, 10, 9, 8, 27, 26, 25, 24);
        instance.two_pos_swap_2 = _mm512_set_epi32(
                7, 6, 5, 4, 23, 22, 21, 20, 3, 2, 1, 0, 19, 18, 17, 16);
        instance.two_pos_swap_3 = _mm512_set_epi32(
                15, 14, 11, 10, 31, 30, 27, 26, 13, 12, 9, 8, 29, 28, 25, 24);
        instance.two_pos_swap_4 = _mm512_set_epi32(
                7, 6, 3, 2, 23, 22, 19, 18, 5, 4, 1, 0, 21, 20, 17, 16);
        instance.two_pos_swap_5 = _mm512_set_epi32(
                15, 14, 7, 6, 31, 30, 23, 22, 11, 10, 3, 2, 27, 26, 19, 18);
        instance.two_pos_swap_6 = _mm512_set_epi32(
                13, 12, 5, 4, 29, 28, 21, 20, 9, 8, 1, 0, 25, 24, 17, 16);
        instance.two_pos_swap_7 = _mm512_set_epi32(
                15, 14, 13, 12, 7, 6, 5, 4, 31, 30, 29, 28, 23, 22, 21, 20);
        instance.two_pos_swap_8 = _mm512_set_epi32(
                11, 10, 9, 8, 3, 2, 1, 0, 27, 26, 25, 24, 19, 18, 17, 16);
        instance.two_pos_swap_9 = _mm512_set_epi32(
                15, 14, 7, 6, 13, 12, 5, 4, 31, 30, 23, 22, 29, 28, 21, 20);
        instance.two_pos_swap_10 = _mm512_set_epi32(
                11, 10, 3, 2, 9, 8, 1, 0, 27, 26, 19, 18, 25, 24, 17, 16);
        instance.two_pos_swap_11 = _mm512_set_epi32(
                15, 14, 31, 30, 7, 6, 23, 22, 13, 12, 29, 28, 5, 4, 21, 20);
        instance.two_pos_swap_12 = _mm512_set_epi32(
                11, 10, 27, 26, 3, 2, 19, 18, 9, 8, 25, 24, 1, 0, 17, 16);

        initialized = 1;
    }
}

__m512i get_mask(int mask){
    init_avx_masks();
    switch (mask)
    {
    case 0:
        return instance.mult2x2_1;
    case 1:
        return instance.mult2x2_2;
    case 2:
        return instance.mult2x2_3;
    case 3:
        return instance.mult2x2_4;
    case 4:
        return instance.mult4x4_1;
    case 5:
        return instance.mult4x4_2;
    case 6:
        return instance.mult4x4_3;
    case 7:
        return instance.mult4x4_4;
    case 8:
        return instance.mult4x4_5;
    case 9:
        return instance.mult4x4_6;
    case 10:
        return instance.mult4x4_7;
    case 11:
        return instance.mult4x4_8;
    case 12:
        return instance.quad_swap_1;
    case 13:
        return instance.quad_swap_2;
    case 14:
        return instance.quad_swap_3;
    case 15:
        return instance.quad_swap_4;
    case 16:
        return instance.quad_swap_5;
    case 17:
        return instance.quad_swap_6;
    case 18:
        return instance.quad_swap_7;
    case 19:
        return instance.quad_swap_8;
    case 20:
        return instance.quad_swap_9;
    case 21:
        return instance.quad_swap_10;
    case 22:
        return instance.quad_swap_11;
    case 23:
        return instance.quad_swap_12;   
    case 24:
        return instance.half_swap_1; 
    case 25:
        return instance.half_swap_2;
    case 26:
        return instance.permute_v_1;
    case 27:
        return instance.permute_v_2;
    case 28:
        return instance.permute_v_3;
    case 29:
        return instance.permute_v_4;
    case 30:
        return instance.permute_v_5;
    case 31:
        return instance.two_pos_swap_1;
    case 32:
        return instance.two_pos_swap_2;
    case 33:
        return instance.two_pos_swap_3;
    case 34:
        return instance.two_pos_swap_4;
    case 35:
        return instance.two_pos_swap_5;
    case 36:
        return instance.two_pos_swap_6;
    case 37:
        return instance.two_pos_swap_7;
    case 38:
        return instance.two_pos_swap_8;
    case 39:
        return instance.two_pos_swap_9;
    case 40:
        return instance.two_pos_swap_10;
    case 41:
        return instance.two_pos_swap_11;
    case 42:
        return instance.two_pos_swap_12;
    default:
        return _mm512_set_epi32(
        5, 4, 5, 4, 1, 0, 1, 0, 5, 4, 5, 4, 1, 0, 1, 0);
    }
}