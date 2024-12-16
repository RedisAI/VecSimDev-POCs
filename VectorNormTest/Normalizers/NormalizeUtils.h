#pragma once
#include <immintrin.h>
#include <iostream>
template <__mmask8 mask> // (2^n)-1, where n is in 1..7 (1, 3, ..., 127)
static inline __m256 my_mm256_maskz_loadu_ps(const float *p) {
    // Load 8 floats (assuming this is safe to do)
    __m256 data = _mm256_loadu_ps(p);
    // Set the mask for the loaded data (set 0 if a bit is 0)
    __m256 masked_data = _mm256_blend_ps(_mm256_setzero_ps(), data, mask);

    return masked_data;
}

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))

static inline float my_mm256_reduce_add_ps(__m256 x) {
    float PORTABLE_ALIGN32 TmpRes[8];
    _mm256_store_ps(TmpRes, x);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
           TmpRes[7];
}

template <__mmask8 mask> // (2^n)-1, where n is in 1..3 (1, 3, 7)
static inline __m256d my_mm256_maskz_loadu_pd(const double *p) {
    // Load 4 doubles (assuming this is safe to do)
    __m256d data = _mm256_loadu_pd(p);
    // Set the mask for the loaded data (set 0 if a bit is 0)
    __m256d masked_data = _mm256_blend_pd(_mm256_setzero_pd(), data, mask);

    return masked_data;
}


static inline float my_mm256_reduce_add_pd(__m256d x) {
    double PORTABLE_ALIGN32 TmpRes[4];
    _mm256_store_pd(TmpRes, x);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}