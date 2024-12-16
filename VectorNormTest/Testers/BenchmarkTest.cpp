#include <benchmark/benchmark.h>
#include "../Normalizers/SelectedNormalizer.h"
#include "../Utils/VectorUtils.h"
/*
 * This file contains macros magic to choose the implementation of a function based on the
 * dimension's remainder. It is used to collapse large and repetitive switch statements that are
 * used to choose and define the templated values of the implementation of the distance functions.
 * We assume that we are dealing with 512-bit blocks, so we define a chunk size of 32 for 16-bit
 * elements, 16 for 32-bit elements, and a chunk size of 8 for 64-bit elements. The main macro is
 * CHOOSE_IMPLEMENTATION, and it's the one that should be used.
 */

// Macro for a single case. Sets __ret_dist_func to the function with the given remainder.

// run command: 
/* 
mkdir -p result_jsons && ./BenchmarkTest --benchmark_format=json --benchmark_out=result_jsons/AVX_FP32.json
mkdir -p result_jsons && ./BenchmarkTest --benchmark_format=json --benchmark_out=result_jsons/AVX_FP64.json

*/

using func_t = void(*)(const void*, size_t);
template<typename Algorithm = Naive > 
static void BM_vector_norm(benchmark::State& state) {
    size_t dim = state.range(0);
    func_t func;
    CHOOSE_IMPLEMENTATION(func, dim, 8, Algorithm:: template Normalize);
    VECT_TYPE* vec = new VECT_TYPE[dim];
    for(int i=0;i<dim;i++) vec[i] = 1;
  for (auto _ : state)
  {
    // std::vector<float> vec(dim,1);
    func(vec,dim);
  }
  delete[] vec;
}

// BENCHMARK(BM_vector_norm<SingleRsqrt>)->DenseRange(32,2048,1);
BENCHMARK(BM_vector_norm<SingleSqrt>)->DenseRange(32,2048,1);
BENCHMARK(BM_vector_norm<SingleScalar>)->DenseRange(32,2048,1);
BENCHMARK(BM_vector_norm<Naive>)->DenseRange(32,2048,1);


BENCHMARK_MAIN();