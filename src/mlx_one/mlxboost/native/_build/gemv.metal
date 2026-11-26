// MLX arithmetic-order/API adaptation; Apple MIT notice in ../MLX_LICENSE.txt.
#include <metal_stdlib>
#include <metal_simdgroup>
#include "mlx/backend/metal/kernels/bf16.h"
using namespace metal;

template <typename T, uint Bits, uint Group>
[[kernel]] void mlxboost_gemv(
    device const T* x [[buffer(0)]], device const uint* w [[buffer(1)]],
    device const T* scales [[buffer(2)]], device const T* biases [[buffer(3)]],
    device T* out [[buffer(4)]], constant uint& k_size [[buffer(5)]],
    constant uint& n_size [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint Pack = 32 / Bits;
  constexpr uint Mask = (1u << Bits) - 1;
  constexpr uint Rows = 4;
  ulong row = (ulong(group.x) * (threads.x / 32) + simd) * Rows;
  if (row >= n_size) return; // Entire SIMD group agrees; safe before simd_sum.
  float partial[Rows] = {0};
  for (uint base = uint(lane) * Pack; base < k_size; base += 32 * Pack) {
    float values[Pack];
    #pragma unroll
    for (uint i=0; i<Pack; ++i) values[i] = float(x[base+i]);
    float sum = 0;
    if constexpr (Bits == 4) {
      // Pinned MLX load_vector forms its bias correction from T-valued sums
      // of four activations. Retain that rounding boundary, notably for BF16.
      #pragma unroll
      for (uint chunk = 0; chunk < Pack; chunk += 4) {
        T sum4 = T(T(T(x[base+chunk] + x[base+chunk+1]) + x[base+chunk+2]) + x[base+chunk+3]);
        sum += float(sum4);
      }
    } else {
      #pragma unroll
      for (uint i = 0; i < Pack; ++i) sum += values[i];
    }
    // Reuse the activation pack and its affine bias correction across four
    // output rows. Registers only; no threadgroup or dequantized workspace.
    #pragma unroll
    for (uint r=0; r<Rows; ++r) {
      if (row+r >= n_size) continue;
      uint word = w[(row+r) * (k_size / Pack) + base / Pack];
      ulong meta = (row+r) * (k_size / Group) + base / Group;
      float dot = 0;
      #pragma unroll
      for (uint chunk=0; chunk<Pack; chunk+=4) {
        float chunk_dot = 0;
        #pragma unroll
        for (uint j=0; j<4; ++j)
          chunk_dot += values[chunk+j] * float((word >> (Bits*(chunk+j))) & Mask);
        dot += chunk_dot;
      }
      partial[r] += float(scales[meta]) * dot + float(biases[meta]) * sum;
    }
  }
  #pragma unroll
  for (uint r=0; r<Rows; ++r) {
    float total = simd_sum(partial[r]);
    if (lane == 0 && row+r<n_size) out[row+r] = T(total);
  }
}

#define GEMV(T, B, G, NAME) \
  template [[host_name(NAME)]] [[kernel]] decltype(mlxboost_gemv<T,B,G>) mlxboost_gemv<T,B,G>;
#define GROUPS(T, B, NAME) \
  GEMV(T,B,32,NAME "32") GEMV(T,B,64,NAME "64") GEMV(T,B,128,NAME "128")
GROUPS(half,4,"mlxboost_gemv_q4_f16_g")
GROUPS(half,8,"mlxboost_gemv_q8_f16_g")
GROUPS(bfloat16_t,4,"mlxboost_gemv_q4_bf16_g")
GROUPS(bfloat16_t,8,"mlxboost_gemv_q8_bf16_g")
