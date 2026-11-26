// MLX encoder/buffer API adaptation. Copyright © 2023-2025 Apple Inc. MIT.
#include <metal_stdlib>
using namespace metal;
template <typename T>
kernel void mlxboost_axpby(device const T* x [[buffer(0)]], device const T* y [[buffer(1)]],
                          device T* z [[buffer(2)]], constant float& a [[buffer(3)]],
                          constant float& b [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  // Match separate MLX multiply/multiply/add, including FP16 product rounding.
  T px = T(T(a) * x[i]);
  T py = T(T(b) * y[i]);
  z[i] = T(px + py);
}
template [[host_name("mlxboost_axpby_f32")]] kernel decltype(mlxboost_axpby<float>) mlxboost_axpby<float>;
template [[host_name("mlxboost_axpby_f16")]] kernel decltype(mlxboost_axpby<half>) mlxboost_axpby<half>;
