// MLX extension API adaptation. Copyright © 2023-2025 Apple Inc. MIT; see ../MLX_LICENSE.txt.
#pragma once
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlxboost {
namespace mx = mlx::core;
mx::array axpby(const mx::array&, const mx::array&, float, float, int, mx::StreamOrDevice);
int max_threads(mx::Dtype, mx::StreamOrDevice);
class Axpby final : public mx::Primitive {
 public:
  Axpby(mx::Stream s, float a, float b, int t) : Primitive(s), a_(a), b_(b), threads_(t) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  std::vector<mx::array> jvp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                            const std::vector<int>&) override;
  std::vector<mx::array> vjp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                            const std::vector<int>&, const std::vector<mx::array>&) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>&, const std::vector<int>&) override;
  const char* name() const override { return "MLXBoostAxpbyInferenceOnly"; }
  bool is_equivalent(const mx::Primitive&) const override;
 private:
  float a_, b_;
  int threads_;
};
}  // namespace mlxboost
