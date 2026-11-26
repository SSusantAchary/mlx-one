// MLX extension API adaptation. Apple MLX MIT notice in ../MLX_LICENSE.txt.
#pragma once
#include "axpby.h"
#include <atomic>

namespace mlxboost {
struct GemvTrace {
  std::atomic<uint64_t> dispatches{0};
};
mx::array quantized_gemv(const mx::array&, const mx::array&, const mx::array&,
                        const mx::array&, int, int, int, mx::StreamOrDevice,
                        std::shared_ptr<GemvTrace>);
int gemv_max_threads(mx::Dtype, int, int, mx::StreamOrDevice);
class QuantizedGemv final : public mx::Primitive {
 public:
  QuantizedGemv(mx::Stream s, int bits, int group, int threads, std::shared_ptr<GemvTrace> trace)
      : Primitive(s), bits_(bits), group_(group), threads_(threads), trace_(std::move(trace)) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  std::vector<mx::array> jvp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                            const std::vector<int>&) override;
  std::vector<mx::array> vjp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                            const std::vector<int>&, const std::vector<mx::array>&) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>&, const std::vector<int>&) override;
  const char* name() const override { return "MLXBoostAffineGemvInferenceOnly"; }
  bool is_equivalent(const mx::Primitive&) const override;
 private:
  int bits_, group_, threads_;
  std::shared_ptr<GemvTrace> trace_;
};
}
