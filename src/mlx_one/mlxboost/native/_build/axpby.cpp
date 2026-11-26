// MLX extension API adaptation. Copyright © 2023-2025 Apple Inc. MIT; see ../MLX_LICENSE.txt.
#include "axpby.h"
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlxboost {
namespace {
std::string library_dir() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&library_dir), &info))
    throw std::runtime_error("MLXBoost cannot locate its bundled Metal library");
  return std::filesystem::path(info.dli_fname).parent_path().string();
}
auto pipeline(mx::Dtype dtype, mx::Stream s) {
  if (dtype != mx::float16 && dtype != mx::float32)
    throw std::invalid_argument("unsupported_dtype: expected float16 or float32");
  auto& dev = mx::metal::device(s.device);
  auto lib = dev.get_library("mlxboost", library_dir());
  return dev.get_kernel(dtype == mx::float16 ? "mlxboost_axpby_f16" : "mlxboost_axpby_f32", lib);
}
[[noreturn]] void no_transform() {
  throw std::invalid_argument("MLXBoost axpby is inference-only: grad/vjp/jvp/vmap unsupported");
}
}

int max_threads(mx::Dtype dtype, mx::StreamOrDevice stream) {
  auto s = mx::to_stream(stream);
  if (s.device.type != mx::Device::gpu)
    throw std::invalid_argument("unsupported_stream: GPU stream required");
  return static_cast<int>(pipeline(dtype, s)->maxTotalThreadsPerThreadgroup());
}

mx::array axpby(const mx::array& x, const mx::array& y, float a, float b, int threads,
                mx::StreamOrDevice stream) {
  auto s = mx::to_stream(stream);
  if (s.device.type != mx::Device::gpu)
    throw std::invalid_argument("unsupported_stream: use MLX reference for CPU");
  if (x.dtype() != y.dtype() || (x.dtype() != mx::float16 && x.dtype() != mx::float32))
    throw std::invalid_argument("unsupported_dtype: matching float16/float32 required");
  if (x.shape() != y.shape() || x.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("unsupported_shape: equal shapes and <= UINT32_MAX elements");
  if (!x.flags().row_contiguous || !y.flags().row_contiguous)
    throw std::invalid_argument("unsupported_layout: row contiguous inputs required");
  if (threads != 64 && threads != 128 && threads != 256)
    throw std::invalid_argument("unsupported_configuration: threadgroup must be 64,128,256");
  return mx::array(x.shape(), x.dtype(), std::make_shared<Axpby>(s, a, b, threads), {x, y});
}

void Axpby::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::invalid_argument("unsupported_stream: MLXBoost demo requires GPU");
}
void Axpby::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  const auto& x = inputs[0]; const auto& y = inputs[1]; auto& out = outputs[0];
  // Recheck realized layout: lazy views may acquire strides during evaluation.
  if (!x.flags().row_contiguous || !y.flags().row_contiguous)
    throw std::invalid_argument("unsupported_layout: realized input is not row contiguous");
  out.set_data(mx::allocator::malloc(out.nbytes()));
  if (!out.size()) return;
  auto kernel = pipeline(out.dtype(), stream());
  if (threads_ > static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()))
    throw std::invalid_argument("unsupported_configuration: device threadgroup limit");
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0); enc.set_input_array(y, 1); enc.set_output_array(out, 2);
  enc.set_bytes(a_, 3); enc.set_bytes(b_, 4);
  enc.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(std::min<size_t>(threads_, out.size()), 1, 1));
}
std::vector<mx::array> Axpby::jvp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                                const std::vector<int>&) { no_transform(); }
std::vector<mx::array> Axpby::vjp(const std::vector<mx::array>&, const std::vector<mx::array>&,
                                const std::vector<int>&, const std::vector<mx::array>&) { no_transform(); }
std::pair<std::vector<mx::array>, std::vector<int>> Axpby::vmap(
    const std::vector<mx::array>&, const std::vector<int>&) { no_transform(); }
bool Axpby::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const Axpby&>(other);
  return a_ == rhs.a_ && b_ == rhs.b_ && threads_ == rhs.threads_;
}
}  // namespace mlxboost
