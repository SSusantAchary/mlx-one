// MLX allocator/encoder API adaptation. Apple MLX MIT notice in ../MLX_LICENSE.txt.
#include "gemv.h"
#include <dlfcn.h>
#include <filesystem>
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlxboost {
namespace {
void encoding(int bits, int group) {
  if ((bits != 4 && bits != 8) || (group != 32 && group != 64 && group != 128))
    throw std::invalid_argument("unsupported_quantization: affine Q4/Q8 groups 32/64/128 only");
}
auto gemv_pipeline(mx::Dtype dtype, int bits, int group, mx::Stream stream) {
  encoding(bits, group);
  if (dtype != mx::float16 && dtype != mx::bfloat16)
    throw std::invalid_argument("unsupported_dtype: FP16/BF16 GEMV only");
  if (stream.device.type != mx::Device::gpu)
    throw std::invalid_argument("unsupported_stream: GPU required");
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&encoding), &info))
    throw std::runtime_error("cannot locate bundled GEMV Metal library");
  auto directory = std::filesystem::path(info.dli_fname).parent_path().string();
  auto& device = mx::metal::device(stream.device);
  auto library = device.get_library("mlxboost", directory);
  auto name = std::string("mlxboost_gemv_q") + std::to_string(bits) +
      (dtype == mx::float16 ? "_f16_g" : "_bf16_g") + std::to_string(group);
  auto kernel = device.get_kernel(name, library);
  if (kernel->threadExecutionWidth() != 32)
    throw std::invalid_argument("unsupported_chip: 32-lane SIMD required");
  return kernel;
}
[[noreturn]] void no_gemv_transform() {
  throw std::invalid_argument("G03 quantized GEMV is inference-only: grad/vjp/jvp/vmap unsupported");
}
}

int gemv_max_threads(mx::Dtype dtype, int bits, int group, mx::StreamOrDevice s) {
  return static_cast<int>(gemv_pipeline(dtype, bits, group, mx::to_stream(s))
                          ->maxTotalThreadsPerThreadgroup());
}

mx::array quantized_gemv(const mx::array& x, const mx::array& w, const mx::array& scales,
                        const mx::array& biases, int bits, int group, int threads,
                        mx::StreamOrDevice stream, std::shared_ptr<GemvTrace> trace) {
  encoding(bits, group);
  auto s = mx::to_stream(stream);
  if (s.device.type != mx::Device::gpu)
    throw std::invalid_argument("unsupported_stream: GPU required");
  if ((x.dtype() != mx::float16 && x.dtype() != mx::bfloat16) ||
      scales.dtype() != x.dtype() || biases.dtype() != x.dtype() || w.dtype() != mx::uint32)
    throw std::invalid_argument("unsupported_dtype: FP16/BF16 matching metadata and uint32 weights");
  if (x.ndim() < 1 || w.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2)
    throw std::invalid_argument("unsupported_shape: vector and packed matrix required");
  int k = x.shape(-1), n = w.shape(0);
  if (k < group || k % group || x.size() != static_cast<size_t>(k) || n < 1 ||
      w.shape(1) != k / (32 / bits) || scales.shape(0) != n ||
      scales.shape(1) != k / group || biases.shape() != scales.shape())
    throw std::invalid_argument("unsupported_shape: B1 vector/packed affine dimensions mismatch");
  if (threads != 64 && threads != 128 && threads != 256)
    throw std::invalid_argument("unsupported_configuration: threadgroup must be 64/128/256");
  for (const auto* a : {&x, &w, &scales, &biases})
    if (!a->flags().row_contiguous)
      throw std::invalid_argument("unsupported_layout: row contiguous arrays required");
  auto shape = x.shape(); shape.back() = n;
  return mx::array(shape, x.dtype(), std::make_shared<QuantizedGemv>(s, bits, group, threads, trace),
                   {x, w, scales, biases});
}

void QuantizedGemv::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::invalid_argument("unsupported_stream: use MLX reference on CPU");
}
void QuantizedGemv::eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  for (const auto& a : inputs)
    if (!a.flags().row_contiguous)
      throw std::invalid_argument("unsupported_layout: realized GEMV input is strided");
  auto& out = outputs[0];
  auto kernel = gemv_pipeline(out.dtype(), bits_, group_, stream());
  if (threads_ > static_cast<int>(kernel->maxTotalThreadsPerThreadgroup()))
    throw std::invalid_argument("unsupported_configuration: device threadgroup limit");
  // Output only. No dequantized weight, temporary matrix or CPU copy is allocated.
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 4; ++i) enc.set_input_array(inputs[i], i);
  enc.set_output_array(out, 4);
  uint32_t k = inputs[0].shape(-1), n = out.shape(-1);
  enc.set_bytes(k, 5); enc.set_bytes(n, 6);
  size_t outputs_per_group = (threads_ / 32) * 4;
  enc.dispatch_threadgroups(MTL::Size((n + outputs_per_group - 1) / outputs_per_group, 1, 1),
                             MTL::Size(threads_, 1, 1));
  // Counts actual primitive evaluation/encoding, not Python graph construction.
  // Each request/layer owns its trace; read only after mx.eval + synchronize.
  if (trace_) trace_->dispatches.fetch_add(1, std::memory_order_relaxed);
}
std::vector<mx::array> QuantizedGemv::jvp(const std::vector<mx::array>&,
    const std::vector<mx::array>&, const std::vector<int>&) { no_gemv_transform(); }
std::vector<mx::array> QuantizedGemv::vjp(const std::vector<mx::array>&,
    const std::vector<mx::array>&, const std::vector<int>&,
    const std::vector<mx::array>&) { no_gemv_transform(); }
std::pair<std::vector<mx::array>, std::vector<int>> QuantizedGemv::vmap(
    const std::vector<mx::array>&, const std::vector<int>&) { no_gemv_transform(); }
bool QuantizedGemv::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const QuantizedGemv&>(other);
  return bits_ == rhs.bits_ && group_ == rhs.group_ && threads_ == rhs.threads_
      && trace_.get() == rhs.trace_.get();
}
}
