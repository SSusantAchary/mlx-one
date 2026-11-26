// Read-only graph inputs, request-owned working storage and fresh output buffers.
// The Python caller proves each captured geometry before serving its results.
#import "replay_trace.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "mlx/array.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace nb = nanobind;
namespace mx = mlx::core;
namespace {
using namespace mlxboost::replay;
using Arrays = std::vector<mx::array>;

void require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
void check(const std::string& error) { require(error.empty(), error); }
id<MTLBuffer> buffer(const mx::array& a) {
  require(a.is_available() && a.size(), "replay requires materialized nonempty arrays");
  return (__bridge id<MTLBuffer>)const_cast<void*>(a.buffer().ptr());
}
id<MTLCommandQueue> queue() {
  static id<MTLCommandQueue> shared = [] {
    auto q = [MTLCreateSystemDefaultDevice() newCommandQueue];
    if (q != nil) register_own_queue((__bridge void*)q);
    return q;
  }();
  return shared;
}
mx::array storage(size_t n) {
  require(n && n <= INT32_MAX, "unsupported replay allocation length");
  mx::array a(mx::Shape{static_cast<int>(n)}, mx::uint8, nullptr, {});
  a.set_data(mx::allocator::malloc(n));
  a.set_status(mx::array::Status::available);
  return a;
}

struct Layout {
  mx::Shape shape;
  mx::Strides strides;
  mx::Dtype dtype;
  mx::array::Flags flags;
  size_t data_size;
  int64_t offset;
  int slot;
  Layout(const mx::array& a, int s)
      : shape(a.shape()), strides(a.strides()), dtype(a.dtype()), flags(a.flags()),
        data_size(a.data_size()), offset(a.offset()), slot(s) {}
};
struct Slot {
  size_t bytes = 0;
  int input = -1;
  bool used = false, read_first = false, written = false, output = false;
  std::optional<mx::array> backing;
  std::vector<uint8_t> seed;
};
struct Launch {
  id<MTLComputePipelineState> pipeline;
  std::vector<TraceSlot> args;
  std::vector<std::pair<uint8_t, uint64_t>> shared;
  MTLSize grid, group;
  bool threads, barrier;
};
MTLSize size3(const uint64_t* s) { return MTLSizeMake(s[0], s[1], s[2]); }

class Program {
 public:
  std::vector<Launch> launches;
  size_t working_bytes = 0;

  Program(const Trace& t, const Arrays& in, const Arrays& out,
          const Arrays& weights, size_t limit) : bytes(t.bytes), slots(t.buffers.size()) {
    require(t.unsupported.empty() && !t.steps.empty(),
            "Metal recording declined: " + (t.unsupported.empty() ? "empty" : t.unsupported));
    std::unordered_map<const void*, int> ids;
    for (size_t i = 0; i < slots.size(); ++i) {
      ids[(__bridge const void*)t.buffers[i].buffer] = i;
      slots[i].bytes = t.buffers[i].length;
    }
    auto find = [&](const mx::array& a) {
      auto it = ids.find((__bridge const void*)buffer(a));
      return it == ids.end() ? -1 : it->second;
    };
    for (const auto& a : in) {
      int s = find(a);
      if (s >= 0 && slots[s].input < 0) slots[s].input = inputs.size();
      inputs.emplace_back(a, s);
    }
    for (const auto& a : out) {
      int s = find(a);
      require(s >= 0, "output missing from Metal recording");
      slots[s].output = true;
      outputs.emplace_back(a, s);
    }
    for (const auto& a : weights) {
      int s = find(a);
      if (s >= 0) slots[s].backing = a;
    }
    for (const auto& step : t.steps) {
      auto* k = kernel_for_pipeline(step.pso);
      require(k->reflected && k->problem.empty() && step.indirect_buffer < 0 &&
              k->name != "fence_wait" && k->name != "fence_update" && k->name != "input_coherent",
              "unsupported Metal pipeline: " + k->name);
      Launch launch{k->pso, {}, {}, size3(step.grid), size3(step.group),
                    bool(step.threads), step.barrier_before};
      bool bound[kMaxSlots] = {};
      // Determine first access across the whole dispatch before marking its writes.
      for (size_t j = step.slots_pos; j < step.slots_pos + step.slots_len; ++j) {
        auto arg = t.slots[j];
        if (k->kind[arg.index] != 1) continue;
        bound[arg.index] = true;
        launch.args.push_back(arg);
        if (arg.kind == 1 && k->access[arg.index] == 'r' && !slots[arg.buffer].written)
          slots[arg.buffer].read_first = true;
      }
      for (const auto& arg : launch.args) {
        if (arg.kind != 1) continue;
        auto& s = slots[arg.buffer];
        s.used = true;
        s.written |= k->access[arg.index] != 'r';
      }
      for (size_t j = step.tg_pos; j < step.tg_pos + step.tg_len; ++j) {
        auto mem = t.tg_memory[j];
        if (k->kind[mem.first] == 2) {
          bound[mem.first] = true;
          launch.shared.push_back(mem);
        }
      }
      for (int j = 0; j < kMaxSlots; ++j)
        require(!k->kind[j] || bound[j], "missing argument in " + k->name);
      launches.push_back(std::move(launch));
    }
    for (size_t i = 0; i < slots.size(); ++i) {
      auto& s = slots[i];
      if (!s.used && !s.output) continue;
      if (s.input >= 0 || s.backing) {
        require(!s.written && !(s.output && s.input < 0), "recorded graph writes a shared input");
        continue;
      }
      working_bytes += s.bytes;
      require(working_bytes <= limit, "Metal replay exceeds request workspace budget");
      if (s.read_first || !s.written) {
        const auto& b = t.buffers[i];
        require(b.has_snapshot && (s.written || !b.changed), "undeclared or changing Metal input");
        s.seed = b.snapshot;
      }
    }
    for (auto& s : slots) {
      if (!s.used || s.input >= 0 || s.backing || s.output) continue;
      s.backing = storage(s.bytes);
      if (!s.seed.empty() && !s.written) {
        std::memcpy(s.backing->data<uint8_t>(), s.seed.data(), s.bytes);
        s.seed.clear();
      }
    }
  }

  Arrays run(const Arrays& in) {
    std::lock_guard guard(mutex);
    @autoreleasepool {
      require(in.size() == inputs.size(), "replay input count changed");
      require(wait_for_mlx(2000000000ull), "MLX stream did not drain before Metal replay");
      std::vector<id<MTLBuffer>> buffers(slots.size(), nil);
      std::vector<int64_t> deltas(slots.size(), 0);
      for (size_t i = 0; i < in.size(); ++i) {
        const auto& a = in[i];
        const auto& spec = inputs[i];
        if (spec.slot < 0 || (!slots[spec.slot].used && !slots[spec.slot].output)) continue;
        bool matches = a.shape() == spec.shape && a.dtype() == spec.dtype;
        for (size_t axis = 0; matches && axis < spec.shape.size(); ++axis)
          matches = spec.shape[axis] == 1 || a.strides()[axis] == spec.strides[axis];
        require(matches, "replay input layout changed at index " + std::to_string(i));
        int s = spec.slot;
        auto b = buffer(a);
        int64_t delta = a.offset() - spec.offset;
        require(!buffers[s] || (buffers[s] == b && deltas[s] == delta), "replay alias changed");
        buffers[s] = b;
        deltas[s] = delta;
      }
      std::vector<std::optional<mx::array>> fresh(slots.size());
      for (size_t i = 0; i < slots.size(); ++i) {
        auto& s = slots[i];
        if (!s.used && !s.output) continue;
        if (s.input >= 0) continue;
        if (s.output) fresh[i] = storage(s.bytes);
        auto& a = s.output ? fresh[i] : s.backing;
        buffers[i] = buffer(*a);
        if (!s.seed.empty()) std::memcpy(a->data<uint8_t>(), s.seed.data(), s.bytes);
      }
      // Validate every relocation before encoding. Input shape and strides above
      // also guarantee that the same logical array extent exists on this call.
      for (auto& launch : launches) for (const auto& arg : launch.args) {
        if (arg.kind != 1) continue;
        int64_t off = static_cast<int64_t>(arg.offset) + deltas[arg.buffer];
        require(off >= 0 && static_cast<uint64_t>(off) < buffers[arg.buffer].length,
                "replay buffer offset out of range");
      }
      auto cb = [queue() commandBuffer];
      auto enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      require(enc != nil, "Metal replay encoder unavailable");
      id<MTLComputePipelineState> pipeline = nil;
      __unsafe_unretained id<MTLBuffer> bound[kMaxSlots] = {};
      NSUInteger offsets[kMaxSlots] = {};
      for (auto& launch : launches) {
        if (launch.barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        if (pipeline != launch.pipeline) {
          pipeline = launch.pipeline;
          [enc setComputePipelineState:pipeline];
        }
        for (const auto& arg : launch.args) {
          const auto index = arg.index;
          if (arg.kind == 1) {
            auto b = buffers[arg.buffer];
            NSUInteger off = arg.offset + deltas[arg.buffer];
            if (bound[index] != b) [enc setBuffer:b offset:off atIndex:index];
            else if (offsets[index] != off) [enc setBufferOffset:off atIndex:index];
            bound[index] = b;
            offsets[index] = off;
          } else {
            [enc setBytes:bytes.data() + arg.bytes_pos length:arg.bytes_len atIndex:index];
            bound[index] = nil;
          }
        }
        for (auto [idx, len] : launch.shared) [enc setThreadgroupMemoryLength:len atIndex:idx];
        if (launch.threads) [enc dispatchThreads:launch.grid threadsPerThreadgroup:launch.group];
        else [enc dispatchThreadgroups:launch.grid threadsPerThreadgroup:launch.group];
      }
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      require(cb.status == MTLCommandBufferStatusCompleted, "Metal replay command buffer failed");
      Arrays result;
      for (const auto& spec : outputs) {
        const auto& s = slots[spec.slot];
        const auto& owner = s.input >= 0 ? in[s.input] : *fresh[spec.slot];
        mx::array a(spec.shape, spec.dtype, nullptr, {});
        a.copy_shared_buffer(owner, spec.strides, spec.flags, spec.data_size,
                             (spec.offset + deltas[spec.slot] - owner.offset()) / static_cast<int64_t>(a.itemsize()));
        a.set_status(mx::array::Status::available);
        result.push_back(std::move(a));
      }
      return result;
    }
  }
 private:
  std::mutex mutex;
  std::vector<uint8_t> bytes;
  std::vector<Slot> slots;
  std::vector<Layout> inputs, outputs;
};
}
void bind_replay(nb::module_& m) {
  m.def("install_replay", [] {
    static const std::string result = install_hooks(MTLCreateSystemDefaultDevice());
    check(result);
  });
  m.def("begin_replay_capture", [] { check(record_begin("Inkling DSpark")); });
  m.def("abort_replay_capture", [] {
    if (recording_on_this_thread()) { std::unique_ptr<Trace> ignored; check(record_end(&ignored)); }
  });
  m.def("finish_replay_capture", [](const Arrays& in, const Arrays& out,
                                    const Arrays& weights, size_t limit) {
    @autoreleasepool {
      std::unique_ptr<Trace> t;
      check(record_end(&t));
      return std::make_shared<Program>(*t, in, out, weights, limit);
    }
  });
  nb::class_<Program>(m, "MetalReplay")
      .def("run", &Program::run, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("dispatches", [](const Program& p) { return p.launches.size(); })
      .def_ro("workspace_bytes", &Program::working_bytes);
}
