"""Registered native causal adapters; architecture details stay out of scheduling."""
from __future__ import annotations

from dataclasses import dataclass

from mlx_one.engine.adapters import NativeModelAdapter
from mlx_one.engine.contracts import BatchPlan, WorkItem


@dataclass(frozen=True)
class CausalRowOutput:
    request_id: str
    logits: object
    committed_tokens: int


class CausalArchitectureAdapter(NativeModelAdapter):
    def describe(self, bundle):
        if bundle is not self.bundle:
            raise ValueError('adapter is bound to a different model instance')
        return self.spec

    def prepare(self, payload):
        from mlx_one.engine.runners import CausalLMRunner

        return CausalLMRunner(self.bundle).prepare(payload)

    def estimate(self, prepared, plan):
        if plan is not None and plan.topology_digest != self.plan.topology_digest:
            raise ValueError('adapter estimate topology mismatch')
        output = prepared.request.max_tokens if prepared.request is not None else 0
        return self.memory_estimate(prepared.prompt_len + output)

    def pack(self, work):
        if not work or any(not isinstance(item, WorkItem) for item in work):
            raise ValueError('adapter pack requires typed work items')
        key = work[0].compatibility
        if any(item.compatibility != key or item.payload is None for item in work):
            raise ValueError('adapter batch compatibility or payload mismatch')
        if (key.model_instance != str(id(self.bundle))
                or key.adapter_version != self.spec.adapter_version
                or key.precision_policy != self.spec.quantization_digest
                or key.state_schema != self.plan.topology_digest):
            raise ValueError("adapter execution identity mismatch")
        requests = tuple(item.payload for item in work)
        if any(request.bundle is not self.bundle or request.request_id != item.request_id
               or request.model_epoch != key.model_epoch
               for request, item in zip(requests, work, strict=True)):
            raise ValueError('adapter request ownership mismatch')
        return requests

    def execute(self, packed, state):
        if not isinstance(state, BatchPlan) or tuple(r.request_id for r in packed) != tuple(
                row.request_id for row in state.sequences):
            raise ValueError('adapter execution requires aligned batch plan')
        for request, row in zip(packed, state.sequences, strict=True):
            if (request.cache[0].offset != row.start_pos or request.phase != row.phase
                    or len(request.token_ids) != row.input_token_count):
                raise ValueError('adapter scheduled boundary mismatch')
        return self._runner_for_execution(packed).execute(packed)

    def _runner_for_execution(self,packed=()):
        from mlx_one.engine.batching import BatchedDenseRunner
        from mlx_one.engine.paged import PagedKVCache, PagedRunner

        if packed and isinstance(packed[0].cache[0],PagedKVCache):
            return PagedRunner(self.bundle)
        return BatchedDenseRunner(self.bundle)

    def unpack(self, output):
        return tuple(CausalRowOutput(request_id, logits, count) for request_id, logits, count in
                     zip(output.request_ids, output.logits,
                         output.committed_token_counts, strict=True))

    def prefill(self, packed, state):
        if any(request.phase != 'prefill' for request in packed):
            raise ValueError('prefill adapter received another stage')
        return self.execute(packed, state)

    def decode(self, packed, state):
        if any(request.phase != 'decode' for request in packed):
            raise ValueError('decode adapter received another stage')
        return self.execute(packed, state)


class GPT2Adapter(CausalArchitectureAdapter):
    supported_architectures = frozenset({'gpt2'})


class LlamaAdapter(CausalArchitectureAdapter):
    supported_architectures = frozenset({'llama'})


class Qwen2Adapter(CausalArchitectureAdapter):
    supported_architectures = frozenset({'qwen2'})


class Qwen3Adapter(CausalArchitectureAdapter):
    qk_normalization = True
    supported_architectures = frozenset({'qwen3'})


class OpenELMAdapter(CausalArchitectureAdapter):
    supported_architectures = frozenset({'openelm'})


class Qwen2MoeAdapter(CausalArchitectureAdapter):
    supported_architectures = frozenset({'qwen2_moe'})


class LFM2Adapter(CausalArchitectureAdapter):
    qk_normalization = True
    supported_architectures = frozenset({'lfm2'})
    supported_states = frozenset({'full_kv','sliding_kv','conv'})

    def prepare(self, payload):
        from mlx_one.engine.cache import CacheBundle
        from mlx_one.engine.components import hybrid_components

        state=super().prepare(payload)
        state.cache.release()
        state.cache=CacheBundle(hybrid_components(self.bundle))
        return state

    def _runner_for_execution(self,packed=()):
        from mlx_one.engine.hybrid import HybridRunner

        return HybridRunner()


class LFM2MoeAdapter(LFM2Adapter):
    supported_architectures = frozenset({'lfm2_moe'})


class Qwen35Adapter(LFM2Adapter):
    supported_architectures = frozenset({'qwen3_5'})
    supported_states = frozenset({'full_kv','recurrent'})

    def __init__(self,bundle,plan,*,allow_bfloat16=False,allow_hybrid=False):
        super().__init__(bundle,plan,allow_bfloat16=False,allow_hybrid=allow_hybrid)

    def prepare(self,payload):
        if any(name in payload.payload for name in ('images','pixel_values','video','audio')):
            raise ValueError('Qwen3.5 compound adapter currently supports text only')
        return super().prepare(payload)

    def additional_transient_bytes(self,*,batch,chunk):
        import math

        carries = sum(math.prod(shape)*4 for spec in self.plan.state_specs
                      if spec.kind=='recurrent' for shape in spec.state_shapes)
        config=self.text_config
        width=2*config.linear_num_key_heads*config.linear_key_head_dim
        width+=config.linear_num_value_heads*config.linear_value_head_dim
        return batch*(carries*3+chunk*width*16)
