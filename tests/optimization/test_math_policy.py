import dataclasses
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from mlx_one.engine.optimization_policy import (
    REFERENCE_MATH,
    NativeMathPolicy,
    current_math,
    math_execution,
)


@pytest.mark.parametrize("options", [{"attention": "unknown"}, {"projection": "fp16_replica"},
    {"logits": "no_logits"}, {"prefill_chunk_size": True}, {"prefill_chunk_size": 1},
    {"hybrid": "ssm"}, {"enable_compile": 1}])
def test_invalid_policy_rejected(options):
    with pytest.raises(ValueError):
        NativeMathPolicy(**options)


def test_policy_immutable_private_and_not_a_qualification_claim():
    policy = NativeMathPolicy.optimized()
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.logits = "all"
    assert current_math().policy == REFERENCE_MATH
    with math_execution(policy, family="qwen3", phase="prefill"):
        assert current_math().policy is policy
        with math_execution(REFERENCE_MATH):
            assert current_math().policy == REFERENCE_MATH
        assert current_math().family == "qwen3"
    assert current_math().policy == REFERENCE_MATH
    assert policy.to_dict()["qualified"] is False


def test_context_restored_after_failure_and_explicitly_propagated_to_owner():
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError), math_execution(NativeMathPolicy.optimized()):
            ctx = copy_context()
            assert executor.submit(ctx.run, current_math).result().policy.logits == "last_only"
            # An unrelated job on that thread has no inherited request policy.
            assert executor.submit(current_math).result().policy == REFERENCE_MATH
            raise RuntimeError("injected")
    assert current_math().policy == REFERENCE_MATH
