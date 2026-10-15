"""Exact-boundary Qwen3.5 compound state; no generic recurrent math is implied."""
from __future__ import annotations

from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.components import ComponentCapabilities

COMPOUND_VERSION = 'qwen35-gated-delta-v1'


class CompoundLinearComponent:
    kind = 'recurrent'
    checkpoint_semantics = COMPOUND_VERSION

    def __init__(self, kernel_size, state_shapes=(), state_dtypes=()):
        self.kernel_size = kernel_size
        self.state_shapes, self.state_dtypes = tuple(state_shapes), tuple(state_dtypes)
        self.conv_state = self.recurrent_state = None
        self._offset = 0

    @property
    def offset(self):
        return self._offset

    @property
    def committed_position(self):
        return self.offset

    @property
    def nbytes(self):
        return array_nbytes((self.conv_state,self.recurrent_state))

    def capabilities(self):
        return ComponentCapabilities(can_batch=True,can_snapshot=True,can_share_prefix=True,
                                     transition_math=True)

    def batch_key(self, width):
        return (COMPOUND_VERSION,self.kernel_size,self.state_shapes,self.state_dtypes,
                self.offset,width)

    def prefix(self, values):
        import mlx.core as mx

        width = self.kernel_size - 1
        prefix = (mx.zeros((values.shape[0],width,values.shape[2]),dtype=values.dtype)
                  if self.conv_state is None else self.conv_state)
        combined = mx.concatenate((prefix,values),axis=1)
        self.conv_state = combined[:, -width:] if width else combined[:, :0]
        self._offset += values.shape[1]
        return combined

    def _copy(self, value):
        if value is None:
            return None
        import mlx.core as mx

        result = mx.zeros(value.shape,dtype=value.dtype)
        result[...] = value
        return result

    def clone(self):
        result = type(self)(self.kernel_size,self.state_shapes,self.state_dtypes)
        result.conv_state = self._copy(self.conv_state)
        result.recurrent_state = self._copy(self.recurrent_state)
        result._offset = self.offset
        return result

    snapshot = clone

    def batch_select(self, indices):
        result = type(self)(self.kernel_size,self.state_shapes,self.state_dtypes)
        result.conv_state = self._copy(None if self.conv_state is None
                                       else self.conv_state[list(indices)])
        result.recurrent_state = self._copy(None if self.recurrent_state is None
                                            else self.recurrent_state[list(indices)])
        result._offset = self.offset
        return result

    @classmethod
    def merge(cls, caches):
        import mlx.core as mx

        if not caches or len({c.batch_key(1) for c in caches}) != 1:
            raise ValueError('compound state requires equal-offset layout buckets')
        first = caches[0]
        result = cls(first.kernel_size,first.state_shapes,first.state_dtypes)
        for name in ('conv_state','recurrent_state'):
            values = [getattr(c,name) for c in caches]
            if any(v is None for v in values) and not all(v is None for v in values):
                raise ValueError('compound state is incomplete')
            if values[0] is not None:
                setattr(result,name,mx.concatenate(values,axis=0))
        result._offset = first.offset
        return result

    def validate(self, spec):
        if (spec.kind != self.kind or spec.checkpoint_semantics != COMPOUND_VERSION
                or tuple(spec.state_shapes) != self.state_shapes or self.offset < 0
                or len(self.state_shapes) != 2 or tuple(spec.state_dtypes) != self.state_dtypes):
            raise ValueError('compound topology or committed position mismatch')
        for index,name in enumerate(('conv_state','recurrent_state')):
            value = getattr(self,name)
            if value is None:
                if self.offset:
                    raise ValueError('compound state is missing a committed tensor')
                continue
            if (tuple(value.shape) != self.state_shapes[index]
                    or str(value.dtype).rsplit('.',1)[-1] != self.state_dtypes[index]):
                raise ValueError('compound state shape or dtype mismatch')

    def trim(self, count):
        if count:
            raise ValueError('compound state requires checkpoint restoration and replay')
        return True

    def reset(self):
        self.conv_state = self.recurrent_state = None
        self._offset = 0

    release = reset
    dispose = reset


def compound_layout(config, dtype):
    width = 2*config.linear_num_key_heads*config.linear_key_head_dim
    width += config.linear_num_value_heads*config.linear_value_head_dim
    return ((1,config.linear_conv_kernel_dim-1,width),
            (1,config.linear_num_value_heads,config.linear_key_head_dim,
             config.linear_value_head_dim)), (
                dtype,'float32')
