"""Direct segmented Metal attention with bounded bindings and online softmax."""

from __future__ import annotations

from functools import lru_cache


def page_bindings(rows):
    """Validate identities/layouts before exporting segment offsets to the GPU."""
    from mlx_one.engine.paged import KERNEL_VERSION

    pages, starts, indices = [], [], {}
    for row, cache in enumerate(rows):
        if sum(r.block.token_count for r in cache._blocks) != cache.offset:
            raise ValueError("paged table committed length mismatch")
        for index, ref in enumerate(cache._blocks):
            block = ref.block  # validates epoch/generation and owning reference
            layout = block.layout
            if (ref.pool is not cache.pool or layout is None or layout.version != 3
                    or layout.layer_id != cache.layer_id
                    or layout.page_tokens != cache.block_tokens
                    or layout.key_dtype != cache.dtype or layout.value_dtype != cache.dtype
                    or layout.kernel_version != KERNEL_VERSION
                    or layout.quantization_group_size != 32
                    or not layout.num_kv_heads or not layout.head_dim
                    or layout.quantization_bits not in {None, cache.bits}):
                raise ValueError("invalid paged arena ownership or layout")
            bits = layout.quantization_bits
            heads, dim, width = layout.num_kv_heads, layout.head_dim, layout.page_tokens
            padded = (dim + 31) // 32 * 32
            expected = ([(1, heads, width, dim)] * 2 if bits is None else
                        [(1, heads, width, padded // (2 if bits == 4 else 1)),
                         (1, heads, width, padded // 32), (1, heads, width, padded // 32)] * 2)
            dtypes = ([cache.dtype] * 2 if bits is None
                      else ["uint8", "float16", "float16"] * 2)
            if (not 0 < block.token_count <= width or len(block.state) != len(expected)
                    or bits is not None and (not block.immutable or block.token_count != width)
                    or any(tuple(t.shape) != shape or str(t.dtype).rsplit(".", 1)[-1] != dtype
                           for t, shape, dtype in zip(block.state, expected, dtypes, strict=True))):
                raise ValueError("paged arena tensor layout mismatch")
            # Repeated physical pages at different logical positions remain distinct bindings.
            key = (ref.block_id, index)
            if key not in indices:
                indices[key] = len(pages)
                pages.append(block)
                starts.append([-1] * len(rows))
            starts[indices[key]][row] = index * cache.block_tokens
    return pages, starts


def _read_expression(name, bits, dim):
    if bits is None:
        return f"float({name}[kh * P * D + t * D + d])"
    stride = (dim + 31) // 32 * 32
    index = f"(kh * P + t) * {stride} + d"
    encoded = (f"(({name}[({index}) / 2] >> ((d % 2) * 4)) & 15)"
               if bits == 4 else f"{name}[{index}]")
    group = f"(kh * P + t) * {stride // 32} + d / 32"
    return f"(float({encoded}) * float({name}s[{group}]) + float({name}b[{group}]))"


def kernel_source(formats, dim, row_count=1):
    """Bounded segment tensors plus persistent per-row table chunks."""
    names = ["q", "segment_ids", "positions", "widths", "lengths", "scale"]
    names.extend(f"table{i}" for i in range(row_count))
    table = "table0"
    for i in range(1, row_count):
        table = f"(row == {i} ? table{i} : {table})"
    blocks = []
    for i, bits in enumerate(formats):
        for name in (f"k{i}", f"v{i}"):
            names.extend([name] if bits is None else [name, name + "s", name + "b"])
        k = _read_expression(f"k{i}", bits, dim).replace("kh * P", "(slot * HK + kh) * P")
        v = _read_expression(f"v{i}", bits, dim).replace("kh * P", "(slot * HK + kh) * P")
        blocks.append(f"""
        for (uint page = 0; page < 16; ++page) {{
            int sid = table[page * 4], slot = table[page * 4 + 1];
            int start = table[page * 4 + 2], count = table[page * 4 + 3];
            if (sid != segment_ids[{i}] || qi >= widths[row]) continue;
            for (int t = 0; t < count; ++t) {{
                if (start + t >= lengths[row] || start + t > positions[row * Q + qi]) break;
                float kval = d < D ? {k} : 0.0f;
                dot[d] = d < D ? float(q[((row * H + head) * Q + qi) * D + d]) * kval : 0;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint s = W / 2; s > 0; s /= 2) {{
                    if (d < s) dot[d] += dot[d + s];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}
                float score = dot[0] * scale[0];
                float next_m = max(m, score);
                float alpha = exp(m - next_m), beta = exp(score - next_m);
                float val = d < D ? {v} : 0.0f;
                acc = acc * alpha + val * beta;
                l = l * alpha + beta;
                m = next_m;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}""")
    if len(names) > 30:
        raise ValueError("Metal kernel exceeds its buffer binding limit")
    source = """
        uint d = thread_position_in_threadgroup.x;
        uint group = threadgroup_position_in_grid.y;
        uint qi = group % Q, head = (group / Q) % H, row = group / (Q * H);
        uint kh = head / (H / HK);
        threadgroup float dot[256];
        float m = -INFINITY, l = 0.0f, acc = 0.0f;
    """ + f"\n        auto table = {table};\n" + "\n".join(blocks) + """
        uint base = ((row * H + head) * Q + qi) * (D + 2);
        if (d < D) out[base + d] = acc;
        if (d == 0) { out[base + D] = m; out[base + D + 1] = l; }
    """
    return names, source


@lru_cache(maxsize=64)
def _kernel(formats, dim, rows=1):
    import mlx.core as mx

    names, source = kernel_source(formats, dim, rows)
    tag = "".join(str(bits or 0) for bits in formats)
    return mx.fast.metal_kernel(name=f"arena_attention_{dim}_{rows}_{tag}", input_names=names,
                               output_names=["out"], source=source,
                               ensure_row_contiguous=True)


def paged_attention(queries, rows, positions, widths, *, scale):
    import mlx.core as mx

    from mlx_one.engine.modes import ModelCapabilityUnsupported

    batch, heads, length, dim = queries.shape
    if (len(rows) != batch or len(positions) != batch or len(widths) != batch
            or any(len(p) != length for p in positions)
            or any(w < 1 or w > length for w in widths)
            or any(not c._blocks or c.offset < w for c, w in zip(rows, widths, strict=True))):
        raise ValueError("invalid paged query row mapping")
    if batch > 8:
        raise ModelCapabilityUnsupported("paged attention supports at most eight rows")
    if dim < 1 or dim > 256 or heads % rows[0]._blocks[0].block.layout.num_kv_heads:
        raise ModelCapabilityUnsupported("paged attention supports GQA head dimensions 1..256")
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise ModelCapabilityUnsupported("paged attention requires MLX Metal custom kernels")
    arena = rows[0].pool
    for cache in rows:
        if cache.pool is not arena:
            raise ValueError("batch spans page arenas")
        cache.page_table.validate(cache._blocks)
        from mlx_one.engine.paged import KERNEL_VERSION

        if sum(r.block.token_count for r in cache._blocks) != cache.offset:
            raise ValueError("page table committed length mismatch")
        for reference in cache._blocks:
            layout = reference.block.layout
            if (layout.version != 3 or layout.kernel_version != KERNEL_VERSION
                    or layout.layer_id != cache.layer_id or layout.page_tokens != cache.block_tokens
                    or layout.key_dtype != cache.dtype or layout.value_dtype != cache.dtype
                    or layout.quantization_bits not in {None, cache.bits}):
                raise ModelCapabilityUnsupported("unsupported arena page layout")
    accum = maximum = normalizer = None
    pos, real_widths = mx.array(positions), mx.array(widths)
    lengths = mx.array([c.offset for c in rows])
    empty = mx.array([[-1, -1, 0, 0]] * 16, dtype=mx.int32)
    with arena.lease():
        for chunk in sorted({key for c in rows for key in c.page_table.chunks}):
            ids = sorted({entry[0] for c in rows for index, entry in c.page_table.entries.items()
                          if index // 16 == chunk})
            # 6 common + B tables + at most 24 segment arrays fit Metal's 31 bindings.
            quantized = any(arena.segments[sid].layout.quantization_bits for sid in ids)
            group_size = max(1, (30 - 6 - batch) // (6 if quantized else 2))
            for start in range(0, len(ids), group_size):
                selected = ids[start:start + group_size]
                segments = [arena.segments[sid] for sid in selected]
                formats = tuple(s.layout.quantization_bits for s in segments)
                inputs = [queries, mx.array(selected), pos, real_widths, lengths,
                          mx.array([scale], dtype=mx.float32)]
                inputs.extend(c.page_table.chunks.get(chunk, empty) for c in rows)
                for segment in segments:
                    inputs.extend(segment.state)
                warp = 1 << max(5, (dim - 1).bit_length())
                layout = segments[0].layout
                if any((s.layout.head_dim, s.layout.num_kv_heads, s.layout.page_tokens)
                       != (dim, layout.num_kv_heads, rows[0].block_tokens) for s in segments):
                    raise ModelCapabilityUnsupported("incompatible arena segment layouts")
                template = [("B", batch), ("H", heads), ("HK", layout.num_kv_heads),
                            ("Q", length), ("D", dim), ("P", rows[0].block_tokens), ("W", warp)]
                partial = _kernel(formats, dim, batch)(
                    inputs=inputs, template=template,
                    grid=(warp, batch * heads * length, 1), threadgroup=(warp, 1, 1),
                    output_shapes=[(batch, heads, length, dim + 2)], output_dtypes=[mx.float32])[0]
                pm = partial[..., dim:dim + 1]
                pl, pa = partial[..., dim + 1:], partial[..., :dim]
                if accum is None:
                    maximum, normalizer, accum = pm, pl, pa
                else:
                    new_max = mx.maximum(maximum, pm)
                    center = mx.where(normalizer + pl > 0, new_max, 0)
                    a = mx.exp(mx.where(normalizer > 0, maximum, -float("inf")) - center)
                    b = mx.exp(mx.where(pl > 0, pm, -float("inf")) - center)
                    accum, normalizer = accum * a + pa * b, normalizer * a + pl * b
                    maximum = new_max
                # Bound partial-result lifetime instead of retaining all partitions.
                mx.eval(accum, normalizer, maximum)
        result = (accum / mx.maximum(normalizer, 1e-20)).astype(queries.dtype)
        mx.eval(result)
        return result


def numpy_paged_attention(queries, rows, positions, widths, *, scale):
    """Independent CPU oracle. Never selected by a production runner."""
    import numpy as np

    pages, starts = page_bindings(rows)
    result = np.zeros_like(queries)
    for row in range(queries.shape[0]):
        for head in range(queries.shape[1]):
            for query in range(widths[row]):
                maximum, normalizer, accum = -np.inf, 0.0, np.zeros(queries.shape[-1])
                for page, offsets in zip(pages, starts, strict=True):
                    if offsets[row] < 0:
                        continue
                    dim = page.layout.head_dim
                    kh = head // (queries.shape[1] // page.layout.num_kv_heads)
                    for token in range(page.token_count):
                        if offsets[row] + token > positions[row][query]:
                            continue
                        vectors = []
                        for at in ((0, 3) if page.layout.quantization_bits else (0, 1)):
                            data = page.state[at][0, kh, token]
                            if page.layout.quantization_bits:
                                bits = page.layout.quantization_bits
                                if bits == 4:
                                    data = np.stack((data & 15, data >> 4), axis=-1).reshape(-1)
                                data = (data.astype(np.float32).reshape(-1, 32)
                                        * page.state[at + 1][0, kh, token, :, None]
                                        + page.state[at + 2][0, kh, token, :, None])
                                data = data.reshape(-1)[:dim]
                            vectors.append(data.astype(np.float32))
                        score = float(np.dot(queries[row, head, query].astype(np.float32),
                                             vectors[0])) * scale
                        new_max = max(maximum, score)
                        a, b = np.exp(maximum - new_max), np.exp(score - new_max)
                        accum, normalizer = accum * a + vectors[1] * b, normalizer * a + b
                        maximum = new_max
                result[row, head, query] = accum / max(normalizer, 1e-20)
    return result
