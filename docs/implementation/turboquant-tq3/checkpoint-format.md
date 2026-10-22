# Exact checkpoint format

Repository: `manjunathshiva/gpt-oss-20b-tq3`.
Revision: `20ba1835e625264e07366d215933fe31a9940a6a`.
The fixture `tests/fixtures/turboquant_metadata/source.json` hashes every retrieved file.
Both shard headers were retrieved using HTTP 206 byte ranges: first eight bytes for header length,
then exactly the JSON header. Header sizes: 53,432 and 56,010 bytes. No weight bodies were fetched.
The index/header sets contain exactly 966 tensors: 169 U32, 675 F16 and 122 BF16.

Metadata is `mode=turboquant`, `bits=3`, `group_size=64`, `rotation=hadamard`,
`rotation_seed=42`, `fuse_rotations=false`, `use_qjl=false`. The eight FP16 codebook values are:
`[-2.15234375,-1.34375,-0.755859375,-0.2451171875,0.2451171875,0.755859375,1.34375,2.15234375]`.
The native format identifier is **turboquant_tq3**, precision label **TQ3 / TurboQuant 3-bit**.
It is not MLX affine Q3.

| Tensor | Physical shape | Dtype |
|---|---|---|
| Dense weight | `[O,ceil(K/10)]` | uint32 |
| Expert weight | `[E,O,ceil(K/10)]` | uint32 |
| Scales | `[O,K/64]` or `[E,O,K/64]` | FP16 |
| Codebook | `[8]`, shared across the bank | FP16 |
| Signs | `[K]`, shared across the bank | FP16, exactly ±1 |
| Projection bias | `[O]` or `[E,O]` | FP16 |

Words pack ten indices LSB first: `index(col)=(word[col//10] >> (3*(col%10))) & 7`.
The upper two bits and final padding slots do not contribute. Packing is row-local and does not
restart at group boundaries. A group boundary can cut a word: e.g. columns 63/64 have separate scales.
The dequantized rotated value is `codebook[index]*scales[...,col//64]`.
Logical matrices are output by input; expert banks are expert by output by input.

Examples from actual headers: q_proj `[4096,288]`, k/v `[512,288]`,
o_proj `[2880,410]` with K=4096 (six valid indices in the last word),
expert gate/up/down `[32,2880,288]` with scales `[32,2880,45]`,
lm_head `[201088,288]`. The embedding `[201088,2880]`, router, norms and sinks stay BF16.
All 169 compressed modules carry their own codebook/sign buffers.

The reference rotates weights as `W_rot=H(W*signs)/sqrt(block)` and inference inputs as
`x_rot=H(x*signs)/sqrt(block)`, then multiplies `x_rot @ W_rot.T`.
`fuse_rotations=false` requires this online input rotation; it does not mean unrotated stored weights.
Stored signs are authoritative; the seed is not used to regenerate them.
Use the largest compatible divisor `m*2**k`, m in `{1,12,20,28}`: block=320 for K=2880,
block=4096 for K=4096. Independent power-of-two tests and reference-generated K=2880 fixtures
verify orientation and block behavior. Inverse reconstruction applies H first, then signs.

GPT geometry remains 24 layers, 32 experts, top four, 64 query/8 KV heads, head dimension 64,
alternating sliding/full attention, window 128, YaRN factor 32/original 4096/theta 150000,
`truncate=false`, capped GPT SwiGLU and biased attention with sinks. Neither 120B defaults nor
extra Hadamard transforms are used.
