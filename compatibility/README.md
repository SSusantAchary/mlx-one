# MLX compatibility policy

`mlx-one` supports a rolling window containing the five most recent stable
Apple MLX releases. A version is `verified` only after the native correctness,
model, training, and performance qualification gates have passed. Merely being
inside the window makes a release `supported`, not `verified`.

The canonical runtime manifest is packaged at
`src/mlx_one/data/compatibility/mlx.json`. Immutable reports produced by the
qualification workflow belong in `compatibility/history/` and are referenced
from that manifest.

Statuses are:

- `verified`: passed the complete qualification contract.
- `supported`: inside the rolling window but not individually certified.
- `unverified`: newer than the latest observed/qualified stable release.
- `degraded`: correct enough to run, with accepted capability or performance regressions.
- `incompatible`: known correctness, safety, serialization, or required-capability failure.
- `legacy`: outside the rolling five-release window.

Promotion is an explicit maintainer action. Release detection and CI must never
mark a release verified automatically.
