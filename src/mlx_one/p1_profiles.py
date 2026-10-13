"""Pinned P1 model assets and offline, explicit qualification profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class P1Checkpoint:
    model_id: str
    revision: str
    architecture: str
    context_tokens: int
    conversion_required: bool = False


CHECKPOINTS = {
    "qwen3-small": P1Checkpoint("mlx-community/Qwen3-0.6B-8bit",
        "11de96878523501bcaa86104e3c186de07ff9068", "qwen3", 40960),
    "qwen3-large": P1Checkpoint("mlx-community/Qwen3-1.7B-8bit",
        "8c24f6782a91421513803ce527a27dcc560ab904", "qwen3", 40960),
    "smol-small": P1Checkpoint("HuggingFaceTB/SmolLM2-360M",
        "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", "llama", 8192, True),
    "smol-large": P1Checkpoint("HuggingFaceTB/SmolLM2-1.7B",
        "effd688a12921b4cc83e3312b6feb579f70f9c71", "llama", 8192, True),
    "lfm-small": P1Checkpoint("mlx-community/LFM2-350M-8bit",
        "3b5014a1966c54313045fd0a23a098a220083b1e", "lfm2", 128000),
    "lfm-large": P1Checkpoint("mlx-community/LFM2-1.2B-8bit",
        "9384737a4b9172faf97f4d65d4bf7799bee182fa", "lfm2", 128000),
}

REFERENCE_PINS = {
    "jundot/omlx": "64bf5d1ba6336b45955c00f7fe0e46acd5c85095",
    "waybarrios/vllm-mlx": "9a939f8b59bddaf98ccab865f4fa7e2fe48c6d65",
    "raullenchai/Rapid-MLX": "b638c2ab6cebe01446ee34b53230b9ef44881453",
}


def profile(checkpoint: str, *, mode: str, prompt_tokens: int = 2048,
            output_tokens: int = 128, slots: int = 1, chunk_tokens: int = 512,
            workload: str = "baseline", checkpoint_path: str | None = None, **settings):
    """Build matched FP16/INT8 profiles without importing MLX or loading weights."""
    from mlx_one.inference_baseline import BaselineError, BaselineProfile

    asset = CHECKPOINTS[checkpoint]
    if mode == "auto":
        raise BaselineError("P1 measurements require an explicit backend")
    if prompt_tokens + output_tokens > asset.context_tokens:
        raise BaselineError("prompt plus output exceeds pinned checkpoint context")
    if slots not in {1, 2, 4} or chunk_tokens not in {128, 256, 512, 1024}:
        raise BaselineError("unsupported P1 batch/chunk profile")
    revision = asset.revision
    if asset.conversion_required:
        from mlx_one.conversion import verify_conversion_manifest

        manifest = verify_conversion_manifest(checkpoint_path) if checkpoint_path else None
        if not manifest or (manifest["source"], manifest["revision"], manifest["precision"],
                manifest["group_size"], manifest["floating_dtype"]) != (
                asset.model_id, asset.revision, "int8-affine", 64, "float16"):
            raise BaselineError(
                "SmolLM2 profiles require the verified pinned native INT8 conversion")
        revision = manifest["artifact_digest"]
    return BaselineProfile(model_id=asset.model_id, revision=revision,
        checkpoint_path=checkpoint_path,
        inference_mode=mode, context_tokens=prompt_tokens, output_tokens=output_tokens,
        slots=slots, repeats=5, execution_dtype="float16", cache_type="f16",
        chunk_tokens=chunk_tokens, workload=workload, **settings)
