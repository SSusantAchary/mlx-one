"""Source temporal regression and real document candidate checks, without promotion."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local/native_tasks/runs/m4-20261103"
CKPT = ROOT / "local/native_tasks/checkpoints/qwen25-existing-q4"


@pytest.mark.skipif(
    not (RUN / "temporal-reference.npz").exists(), reason="pinned-source oracle absent"
)
def test_video_positions_preserve_source_seconds_integer_cast():
    import mlx.core as mx

    from mlx_one.models.vision_language.qwen2_5_vl.positions import video_position_ids

    ids = mx.array([[42, 151652] + [151656] * 8 + [151653, 43]])
    types = mx.array([[0, 0] + [2] * 8 + [0, 0]])
    with np.load(RUN / "temporal-reference.npz", allow_pickle=False) as ref:
        for i, interval in enumerate((None, [0.5], [1.0], [2.5])):
            pos, delta = video_position_ids(
                ids,
                mm_token_type_ids=types,
                image_grid_thw=None,
                video_grid_thw=[[2, 4, 4]],
                spatial_merge_size=2,
                attention_mask=None,
                second_per_grid_ts=interval,
                tokens_per_second=2,
            )
            assert np.array_equal(np.asarray(pos), ref[f"positions_{i}"])
            assert np.array_equal(np.asarray(delta), ref[f"delta_{i}"])


@pytest.mark.skipif(
    not (RUN / "temporal-weights.npz").exists(), reason="pinned-source oracle absent"
)
def test_qwen25_global_attention_is_per_temporal_grid():
    import mlx.core as mx

    from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLVisionConfig
    from mlx_one.models.vision_language.qwen2_5_vl.model import Qwen2_5VisionTransformer

    cfg = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=8,
        fullatt_block_indexes=(1,),
        tokens_per_second=2,
    )
    model = Qwen2_5VisionTransformer(cfg)
    weights = []
    with np.load(RUN / "temporal-weights.npz", allow_pickle=False) as f:
        for key in f.files:
            value = f[key]
            if value.ndim == 5:
                value = value.reshape(value.shape[0], -1)
            key = key.replace("merger.mlp.0.", "merger.mlp_0.").replace(
                "merger.mlp.2.", "merger.mlp_2."
            )
            weights.append((key, mx.array(value)))
    model.load_weights(weights, strict=True)
    model.eval()
    pixels = np.random.default_rng(20261103).normal(size=(32, 24)).astype(np.float32)
    together, _ = model(mx.array(pixels), [[2, 4, 4]])
    first, _ = model(mx.array(pixels[:16]), [[1, 4, 4]])
    second, _ = model(mx.array(pixels[16:]), [[1, 4, 4]])
    assert np.allclose(
        np.asarray(together), np.asarray(mx.concatenate((first, second))), atol=1e-4, rtol=1e-4
    )
    with np.load(RUN / "temporal-reference.npz", allow_pickle=False) as ref:
        assert np.allclose(np.asarray(together), ref["video_merged"], atol=1e-4, rtol=1e-4)


def test_video_sampling_bounds_and_exact_known_source_indices():
    from mlx_one.vision.video_processing import sample_frame_indices

    assert sample_frame_indices(300, 30, num_frames=8).tolist() == [
        0,
        37,
        75,
        112,
        150,
        187,
        225,
        262,
    ]
    assert sample_frame_indices(5, 30, num_frames=4).tolist() == [0, 1, 2, 3]
    for total, fps, count in [
        (0, 30, 8),
        (1, 30, 2),
        (300, 0, 8),
        (300, 30, 1),
        (300, 30, 18),
        (4, 30, 8),
    ]:
        with pytest.raises(ValueError):
            sample_frame_indices(total, fps, num_frames=count)


@pytest.mark.skipif(not (CKPT / "model.safetensors").exists(), reason="pinned existing Q4 absent")
def test_real_document_candidate_answer_and_bounds():
    from mlx_one.vision.document_qa import REVISION, DocumentQAEngine

    cases = json.loads((RUN / "document-cases.json").read_text())["cases"]
    with DocumentQAEngine(CKPT) as qa:
        for question in ("", "<image> hello", "<|im_start|>hello", "a" * 8193):
            with pytest.raises(ValueError):
                qa.ask(cases[0]["image"], question)
        with pytest.raises(ValueError):
            qa.ask("page.pdf", "What is here?")
        answer = qa.ask(cases[0]["image"], cases[0]["question"])
        assert answer.answer == "Lee A. Waller"
        assert answer.revision == REVISION and answer.page_index == 1
        assert answer.token_ids and answer.image_metadata
        assert qa.engine.service.pool.stats()["active_leases"] == 0
