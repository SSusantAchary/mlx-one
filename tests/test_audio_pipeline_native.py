"""Real tiny MLX execution and frozen checkpoint metadata contracts, no downloads."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on Metal", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.audio.input import AudioInput, prepare_audio
from mlx_one.audio.lfm import stream_lfm
from mlx_one.models.audio.lfm2_audio.checkpoint_config import LFM2AudioConfig
from mlx_one.models.audio.lfm2_audio.detokenizer import DetokenizerConfig, LFM2AudioDetokenizer
from mlx_one.models.audio.lfm2_audio.dsp import preprocess
from mlx_one.models.audio.lfm2_audio.loading import LoadedLFMAudio
from mlx_one.models.audio.lfm2_audio.native import NativeLFMAudio

FIXTURES = Path(__file__).parent / "fixtures/audio_checkpoints"


def tiny_config():
    return LFM2AudioConfig.from_dict(
        {
            "lfm": {
                "model_type": "lfm2",
                "vocab_size": 256,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "layer_types": ["conv", "full_attention"],
                "block_multiple_of": 8,
            },
            "encoder": {
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 2,
                "subsampling_conv_channels": 4,
                "conv_kernel_size": 3,
            },
            "depthformer": {"layers": 1, "dim": 32, "num_heads": 4, "num_kv_heads": 2},
            "adapter_hidden_dims": [32],
        }
    )


def tiny_codec():
    return LFM2AudioDetokenizer(
        DetokenizerConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=("conv", "sliding_attention"),
            intermediate_size=32,
            output_size=18,
            n_fft=16,
            hop_length=4,
            upsample_factor=2,
        )
    )


def test_lfm_real_frontend_encoder_backbone_head_and_codec():
    config = tiny_config()
    model = NativeLFMAudio(config)
    model.eval()
    features = preprocess(mx.zeros((1600,)), config.preprocessor)
    encoded, lengths = model.encode(features)
    cache = model.make_cache()
    hidden, _, _ = model.lfm(None, input_embeddings=encoded, cache=cache)
    frame = model.audio_frame(hidden[:, -1:])
    codec = tiny_codec()
    codec.eval()
    waveform = codec(mx.zeros((1, 8, 4), dtype=mx.int32))
    mx.eval(features, encoded, hidden, frame, waveform)
    assert encoded.shape[-1] == 32 and frame.shape == (1, 8)
    assert waveform.shape == (1, 32) and bool(mx.all(mx.isfinite(waveform)).item())
    assert bool(mx.all(mx.isfinite(features)).item())
    for state in cache:
        state.reset()


def test_native_generation_budget_and_cancellation_release_caches():
    model = NativeLFMAudio(tiny_config())
    model.eval()
    caches = model.make_cache()
    model.make_cache = lambda: caches
    embeddings = model.lfm.embed_tokens(mx.array([[1, 2]]))
    with pytest.raises(RuntimeError, match="cancelled"):
        list(model.generate(embeddings, cancelled=lambda: True))
    assert all(state.offset == 0 for state in caches)
    # A real forward at least once, followed by bounded completion or a clear budget failure.
    try:
        list(model.generate(embeddings, max_tokens=1))
    except ValueError as exc:
        assert "terminate" in str(exc)
    assert all(state.offset == 0 for state in caches)


def test_stream_codec_outputs_real_pcm_offsets_order_and_completion(monkeypatch):
    import mlx_one.audio.lfm as pipeline

    codec = tiny_codec()
    codec.eval()

    class Model:
        def generate(self, *args, **kwargs):
            yield "text", 2
            for _ in range(5):
                yield "audio", mx.zeros((1, 8), dtype=mx.int32)
            yield "audio_end", mx.full((1, 8), 2048, dtype=mx.int32)
            yield "text", 3
            for _ in range(5):
                yield "audio", mx.zeros((1, 8), dtype=mx.int32)
            yield "end", 7

    tokenizer = SimpleNamespace(decode=lambda tokens, **_: "".join(str(t) for t in tokens))
    bundle = LoadedLFMAudio(Model(), tokenizer, codec, Path("/tiny"), "a" * 40, {})
    monkeypatch.setattr(pipeline, "_prompt", lambda *a, **kw: mx.zeros((1, 1, 32)))
    events = list(stream_lfm(bundle, text="hello", chunk_frames=2))
    assert events[-1].kind == "completion"
    assert [e.sequence for e in events] == list(range(len(events)))
    chunks = [e for e in events if e.kind == "audio_chunk"]
    assert chunks and sum(len(e.waveform) for e in chunks) == 80
    offset = 0
    for event in chunks:
        assert event.sample_offset == offset and np.isfinite(event.waveform).all()
        offset += len(event.waveform)
    assert events[-1].timings["time_to_first_audio_seconds"] > 0


@pytest.mark.parametrize(
    "name",
    [
        "Qwen3-ASR-0.6B-8bit",
        "Qwen3-ASR-1.7B-8bit",
        "Qwen3-ForcedAligner-0.6B-8bit",
        "LFM2.5-Audio-1.5B-4bit",
    ],
)
def test_pinned_header_names_shapes_and_packed_widths(name):
    data = json.loads((FIXTURES / (name + ".json")).read_text())
    header = dict(data["headers"])
    if name.startswith("Qwen"):
        from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig
        from mlx_one.models.audio.qwen3_asr.model import QwenASRModel

        model = QwenASRModel(QwenASRConfig.from_dict(data["config"]))
        packing = 4  # eight-bit values packed in uint32
    else:
        model = NativeLFMAudio(LFM2AudioConfig.from_dict(data["config"]))
        packing = 8
        for key in list(header):
            if key.startswith("lfm.") and key.endswith(".conv.conv.weight"):
                original = header.pop(key)
                header[key.removesuffix(".weight")] = {"shape": original["shape"][:2]}
    params = dict(tree_flatten(model.parameters()))
    assert not params.keys() - header.keys()
    assert not {k for k in header.keys() - params.keys() if not k.endswith((".scales", ".biases"))}
    for key, value in params.items():
        shape = list(header[key]["shape"])
        if key.endswith(".weight") and key.removesuffix(".weight") + ".scales" in header:
            shape[-1] *= packing
        assert tuple(shape) == value.shape, key


def test_parakeet_mlx_map_matches_full_native_contract():
    from mlx_one.models.audio.parakeet.mlx_export import config_from_mlx, map_weights
    from mlx_one.models.audio.parakeet.weights import weight_contract

    data = json.loads((FIXTURES / "parakeet-tdt-0.6b-v3.json").read_text())
    config = config_from_mlx(data["config"])
    tensors = {k: mx.zeros(v["shape"], dtype=mx.float32) for k, v in data["headers"].items()}
    mapped = map_weights(tensors, config)
    weight_contract(config).validate(mapped)
    assert bool(mx.all(mapped["decoder.lstm.bias_hh_l0"] == 0).item())


def test_encoded_wav_bytes_and_invalid_decoder_input():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(np.zeros((800, 2), dtype="<i2").tobytes())
    pcm = prepare_audio(AudioInput(buffer.getvalue(), media_type="audio/wav"))
    assert len(pcm) == 1600 and np.isfinite(pcm).all()
    with pytest.raises(ValueError, match="decode"):
        prepare_audio(AudioInput(b"not audio", media_type="audio/wav"))


def write_tiny_lfm_checkpoint(root, *, asr=False):
    """Engineered test weights exercise real termination; they are not speech quality evidence."""
    from dataclasses import asdict

    from tokenizers import Tokenizer, models, pre_tokenizers

    root.mkdir()
    config = tiny_config()
    model = NativeLFMAudio(config)
    # Residual operators are zero; select an explicit deterministic modality sequence.
    model.load_weights([(k, mx.zeros_like(v)) for k, v in tree_flatten(model.parameters())])
    embedding = mx.zeros((256, 32)).at[9, 0].add(1).at[128, 0].add(2)
    embedding = embedding.at[7, 0 if asr else 1].add(3)
    model.lfm.embed_tokens.weight = embedding
    model.lfm.embedding_norm.weight = mx.ones((32,))
    model.depth_linear.weight = mx.zeros((8 * 32, 32)).at[mx.arange(8) * 32, 0].add(1)
    for head in model.depth_embeddings:
        head.embedding_norm.weight = mx.ones((32,))
        head.to_logits.weight = mx.zeros((2049, 32)).at[0, 0].add(1).at[2048, 0].add(-1)
    for codebook in range(8):
        model.audio_embedding.embedding.weight = (
            model.audio_embedding.embedding.weight.at[codebook * 2049, 0]
            .add(-1)
            .at[codebook * 2049 + 2048, 1]
            .add(1)
        )
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if name.startswith("lfm.") and name.endswith(".conv.conv"):
            name, value = name + ".weight", value[..., None]
        weights[name] = value
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(json.dumps(asdict(config)))
    special = {
        1: "<|startoftext|>",
        6: "<|im_start|>",
        7: "<|im_end|>",
        128: "<|audio_start|>",
        130: "<|text_end|>",
    }
    vocab = {special.get(i, {0: "unk", 9: "assistant"}.get(i, f"unused{i}")): i for i in range(256)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="unk"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(list(special.values()))
    tokenizer.save(str(root / "tokenizer.json"))
    codec = tiny_codec()
    codec_weights = dict(tree_flatten(codec.parameters()))
    codec_weights["istft.window"] = codec.window
    folder = root / "audio_detokenizer"
    folder.mkdir()
    # Match the actual published OIK convolution layout.
    codec_weights = {
        k: v.transpose(0, 2, 1) if k.endswith(".conv.conv.weight") else v
        for k, v in codec_weights.items()
    }
    mx.save_safetensors(str(folder / "model.safetensors"), codec_weights)
    (folder / "config.json").write_text(json.dumps(asdict(codec.config)))
    return root


@pytest.mark.parametrize("asr", [True, False])
def test_real_tiny_checkpoint_load_and_native_task_end_to_end(tmp_path, asr):
    from mlx_one.audio import load_audio_model, transcribe
    from mlx_one.audio.lfm import generate_speech

    root = write_tiny_lfm_checkpoint(tmp_path / "tiny", asr=asr)
    bundle = load_audio_model(root, task="transcribe" if asr else "synthesize", offline=True)
    if asr:
        result = transcribe(bundle, np.zeros(1600, dtype=np.float32), sample_rate=16000)
        assert result.text == "" and result.duration == 0.1
    else:
        first = generate_speech(bundle, "hello", max_tokens=8)
        second = generate_speech(bundle, "hello", max_tokens=8)
        assert len(first.waveform) > 0 and np.isfinite(first.waveform).all()
        np.testing.assert_array_equal(first.waveform, second.waveform)
        assert first.sample_rate == 24000 and first.audio_bytes()[:4] == b"RIFF"


def test_managed_audio_engine_releases_residency_after_native_tts(tmp_path):
    from mlx_one.audio import AudioEngine

    root = write_tiny_lfm_checkpoint(tmp_path / "managed")
    with AudioEngine(root, offline=True) as engine:
        result = engine.generate_speech("hello", max_tokens=8)
        assert result.duration > 0
        assert engine.service.pool.stats()["active_leases"] == 0
        assert engine.service.pool.stats()["resident_models"] == 1
    assert engine.service.pool.stats()["resident_models"] == 0


def test_native_qwen_q8_keeps_encoder_float_and_strict_packed_weights():
    import mlx.nn as nn

    from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig, QwenAudioConfig
    from mlx_one.models.audio.qwen3_asr.model import QwenASRModel
    from mlx_one.models.language.qwen3.config import Qwen3Config
    from mlx_one.text.loading import _prepare_quantized_model, _validate_quantized_tensors

    config = QwenASRConfig(
        "qwen3_asr",
        QwenAudioConfig(64, 1, 2, 128, 64, downsample_hidden_size=2),
        Qwen3Config.from_dict(
            {
                "model_type": "qwen3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "vocab_size": 256,
                "tie_word_embeddings": True,
            }
        ),
        audio_token_id=3,
        audio_start_token_id=4,
        audio_end_token_id=5,
        timestamp_token_id=6,
    )
    source = QwenASRModel(config)
    nn.quantize(
        source,
        bits=8,
        group_size=64,
        class_predicate=lambda p, m: p.startswith("model.") and hasattr(m, "to_quantized"),
    )
    weights = dict(tree_flatten(source.parameters()))
    target = QwenASRModel(config)
    _prepare_quantized_model(target, weights, {"bits": 8, "group_size": 64, "mode": "affine"})
    _validate_quantized_tensors(target, weights)
    target.load_weights(list(weights.items()), strict=True)
    logits = target(mx.array([[1, 2]]))
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert target.model.embed_tokens.weight.dtype == mx.uint32
    assert target.audio_tower.conv2d1.weight.dtype == mx.float32
    broken = dict(weights)
    del broken["model.embed_tokens.scales"]
    with pytest.raises(ValueError):
        target.load_weights(list(broken.items()), strict=True)


def test_native_audio_primary_server_uses_existing_manager_and_routes(tmp_path):
    from fastapi.testclient import TestClient

    from mlx_one.server import ModelManager, ServerConfig, create_app

    root = write_tiny_lfm_checkpoint(tmp_path / "http")
    manager = ModelManager(alias="tiny-audio")
    manager.load(root, offline=True)
    app = create_app(model_manager=manager, config=ServerConfig(warmup=False, timeout=0))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech", json={"model": "tiny-audio", "input": "hello", "max_tokens": 8}
        )
        assert response.status_code == 200, response.text
        assert response.content[:4] == b"RIFF"
        response = client.post(
            "/v1/chat/completions",
            json={"model": "tiny-audio", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 400
    assert manager._pool.stats()["resident_models"] == 0


def test_lfm_bf16_encoder_depthformer_and_waveform_codec_accept_float_frontend():
    model = NativeLFMAudio(tiny_config())
    model.set_dtype(mx.bfloat16)
    model.eval()
    features = preprocess(mx.zeros((1600,)), model.config.preprocessor)
    encoded, _ = model.encode(features)
    hidden, _, _ = model.lfm(None, input_embeddings=encoded, cache=model.make_cache())
    codes = model.audio_frame(hidden[:, -1:])
    codec = tiny_codec()
    codec.set_dtype(mx.bfloat16)
    codec.eval()
    waveform = codec(mx.zeros((1, 8, 3), dtype=mx.int32))
    mx.eval(encoded, codes, waveform)
    assert bool(mx.all(mx.isfinite(encoded)).item())
    assert bool(mx.all(mx.isfinite(waveform)).item())


def test_whisper_word_alignment_uses_segment_token_rows_and_frame_window():
    from mlx_one.audio.decoding import align_words

    tokenizer = SimpleNamespace(split_to_word_tokens=lambda ids: (["one", "two"], [[1], [2]]))
    attention = mx.zeros((1, 1, 6, 20)).at[0, 0, 2, 4].add(10).at[0, 0, 3, 6].add(10)
    result = align_words(
        (1, 2),
        (attention,),
        tokenizer,
        alignment_heads=[[0, 0]],
        median_width=1,
        time_offset=10.0,
        token_start=2,
        frame_start=3,
        frame_end=8,
    )
    assert len(result) == 2 and result[0].start == 10.0
    assert all(10 <= word.start <= word.end <= 10.08 for word in result)


def test_pinned_codec_header_distinguishes_text_vocab_from_codebook_vocab():
    data = json.loads((FIXTURES / "LFM2.5-Audio-1.5B-4bit.json").read_text())
    config = DetokenizerConfig.from_dict(data["codec_config"])
    assert config.vocab_size == 2048
    config.intermediate_size = data["codec_headers"]["lfm.layers.0.feed_forward.w1.weight"][
        "shape"
    ][0]
    model = LFM2AudioDetokenizer(config)
    params = dict(tree_flatten(model.parameters()))
    header = data["codec_headers"]
    assert set(header) - params.keys() == {"istft.window"}
    for key, value in params.items():
        shape = header[key]["shape"]
        if key.endswith(".conv.conv.weight"):
            shape = [shape[0], shape[2], shape[1]]
        assert list(value.shape) == shape, key


def test_lfm_frontend_matches_publisher_nemo_window_and_padding_contract():
    from mlx_one.models.audio.parakeet.processing import ParakeetProcessor
    from mlx_one.models.audio.parakeet.processing import preprocess as shared_frontend

    waveform = mx.sin(mx.arange(1600) * 0.05)
    result = preprocess(waveform, tiny_config().preprocessor)
    reference, _ = shared_frontend(waveform, ParakeetProcessor())
    mx.eval(result, reference)
    assert result.shape == reference.shape
    assert float(mx.max(mx.abs(result - reference)).item()) < 1e-4
    assert bool(mx.all(result[:, -1] == 0).item())


def test_managed_whisper_language_detection_runs_native_encoder_and_probabilities():
    from test_native_whisper_mlx import config as whisper_config

    from mlx_one.audio import AudioEngine
    from mlx_one.models.audio.whisper.loading import LoadedWhisper
    from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration

    def load_on_owner(_):
        # Construct lazy MLX arrays on the device owner, as real checkpoint loaders do.
        native = WhisperForConditionalGeneration(whisper_config(max_source_positions=1500))
        tokenizer = SimpleNamespace(
            special_tokens={"<|en|>": 3, "<|fr|>": 4},
            token_id=lambda text: 2 if text == "<|startoftranscript|>" else 3,
        )
        return LoadedWhisper(
            native, tokenizer, SimpleNamespace(mel_filters=None), {}, Path("/tiny-whisper"), None
        )

    with AudioEngine("tiny-whisper", loader=load_on_owner) as engine:
        result = engine.detect_language(np.zeros(1600, dtype=np.float32), sample_rate=16000)
        assert result["language"] in {"en", "fr"}
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        assert engine.service.pool.stats()["active_leases"] == 0


def test_alignment_benchmark_accepts_alignment_without_speech_generation_timings(
    tmp_path, monkeypatch
):
    from argparse import Namespace

    import mlx_one.audio
    from benchmarks.audio.result_schema import digest_file
    from benchmarks.audio.run import benchmark
    from mlx_one.audio.schemas import AlignmentUnit, AudioAlignment

    # Routing/schema fixture only: no recorded values are published as model performance.
    root = tmp_path / "benchmark-fixture"
    root.mkdir()
    config = root / "config.json"
    config.write_text('{"model_type":"whisper"}')
    (root / "mlx-one-download.json").write_text(
        json.dumps(
            {
                "checkpoint": "test-only/fixture",
                "revision": "a" * 40,
                "assets": {"config.json": digest_file(config)},
            }
        )
    )
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture-only")
    text = tmp_path / "text.txt"
    text.write_text("hello")
    monkeypatch.setattr(
        "mlx_one.audio.input.prepare_audio", lambda *a, **kw: np.zeros(160000, dtype=np.float32)
    )
    monkeypatch.setattr(mlx_one.audio, "load_audio_model", lambda *a, **kw: object())
    alignment = AudioAlignment(
        (AlignmentUnit("hello", 0, 1),),
        "hello",
        "English",
        10.0,
        "test-only/fixture",
        "a" * 40,
        0.08,
    )
    monkeypatch.setattr(mlx_one.audio, "align_audio", lambda *a, **kw: alignment)
    output = tmp_path / "unit-result.json"
    benchmark(
        Namespace(
            model=root,
            revision="a" * 40,
            checkpoint="test-only/fixture",
            output=output,
            audio=audio,
            text_file=text,
            task="align",
            workload="ALIGN-S1",
            language="English",
            measurement_type="direct-model-prepared-pcm",
            name="unit fixture",
            precision="FP32",
            max_tokens=None,
            voice="us_male",
            boundaries_file=None,
        )
    )
    record = json.loads(output.read_text())
    assert record["metrics"]["time_to_first_audio_ms"] is None
    assert record["metrics"]["alignment_mean_error_ms"] is None
    assert len(record["runs"]) == 5
