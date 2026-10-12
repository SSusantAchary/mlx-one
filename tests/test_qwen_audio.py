"""Network-free Qwen contracts, task boundaries, CLI and session lifecycle tests."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mlx_one.audio import (
    AlignmentUnit,
    AudioAlignment,
    AudioStreamEvent,
    QwenASROptions,
    align_audio,
    load_audio_model,
    transcribe,
)
from mlx_one.audio.moonshine import AudioResourceLimitError
from mlx_one.audio.streaming import QwenAudioStream, decoded_pcm_chunks
from mlx_one.cli import main
from mlx_one.core.config import ConfigError
from mlx_one.core.registry import get_registration
from mlx_one.core.weights import WeightContractError
from mlx_one.models.audio.moonshine.loading import LoadedMoonshine
from mlx_one.models.audio.qwen3_asr.alignment import fix_timestamps, text_units
from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig, audio_token_length
from mlx_one.models.audio.qwen3_asr.loading import LoadedQwenAudio
from mlx_one.models.audio.qwen3_asr.processing import (
    language_name,
    parse_output,
    read_json,
    render_prompt,
)
from mlx_one.models.audio.qwen3_asr.weights import sanitize_weights, weight_contract


def payload(kind="qwen3_asr"):
    return {
        "model_type": "qwen3_asr",
        "timestamp_token_id": 6,
        "support_languages": ["English", "Chinese"],
        "thinker_config": {
            "model_type": kind,
            "audio_token_id": 3,
            "audio_start_token_id": 4,
            "audio_end_token_id": 5,
            "classify_num": 12,
            "audio_config": {
                "d_model": 8,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "output_dim": 8,
                "downsample_hidden_size": 2,
            },
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": kind == "qwen3_asr",
            },
        },
    }


@dataclass(frozen=True)
class Shape:
    shape: tuple[int, ...]

    @property
    def ndim(self):
        return len(self.shape)

    def transpose(self, *axes):
        return Shape(tuple(self.shape[i] for i in axes))


def test_imports_and_registry_remain_backend_free():
    code = (
        "import sys; import mlx_one.audio; from mlx_one.core.registry import get_registration; "
        "get_registration('qwen3_asr').config_class(); "
        "assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; assert 'mlx_audio' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    asr, aligner = [get_registration(kind) for kind in ("qwen3_asr", "qwen3_forced_aligner")]
    assert asr.supports("transcribe") and asr.supports("stream")
    assert not asr.supports("align")
    assert aligner.supports("align") and not aligner.supports("transcribe")


def test_nested_dispatch_rejects_wrong_task_before_backend(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(payload("qwen3_forced_aligner")))
    with pytest.raises(ValueError, match="does not support"):
        load_audio_model(tmp_path, task="transcribe")


def test_loaded_alignment_bundle_never_transcribes_even_empty_audio():
    native = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_forced_aligner"))
    bundle = LoadedQwenAudio(native, SimpleNamespace(), {}, "", Path("/fixture"), "pin")
    with pytest.raises(ValueError, match="ForcedAligner cannot transcribe"):
        transcribe(bundle, [], sample_rate=16000)
    asr = LoadedQwenAudio(
        SimpleNamespace(config=SimpleNamespace(model_type="qwen3_asr")),
        SimpleNamespace(),
        {},
        "",
        Path("/fixture"),
        "pin",
    )
    with pytest.raises(ValueError, match="requires a native ForcedAligner"):
        align_audio(asr, [], "", language="English", sample_rate=16000)


def test_qwen_options_are_not_silently_ignored_by_moonshine():
    moonshine = LoadedMoonshine(SimpleNamespace(), SimpleNamespace(), {}, Path("/fixture"), "pin")
    with pytest.raises(ValueError, match="Moonshine does not support Qwen"):
        transcribe(moonshine, [], sample_rate=16000, options=QwenASROptions(context="names"))


@pytest.mark.parametrize(
    "frames,expected", [(0, 0), (1, 1), (8, 1), (9, 2), (99, 13), (100, 13), (101, 14), (800, 104)]
)
def test_chunked_convolution_lengths(frames, expected):
    assert audio_token_length(frames) == expected


def test_config_and_tensor_contract():
    values = payload()
    values["thinker_config"]["text_config"]["head_dim"] = 8
    config = QwenASRConfig.from_dict(values)
    assert (
        config.text_config.hidden_size
        != config.text_config.num_attention_heads * config.text_config.head_dim
    )
    shapes = weight_contract(config).expected
    weights = {
        "thinker." + name: Shape(
            (s[0], s[3], s[1], s[2]) if "conv2d" in name and name.endswith(".weight") else s
        )
        for name, s in shapes.items()
    }
    cleaned = sanitize_weights(weights, config)
    assert cleaned["audio_tower.conv2d1.weight"].shape == (2, 3, 3, 1)
    for variant in (
        {**weights, "thinker.unknown": Shape((1,))},
        {k: v for k, v in weights.items() if k != "thinker.model.norm.weight"},
    ):
        with pytest.raises(WeightContractError):
            sanitize_weights(variant, config)
    integer = dict(weights)
    integer["thinker.model.norm.weight"] = SimpleNamespace(shape=(8,), dtype="int32")
    with pytest.raises(WeightContractError, match="precision"):
        sanitize_weights(integer, config)
    aligner = QwenASRConfig.from_dict(payload("qwen3_forced_aligner"))
    assert weight_contract(aligner).expected["lm_head.weight"] == (12, 8)
    for mutate in (
        lambda p: p.update(quantization_config={}),
        lambda p: p.update(quantization={}),
        lambda p: p.update(support_languages="English"),
        lambda p: p.update(timestamp_segment_time=True),
        lambda p: p["thinker_config"].update(model_type="qwen3_other"),
        lambda p: p["thinker_config"].update(quantization={}),
        lambda p: p["thinker_config"].update(text_config=None),
        lambda p: p["thinker_config"]["text_config"].pop("rope_theta"),
        lambda p: p["thinker_config"]["text_config"].update(quantization={}),
        lambda p: p["thinker_config"]["audio_config"].update(n_window=100),
        lambda p: p["thinker_config"]["audio_config"].update(max_source_positions=12),
        lambda p: p["thinker_config"]["text_config"].update(attention_bias=True),
        lambda p: p["thinker_config"]["text_config"].update(rope_theta=float("nan")),
        lambda p: p["thinker_config"]["text_config"].update(rms_norm_eps=float("inf")),
        lambda p: p["thinker_config"]["text_config"].update(head_dim=3),
        lambda p: p["thinker_config"]["text_config"].update(tie_word_embeddings=1),
    ):
        bad = copy.deepcopy(payload())
        mutate(bad)
        with pytest.raises(ConfigError):
            QwenASRConfig.from_dict(bad)


def test_prompts_and_unknown_metrics():
    template = "{{ '<|im_start|>system\\\\n' + messages[0].content + '<|audio_pad|>' }}"
    prompt = render_prompt(template, "names", "English", 3)
    assert "system\\n" in prompt and prompt.count("<|audio_pad|>") == 3
    assert prompt.endswith("language English<asr_text>")
    assert parse_output("language French<asr_text> Bonjour.", None) == ("French", "Bonjour.")
    assert parse_output("language None<asr_text>", None) == (None, "")
    assert parse_output("text", None) == (None, "text")
    assert language_name("english") == "English"
    with pytest.raises(ValueError):
        language_name("en")
    with pytest.raises(ValueError):
        QwenASROptions(context=1)


def test_ambiguous_checkpoint_json_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"thinker_config": {"model_type": "qwen3_asr", "model_type": "other"}}')
    with pytest.raises(ValueError, match="duplicate key"):
        read_json(path)


def test_alignment_units_repair_and_missing_assets(tmp_path):
    assert text_units("don't, co-op 北京abc 中文。", "English", tmp_path) == (
        "don't",
        "coop",
        "北",
        "京",
        "abc",
        "中",
        "文",
    )
    assert fix_timestamps([]) == []
    assert fix_timestamps([0, 80, 160, 80, 240]) == [0, 80, 160, 160, 240]
    assert fix_timestamps([300, 80, 90, 100, 200]) == [80, 80, 90, 100, 200]
    with pytest.raises(ValueError, match="supplemental asset"):
        text_units("한국", "Korean", tmp_path)


def session(monkeypatch):
    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: SimpleNamespace(ids=list(range(len(text)))),
        decode=lambda ids, **kw: "x" * len(ids),
    )
    native = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_asr"))
    bundle = LoadedQwenAudio(native, tokenizer, {}, "template", Path("/fixture"), "pin")
    monkeypatch.setattr("mlx_one.audio.streaming.provenance", lambda _: {})
    result = QwenAudioStream(
        bundle, model_name="fixture", sample_rate=16000, language=None, options=None
    )
    calls = []

    def decode(*, final_tail):
        calls.append((len(result._audio), final_tail, result._prefix(final_tail)))
        result._raw = "abcdefghij"
        result._last_text = f"hypothesis-{len(calls)}"
        result._chunks += 1

    monkeypatch.setattr(result, "_decode", decode)
    return result, calls, bundle


def test_explicit_pcm16_stream_scaling(monkeypatch):
    stream, calls, _ = session(monkeypatch)
    pcm = SimpleNamespace(dtype="int16", ndim=1, tolist=lambda: [-32768, 32767])

    class PCM16:
        dtype, ndim = pcm.dtype, pcm.ndim

        def __len__(self):
            return 2

        def tolist(self):
            return pcm.tolist()

    assert stream.push(PCM16()) == ()
    assert list(stream._pending) == [-1.0, 32767 / 32768]
    assert not calls


def test_fragmentation_rollback_flush_and_reset(monkeypatch):
    stream, calls, _ = session(monkeypatch)
    assert stream.push([0.0] * 31999) == ()
    first = stream.push([0.0])
    assert first[0].kind == "partial" and first[0].text == "hypothesis-1"
    stream.push([0.0] * 64000)
    assert calls[2] == (96000, False, "xxxxx")
    assert stream.push([0.0] * 10) == ()
    final = stream.finish()
    assert calls[-1] == (96010, True, "xxxxx")
    assert final.kind == "final" and final.audio_seconds == 96010 / 16000
    assert not stream._audio and final.confidence is None and final.timestamps is None
    with pytest.raises(RuntimeError, match="reset"):
        stream.push([0.0])
    stream.reset()
    assert stream.finish().text == ""
    stream.close()
    with pytest.raises(RuntimeError, match="closed"):
        stream.reset()


def test_validation_limit_cancellation_and_isolation(monkeypatch):
    one, _, bundle = session(monkeypatch)
    two, _, _ = session(monkeypatch)
    for pcm in ([float("nan")], [[0.0]], b"pcm"):
        with pytest.raises(ValueError):
            one.push(pcm)
    assert one._received == 0
    with pytest.raises(AudioResourceLimitError, match="30 seconds"):
        one.push([0.0] * 480001)
    one._cancelled = lambda: True
    with pytest.raises(RuntimeError, match="cancelled"):
        one.push([])
    assert one._closed and two._bundle is not None and bundle.model is not None
    assert two.finish().kind == "final"


def test_align_cli_export_and_overwrite_refusal(tmp_path, monkeypatch):
    audio, text, output = [tmp_path / name for name in ("audio.wav", "text.txt", "alignment.json")]
    audio.touch()
    text.write_text("hello")
    captured = {}

    def align(model, audio, transcript, **kwargs):
        captured.update(kwargs)
        return AudioAlignment(
            (AlignmentUnit("hello", 0, 0.08),), transcript, "English", 1, model, "pin", 0.08
        )

    monkeypatch.setattr("mlx_one.audio.align_audio", align)
    args = [
        "align",
        "model",
        str(audio),
        "--text-file",
        str(text),
        "--language",
        "English",
        "--output",
        str(output),
        "--offline",
        "--revision",
        "pin",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["task"] == "align" and captured["offline"]
    previous = output.read_bytes()
    assert CliRunner().invoke(main, args).exit_code != 0
    assert output.read_bytes() == previous


def test_stream_cli_complete_hypotheses_and_family_options(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.touch()
    captured = {}

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def push(self, pcm):
            return (AudioStreamEvent("partial", "old hypothesis", 0, 2, "model", "pin"),)

        def finish(self):
            return AudioStreamEvent("final", "replacement hypothesis", 1, 2.1, "model", "pin")

    def open_stream(*args, **kwargs):
        captured.update(kwargs)
        return Session()

    @contextmanager
    def decode(*args, **kwargs):
        yield iter([[0.0]])

    monkeypatch.setattr("mlx_one.audio.open_audio_stream", open_stream)
    monkeypatch.setattr("mlx_one.audio.streaming.decoded_pcm_chunks", decode)
    args = [
        "transcribe",
        "model",
        str(audio),
        "--stream",
        "--json-output",
        "--context",
        "names",
        "--offline",
        "--revision",
        "pin",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.output.splitlines()]
    assert [e["kind"] for e in events] == ["partial", "final"]
    assert events[-1]["text"] == "replacement hypothesis"
    assert captured["sample_rate"] == 16000 and captured["offline"]
    assert captured["options"].context == "names"
    assert CliRunner().invoke(main, args + ["--beam-size", "1"]).exit_code != 0


def test_incremental_decoder_cleanup_and_malformed_pcm(tmp_path, monkeypatch):
    source = tmp_path / "audio.wav"
    source.touch()
    processes = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(b"\x00" * 12)
            self.terminated = False
            self.completed = False
            processes.append(self)

        def poll(self):
            return 0 if self.completed else None

        def terminate(self):
            self.terminated = True

        def wait(self, **kwargs):
            self.completed = True
            return 0

    monkeypatch.setattr("mlx_one.audio.streaming.shutil.which", lambda _: "/ffmpeg")
    monkeypatch.setattr("mlx_one.audio.streaming.subprocess.Popen", Process)
    with decoded_pcm_chunks(source, chunk_samples=1) as chunks:
        assert len(next(chunks)) == 1
    assert processes[-1].stdout.closed and processes[-1].terminated
    with decoded_pcm_chunks(source) as chunks:
        assert sum(len(chunk) for chunk in chunks) == 3
    assert processes[-1].completed and not processes[-1].terminated
    with decoded_pcm_chunks(source) as chunks:
        processes[-1].stdout = io.BytesIO(b"bad")
        with pytest.raises(RuntimeError, match="malformed"):
            next(chunks)
    assert processes[-1].terminated
