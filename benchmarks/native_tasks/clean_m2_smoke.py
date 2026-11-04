#!/usr/bin/env python3
"""Installed-wheel NER and CLIP smoke; run -I outside the checkout, Hub offline."""

import importlib.util
import io
import json
import socket
import subprocess
import sys
import wave
from pathlib import Path

FORBIDDEN = {"torch", "transformers", "onnxruntime", "mlx_lm", "omlx"}
for name in FORBIDDEN:
    assert importlib.util.find_spec(name) is None, name


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled in installed-wheel smoke")


socket.socket.connect = deny_network
from fastapi.testclient import TestClient  # noqa: E402

import mlx_one  # noqa: E402
from mlx_one.classification import token_classify, zero_shot_classify  # noqa: E402
from mlx_one.server import ModelManager, ServerConfig, create_app  # noqa: E402

ner, clip, image = sys.argv[1:4]
text = "Marie Curie lived in Paris."
results = []
cases = [
    (
        "token-classification",
        ner,
        ["ner", ner, text, "--aggregation-strategy", "simple"],
        {"model": "test", "input": text, "aggregation_strategy": "simple"},
        None,
        token_classify(ner, text, aggregation_strategy="simple").to_dict(),
    ),
    (
        "zero-shot-image-classification",
        clip,
        ["zero-shot-image", clip, image, "--label", "cat", "--label", "dog"],
        {"model": "test", "candidate_labels": json.dumps(["cat", "dog"])},
        {"file": ("cat.png", Path(image).read_bytes(), "image/png")},
        zero_shot_classify(clip, image, ["cat", "dog"]).to_dict(),
    ),
]
if len(sys.argv) >= 5:
    import numpy as np

    from mlx_one import VADEngine

    vad = sys.argv[4]
    wav = Path("/private/tmp/mlx-one-clean-vad-silence.wav")
    data = io.BytesIO()
    with wave.open(data, "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(np.zeros(1025, dtype="<i2").tobytes())
    # Explicit temporary fixture, no server filesystem-path input.
    wav.write_bytes(data.getvalue())
    with VADEngine(vad) as engine:
        expected = engine.detect(data.getvalue()).to_dict()
    cases.append(
        (
            "voice-activity-detection",
            vad,
            ["vad", vad, str(wav)],
            {"model": "test"},
            {"file": ("silence.wav", data.getvalue(), "audio/wav")},
            expected,
        )
    )
for task, checkpoint, cli_args, payload, files, expected in cases:
    expected = json.loads(json.dumps(expected))
    manager = ModelManager(alias="test")
    manager.load(checkpoint, offline=True)
    with TestClient(
        create_app(model_manager=manager, config=ServerConfig(api_keys=("key",), warmup=False))
    ) as client:
        args = {"json": payload} if files is None else {"data": payload, "files": files}
        route = f"/v1/tasks/{task}"
        assert client.post(route, **args).status_code == 401
        client.headers["Authorization"] = "Bearer key"
        for _ in range(2):
            response = client.post(route, **args)
            assert response.status_code == 200, response.text
            assert response.json() == expected
    cli = subprocess.run(
        [sys.executable, "-I", "-m", "mlx_one.cli", *cli_args, "--json-output"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(cli.stdout) == expected
    results.append({"task": task, "python_cli_http_equal": True, "result": expected})
if len(sys.argv) >= 7:
    from mlx_one.vision.document_qa import DocumentQAEngine

    with DocumentQAEngine(sys.argv[5]) as qa:
        answer = qa.ask(
            sys.argv[6],
            "Who is presiding over the TRRF general session? "
            "Answer briefly using only the document.",
        )
    assert answer.answer == "Lee A. Waller"
    results.append({"diagnostic_document_helper": answer.to_dict(), "task_api_admitted": False})
if len(sys.argv) >= 9:
    import mlx.core as mx

    from mlx_one.models.depth.depth_anything import load_depth
    from mlx_one.vision.depth_processing import preprocess_depth

    pixels, size = preprocess_depth(sys.argv[8])
    model = load_depth(sys.argv[7])
    depth = model(mx.array(pixels[None]))["predicted_depth"]
    mx.eval(depth)
    assert bool(mx.all(mx.isfinite(depth)).item())
    results.append(
        {
            "diagnostic_depth_shape": list(depth.shape),
            "relative_depth_only": True,
            "task_api_admitted": False,
        }
    )
loaded = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN)
assert not loaded, loaded
assert "/site-packages/" in mlx_one.__file__, mlx_one.__file__
print(
    json.dumps(
        {"package": mlx_one.__file__, "forbidden_modules": loaded, "results": results}, indent=2
    )
)
