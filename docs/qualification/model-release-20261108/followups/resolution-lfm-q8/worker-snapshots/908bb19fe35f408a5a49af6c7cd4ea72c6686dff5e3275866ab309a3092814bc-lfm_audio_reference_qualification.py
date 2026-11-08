"""Run an isolated, publisher-pinned LFM forced-history diagnostic."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def diagnose(model, fixtures, report):
    import numpy as np
    from scipy.io import wavfile

    root = Path(__file__).resolve().parents[2]
    work = fixtures.parent
    helper = root / "scripts/qualification/lfm_audio_reference_probe.py"
    outputs = {}
    for backend, python in [("native", sys.executable), ("reference", report["reference_python"])]:
        output = model.parent / f"lfm-{backend}-forced.npz"
        log = Path(report["evidence_directory"]) / f"lfm-{backend}-forced.log"
        command = [
            str(python),
            str(helper),
            "--backend",
            backend,
            "--model",
            str(model),
            "--audio",
            str(fixtures / "audio/librispeech-10s.wav"),
            "--trace",
            str(model.parent / "lfm-events.json"),
            "--output",
            str(output),
        ]
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
        env.pop("PYTHONPATH", None)
        if backend == "reference":
            env["PYTHONPATH"] = str(
                work / "reference-sources/mlx-audio-76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d"
            )
        with log.open("w") as stream:
            result = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200
            )
        report.setdefault("lfm_reference_runs", []).append(
            dict(backend=backend, exit_code=result.returncode, log=log.name)
        )
        if result.returncode:
            raise ValueError(f"LFM {backend} reference diagnostic failed; see {log.name}")
        outputs[backend] = output
    with np.load(outputs["native"]) as native, np.load(outputs["reference"]) as reference:
        metrics = {}
        for key in native.files:
            a, b = native[key], reference[key]
            metric = dict(native_shape=list(a.shape), reference_shape=list(b.shape))
            if a.shape == b.shape:
                metric.update(
                    max_absolute_error=float(np.max(np.abs(a - b))),
                    mean_absolute_error=float(np.mean(np.abs(a - b))),
                )
                if key.endswith("logits"):
                    metric["argmax_agreement"] = float(
                        np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1))
                    )
            metrics[key] = metric
        public_path = Path(report["evidence_directory"]) / "audio-outputs/lfm2_audio-sts_10s.wav"
        if public_path.exists():
            _, public = wavfile.read(public_path)
            if np.issubdtype(public.dtype, np.integer):
                public = public.astype(np.float32) / (1 << (np.iinfo(public.dtype).bits - 1))
            full = native["codec_waveform"]
            metrics["public_stream_vs_full_codec"] = dict(
                same_shape=public.shape == full.shape,
                max_absolute_error=float(np.max(np.abs(public - full)))
                if public.shape == full.shape
                else None,
                correlation=float(np.corrcoef(public, full)[0, 1])
                if public.shape == full.shape
                else None,
            )
    report["lfm_reference_diagnostic"] = dict(
        source_revision="76f35931cfe7db8f1f6b604d7b205ffe9d74ff8d",
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        scope=(
            "Independent frontend and stock model modules, forced native history; "
            "numerical diagnosis only"
        ),
        metrics=metrics,
    )
