"""Network-free task route validation and lease/error behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mlx_one.server.batch1 import install_native_task_routes


class Manager:
    active = 0

    def model_info(self):
        return SimpleNamespace(id="candidate")

    @contextmanager
    def model_lease(self):
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1


@pytest.mark.parametrize("failure", [False, True])
def test_text_to_text_route_releases_lease_on_result_and_failure(failure):
    manager = Manager()

    def run(name, task, payload, *, cancel):
        assert manager.active == 1
        assert name == "primary" and task == "text-to-text-generation"
        assert payload["texts"] == "translate English to German: hello"
        assert not cancel.is_set()
        if failure:
            raise ValueError("unsupported input profile")
        return SimpleNamespace(output=SimpleNamespace(to_dict=lambda: {"texts": ["Hallo"]}))

    app = FastAPI()
    install_native_task_routes(
        app,
        manager,
        {"text-to-text-generation": SimpleNamespace(run_task=run)},
        Depends(lambda: None),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks/text-to-text-generation",
            json=dict(model="candidate", input="translate English to German: hello"),
        )
        assert response.status_code == (422 if failure else 200)
        assert manager.active == 0
        assert (
            client.post(
                "/v1/tasks/text-to-text-generation", json=dict(model="wrong", input="hello")
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/v1/tasks/text-to-text-generation",
                json=dict(model="candidate", input="hello", max_new_tokens=True),
            ).status_code
            == 422
        )


def test_image_route_rejects_paths_and_speech_capability_is_explicit():
    def must_not_execute(*args, **kwargs):
        raise AssertionError("invalid image reached native execution")

    app, manager = FastAPI(), Manager()
    install_native_task_routes(
        app,
        manager,
        {"image-classification": SimpleNamespace(run_task=must_not_execute)},
        Depends(lambda: None),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks/image-classification",
            json=dict(model="candidate", images=["file:///private/image.png"]),
        )
        assert response.status_code == 422 and manager.active == 0
        response = client.post(
            "/v1/tasks/speech-generation", json=dict(model="candidate", input="hello")
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "capability_error"


def test_vit_actor_cost_counts_patches_instead_of_pixels():
    from mlx_one.engine.task_runners.batch1 import ViTAdapter

    bundle = SimpleNamespace(
        architecture="vit",
        model=SimpleNamespace(
            config=SimpleNamespace(image_size=224, patch_size=16, num_labels=1000)
        ),
    )
    request = ViTAdapter(bundle).prepare("image-classification", {"images": ["a", "b"]})
    assert request.lengths == (197, 197)


def test_kokoro_reserves_measured_waveform_working_memory():
    from mlx_one.engine.task_runners.batch1 import KokoroAdapter

    request = KokoroAdapter(SimpleNamespace(architecture="kokoro")).prepare(
        "speech-generation", {"text": "Hello."}
    )
    assert request.transient_bytes >= 3 * 1024**3
