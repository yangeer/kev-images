"""Service routing checks without model weights."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from kev import serve


@pytest.fixture
def service(monkeypatch):
    tok = Mock(return_value=SimpleNamespace(input_ids=[1]))
    model = Mock(head=SimpleNamespace(temperature=1.0))
    vision = Mock()
    vision.probs.return_value = ([torch.tensor([0.25, 0.75])], {"tokens": 42, "state_tokens": 30})
    s = serve.Server(SimpleNamespace(requested="test", meta=SimpleNamespace(base="test", lora=16)),
                     tok, model, "cpu", vision=vision)
    monkeypatch.setattr(serve.app.state, "server", s, raising=False)
    return s


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/systemone/separate", "/v1/systemone/permute"])
def test_image_routes_use_vision_without_text_cache(service, path):
    body = {"state": "photo", "images": ["encoded-image"], "questions": {
        "category": {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}}}}
    if path.endswith("/permute"):
        body = {"request": body, "question": "category", "n_perm": 1}
    response = TestClient(serve.app).post(path, json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    if path.endswith("/permute"):
        assert result["runs"][0]["choice"] == "b"
    else:
        assert result["answers"]["category"]["choice"] == "b"
        assert result["usage"]["input_tokens"] == 42
    args, kwargs = service.vision.probs.call_args
    assert args[1] == ["encoded-image"]
    assert kwargs == {"max_state": serve.INFER_MAX_STATE, "max_branch": serve.INFER_MAX_BRANCH}
    service.model.encode.assert_not_called()
    assert not service.prefix_cache and service.prefix_hits == service.prefix_misses == 0


def test_image_errors_are_reported_as_422(service):
    with TestClient(serve.app) as client:
        body = {"state": "photo", "images": ["bad"], "questions": {
            "present": {"type": "noul", "instructions": "Present?"}}}
        service.vision.probs.side_effect = ValueError("invalid image")
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422 and response.json()["detail"] == "invalid image"
        service.vision = None
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 422 and "--vision" in response.json()["detail"]
        assert client.get("/v1/models").json()["models"][0]["vision"] is False


def test_empty_images_keep_text_path(service):
    service.model.encode.return_value = {"seg": [0, 1], "ids": [1, 2]}
    service.model.probs.return_value = [torch.tensor([0.25, 0.75])]
    ps, metrics = service.probs({"state": "text"}, [])
    assert ps == [[0.25, 0.75]] and metrics["tokens"] == 2
    service.vision.probs.assert_not_called()


def test_cli_loads_vision_with_checkpoint_tokenizer(monkeypatch, service):
    import uvicorn

    ck = Mock()
    ck.load.return_value = service.tok, service.model
    ck.load_vision.return_value = service.vision
    monkeypatch.setattr(serve, "Checkpoint", Mock(return_value=ck))
    monkeypatch.setattr(serve, "default_device", lambda: "cpu")
    monkeypatch.setattr("sys.argv", ["kev.serve", "--run", "owner/model", "--vision"])
    monkeypatch.setattr(uvicorn, "run", Mock())
    serve.main()
    ck.load_vision.assert_called_once_with(service.model, service.tok, ck.load.call_args.args[1])
    assert serve.server().vision is service.vision
