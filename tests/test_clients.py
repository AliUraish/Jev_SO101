import json

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

from jev_arm.clients import AstraVision, JevJudge, questions
from jev_arm.demo import answers_for, demo_config, scene_for
from jev_arm.models import Answers, Scene


def test_jev_wire_contract(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    captured = []

    def handler(request):
        captured.append(request)
        answers = answers_for("approach").model_dump()
        answers["task"]["probabilities"] = {"approach": 1.0, "observe": 0.0, "hold": 0.0}
        return httpx.Response(200, json={"model": "jev-latest", "answers": answers})

    observed = scene_for("approach").model_dump()
    observed["hazards"] = ["Loose cable near the pickup path"]
    observed["objects"] = [{"object_id": "cable", "kind": "other", "description": "Loose cable near the pickup path",
                            "regions": [{"view": "overhead", "x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}]}]
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        answers = JevJudge(demo_config(), http).decide(Scene.model_validate(observed), {"expected_skill": "approach"})
    payload = json.loads(captured[0].content)
    assert str(captured[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert captured[0].headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "jev-latest"
    assert payload["questions"] == questions("approach")
    assert set(payload["questions"]["task"]["criteria"]) == {"approach", "observe", "hold"}
    assert payload["questions"]["task"]["type"] == "choice"
    assert payload["questions"]["unsafe"]["type"] == "noul"
    assert payload["questions"]["done"]["type"] == "noul"
    evidence = payload["state"]["scene"]
    assert evidence["hazards"] == observed["hazards"]
    assert evidence["objects"][0]["description"] == observed["objects"][0]["description"]
    assert "regions" not in evidence["objects"][0]
    assert {k: v for k, v in evidence.items() if k != "objects"} == {k: v for k, v in observed.items() if k != "objects"}
    assert answers.task.choice == "approach"


def test_jev_rejects_options_not_offered_for_current_stage(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    def handler(request):
        return httpx.Response(200, json={"answers": answers_for("approach").model_dump()})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ValueError, match="eligible skills"):
            JevJudge(demo_config(), http).decide(scene_for("approach"), {"expected_skill": "approach"})


@pytest.mark.parametrize("status", [401, 429, 500])
def test_jev_http_failures_propagate_without_retries(monkeypatch, status):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "test failure"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            JevJudge(demo_config(), http).decide(scene_for("approach"), {"expected_skill": "approach"})
    assert len(requests) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "0.9", True])
def test_invalid_probabilities_rejected(value):
    data = answers_for("approach").model_dump()
    data["unsafe"]["noul"] = value
    with pytest.raises(ValidationError):
        Answers.model_validate(data)


def test_missing_answers_and_unknown_skill_rejected():
    data = answers_for("approach").model_dump()
    del data["unsafe"]
    with pytest.raises(ValidationError):
        Answers.model_validate(data)
    data = answers_for("approach").model_dump()
    data["task"]["choice"] = "invented_action"
    with pytest.raises(ValidationError):
        Answers.model_validate(data)


def test_astra_actual_sdk_serializes_two_images_and_parses_schema():
    captured = []
    scene = scene_for("approach")

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-6-astra", "status": "completed",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": scene.model_dump_json(), "annotations": []}]}],
        })

    with OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0) as client:
        result = AstraVision(demo_config(), client).describe({"overhead": b"test-overhead", "wrist": b"test-wrist"}, {"expected_skill": "approach"})
    payload = captured[0]
    assert result == scene
    assert payload["model"] == "gpt-6-astra"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    images = [p for p in payload["input"][1]["content"] if p["type"] == "input_image"]
    assert len(images) == 2
    assert all(p["image_url"].startswith("data:image/jpeg;base64,") for p in images)


@pytest.mark.parametrize("status,content", [
    ("incomplete", [{"type": "output_text", "text": "", "annotations": []}]),
    ("completed", [{"type": "refusal", "refusal": "Cannot determine scene"}]),
])
def test_astra_incomplete_or_refusal_rejected(status, content):
    def handler(request):
        return httpx.Response(200, json={"id": "resp_test", "object": "response", "created_at": 0, "model": "gpt-6-astra", "status": status,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": status, "content": content}]})
    with OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0) as client:
        with pytest.raises(ValueError):
            AstraVision(demo_config(), client).describe({"overhead": b"x", "wrist": b"x"}, {})
