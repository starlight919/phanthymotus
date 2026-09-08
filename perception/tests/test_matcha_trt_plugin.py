from __future__ import annotations

import pytest

from vision_stubs import _FakeExecutor  # noqa: F401

import plugins.matcha_phonetone.plugin as plugin_module  # noqa: E402


class _Adapter:
    def __init__(self, engine_dir, speed):
        self.engine_dir = engine_dir
        self.speed = speed
        self.closed = False

    def set_speed(self, speed):
        self.speed = float(speed)

    def synthesize(self, text):
        return b"\x00\x00" * 4

    def synthesize_stream(self, text):
        yield self.synthesize(text)

    def close(self):
        self.closed = True


@pytest.fixture
def plugin(monkeypatch):
    calls = []

    import utils.model_downloader as downloader

    monkeypatch.setattr(
        downloader,
        "ensure_matcha_trt_model",
        lambda model_dir: calls.append(model_dir) or f"{model_dir}/engines/jp61",
        raising=False,
    )
    monkeypatch.setattr(plugin_module, "MatchaTensorRTAdapter", _Adapter)
    return (
        plugin_module.MatchaTensorRTTTSPlugin(
            {"model_dir": "/models/matcha-trt", "speed": 1.25}, _FakeExecutor()
        ),
        calls,
    )


def test_info_does_not_load_the_runtime(plugin):
    instance, calls = plugin

    info = instance.dispatch("tts", {"action": "info"})

    assert info["state"] == "idle"
    assert calls == []


def test_start_loads_matcha_once_and_creates_a_shared_tts_node(plugin):
    instance, calls = plugin

    result = instance.dispatch("tts", {
        "action": "start", "instance_id": "card-a", "input_topic": "/say"
    })

    assert result["state"] == "running"
    assert calls == ["/models/matcha-trt"]
    assert instance._adapter.engine_dir == "/models/matcha-trt/engines/jp61"
    assert instance._adapter.speed == pytest.approx(1.25)
    assert instance.dispatch("tts", {"action": "info", "instance_id": "card-a"})[
        "topic_out"
    ] == [{
        "topic": "/say/tts", "format": "audio/pcm-16k", "desc": "synthesized PCM audio"
    }]


def test_config_updates_a_loaded_adapter_without_reloading(plugin):
    instance, calls = plugin
    instance.dispatch("tts", {"action": "start"})
    adapter = instance._adapter

    assert instance.dispatch("tts", {"action": "config", "speed": 0.8}) == {
        "status": "configured"
    }
    assert calls == ["/models/matcha-trt"]
    assert instance._adapter is adapter
    assert adapter.speed == pytest.approx(0.8)


def test_stop_disposes_nodes_but_keeps_the_resident_runtime(plugin):
    instance, _ = plugin
    instance.dispatch("tts", {"action": "start", "instance_id": "card-a"})
    adapter = instance._adapter

    assert instance.dispatch("tts", {"action": "stop", "instance_id": "card-a"}) == {
        "state": "idle"
    }
    assert instance._nodes == {}
    assert adapter.closed is False


def test_model_load_failure_is_reported_without_creating_a_node(monkeypatch):
    import utils.model_downloader as downloader

    monkeypatch.setattr(
        downloader,
        "ensure_matcha_trt_model",
        lambda model_dir: (_ for _ in ()).throw(RuntimeError("archive checksum mismatch")),
        raising=False,
    )
    instance = plugin_module.MatchaTensorRTTTSPlugin({}, _FakeExecutor())

    result = instance.dispatch("tts", {"action": "start"})

    assert result == {"state": "error", "message": "archive checksum mismatch"}
    assert instance._nodes == {}
