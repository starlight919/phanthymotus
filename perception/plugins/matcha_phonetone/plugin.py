"""MCP/ROS lifecycle for the verified Matcha PhoneTone TensorRT runtime."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from plugins.tts import TTSAdapter, _TTSNode

from .adapter import MatchaTensorRTAdapter


log = logging.getLogger(__name__)


class MatchaTensorRTTTSPlugin:
    """Expose Matcha TensorRT behind the shared public ``tts`` tool."""

    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg = dict(plugin_cfg)
        self._executor = executor
        self._adapter: Optional[MatchaTensorRTAdapter] = None
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._nodes: dict[str, _TTSNode] = {}
        self._nodes_lock = threading.RLock()

    def _ensure_adapter(self) -> MatchaTensorRTAdapter:
        if self._adapter is not None:
            return self._adapter
        with self._load_lock:
            if self._adapter is not None:
                return self._adapter
            try:
                from utils.model_downloader import ensure_matcha_trt_model

                engine_dir = ensure_matcha_trt_model(
                    self._cfg.get("model_dir", "/models/matcha-trt")
                )
                self._adapter = MatchaTensorRTAdapter(
                    engine_dir,
                    speed=float(self._cfg.get("speed", 1.0)),
                )
                self._load_error = None
            except Exception as error:
                self._load_error = str(error)
                log.exception("[matcha_trt] failed to load runtime")
                raise RuntimeError("Matcha TensorRT runtime load failed") from error
            return self._adapter

    def get_tools(self) -> list:
        from plugins.tts import TOOLS

        return TOOLS

    def _dispose_node(self, key: str) -> dict:
        node = self._nodes.pop(key)
        result = node.stop()
        self._executor.remove_node(node)
        node.destroy_node()
        return result

    def _create_node(self, key: str, input_topic: str) -> _TTSNode:
        suffix = key.replace("/", "_").replace("-", "_")
        node = _TTSNode(input_topic or None, self._ensure_adapter(), suffix)
        self._executor.add_node(node)
        self._nodes[key] = node
        return node

    def _info(self, instance_id: str, input_topic: str) -> dict:
        if self._load_error:
            return {
                "name": "Matcha PhoneTone TensorRT",
                "manufacture": "Embodied",
                "model": "matcha-trt",
                "state": "error",
                "error": self._load_error,
                "desc": self._load_error,
            }
        with self._nodes_lock:
            node = self._nodes.get(instance_id) if instance_id else None
            state = "running" if any(n.state == "running" for n in self._nodes.values()) else "idle"
        output_topic = f"{input_topic}/tts" if input_topic else "/perception/tts"
        if node is not None:
            return {
                "name": "Matcha PhoneTone TensorRT",
                "manufacture": "Embodied",
                "model": "matcha-trt",
                **node._status_dict(),
                "desc": "Matcha PhoneTone TensorRT text-to-speech",
            }
        return {
            "name": "Matcha PhoneTone TensorRT",
            "manufacture": "Embodied",
            "model": "matcha-trt",
            "state": state,
            "topic_in": ([{"topic": input_topic, "format": "data/json", "desc": ""}]
                         if input_topic else []),
            "topic_out": [{"topic": output_topic, "format": "audio/pcm-16k", "desc": ""}],
            "desc": "Matcha PhoneTone TensorRT text-to-speech",
        }

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")
        input_topic = args.get("input_topic") or ""

        if action == "info":
            return self._info(instance_id, input_topic)
        if action == "start":
            try:
                self._ensure_adapter()
            except RuntimeError:
                return {"state": "error", "message": self._load_error}
            key = instance_id or input_topic or "_default"
            with self._nodes_lock:
                if key in self._nodes and input_topic != self._nodes[key]._input_topic:
                    self._dispose_node(key)
                node = self._nodes.get(key) or self._create_node(key, input_topic)
                return node.start()
        if action == "stop":
            with self._nodes_lock:
                if instance_id and instance_id in self._nodes:
                    return self._dispose_node(instance_id)
                if not instance_id:
                    for key in list(self._nodes):
                        self._dispose_node(key)
            return {"state": "idle"}
        if action == "speak":
            text = (args.get("text") or "").strip()
            if not text:
                raise ValueError("text is required")
            try:
                self._ensure_adapter()
            except RuntimeError:
                return {"state": "error", "message": self._load_error}
            with self._nodes_lock:
                node = next((n for n in self._nodes.values() if n.state == "running"), None)
                if node is None:
                    key = instance_id or "_default"
                    node = self._nodes.get(key) or self._create_node(key, input_topic)
                    node.start()
            node.enqueue(text, trace_id=args.get("_trace_id", ""))
            return {"status": "queued", "chars": len(text)}
        if action == "config":
            if "speed" in args:
                speed = float(args["speed"])
                self._cfg["speed"] = speed
                if self._adapter is not None:
                    self._adapter.set_speed(speed)
            return {"status": "configured"}
        if action == "interrupt":
            with self._nodes_lock:
                nodes = ([self._nodes[instance_id]] if instance_id in self._nodes else []) if instance_id else list(self._nodes.values())
            return {"status": "interrupted", "nodes": len(nodes),
                    "cleared": sum(node.interrupt().get("cleared", 0) for node in nodes)}
        return None

    def synthesize_raw(self, text: str) -> bytes:
        return self._ensure_adapter().synthesize(text)
