"""Optional VITS2 TTS plugin with TensorRT and ONNX backends.

Keep ROS imports lazy so the frontend/adapter can be reused by offline
evaluation without requiring ``audio_msgs`` or ``rclpy``.
"""

__all__ = ["TTSPlugin"]


def __getattr__(name):
    if name == "TTSPlugin":
        from .plugin import TTSPlugin

        return TTSPlugin
    raise AttributeError(name)
