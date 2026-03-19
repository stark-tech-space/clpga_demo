"""Golf ball tracker — track and crop golf balls from video using SAM3."""


def __getattr__(name: str):
    """Lazy imports — avoids ImportError before pipeline.py exists."""
    if name == "process_video":
        from clpga_demo.pipeline import process_video
        return process_video
    if name == "process_stream":
        from clpga_demo.pipeline import process_stream
        return process_stream
    raise AttributeError(f"module 'clpga_demo' has no attribute {name!r}")


__all__ = ["process_video", "process_stream"]
