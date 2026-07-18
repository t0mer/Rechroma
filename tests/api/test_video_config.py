from pathlib import Path

from app.config import Settings
from app.main import video_caps_from_settings


def test_video_defaults():
    s = Settings(data_dir=Path("/tmp/x"))
    assert s.video_enabled is True
    assert s.video_max_seconds == 30
    assert s.video_render_factor == 21
    assert s.video_workspace_dir == Path("/tmp/x/video")


def test_animate_defaults():
    s = Settings(data_dir=Path("/tmp/x"))
    assert s.animate_enabled is True
    assert s.animate_driver == "subtle"
    assert s.animate_workspace_dir == Path("/tmp/x/animate")


def test_caps_from_settings():
    s = Settings(data_dir=Path("/tmp/x"), video_max_seconds=10)
    caps = video_caps_from_settings(s)
    assert caps.max_seconds == 10
    assert caps.smoothing_window == 5
    assert caps.render_factor == 21
