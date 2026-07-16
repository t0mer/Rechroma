import numpy as np
from PIL import Image

from app.core import cli


def test_cli_colorize_writes_output(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(src)

    class FakeColorizer:
        def __init__(self, *a, **k):
            pass

        def colorize(self, img, *, render_factor):
            return img.convert("RGB")

    monkeypatch.setattr(cli, "DeOldifyColorizer", FakeColorizer)
    rc = cli.main(["colorize", str(src), str(dst), "--device", "cpu"])
    assert rc == 0
    assert dst.exists()
    assert Image.open(dst).size == (32, 32)


def test_cli_missing_input_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DeOldifyColorizer", lambda *a, **k: None)
    rc = cli.main(
        ["colorize", str(tmp_path / "nope.png"), str(tmp_path / "o.png"), "--device", "cpu"]
    )
    assert rc != 0


def test_cli_render_factor_defaults_to_device(tmp_path, monkeypatch):
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.fromarray(np.full((16, 16, 3), 100, np.uint8)).save(src)
    seen = {}

    class FakeColorizer:
        def __init__(self, *a, **k):
            pass

        def colorize(self, img, *, render_factor):
            seen["rf"] = render_factor
            return img.convert("RGB")

    monkeypatch.setattr(cli, "DeOldifyColorizer", FakeColorizer)
    rc = cli.main(["colorize", str(src), str(dst), "--device", "cpu"])
    assert rc == 0
    assert seen["rf"] == 25  # CPU default render factor
