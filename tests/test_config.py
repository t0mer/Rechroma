from pathlib import Path

from app.config import load_settings


def test_defaults():
    s = load_settings(config_path=None)
    assert s.device == "auto"
    assert s.models_dir == Path("/data/models")


def test_yaml_below_env(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\nlog_level: debug\n")
    monkeypatch.setenv("RECHROMA_DEVICE", "cuda")
    s = load_settings(config_path=cfg)
    assert s.device == "cuda"  # env beats yaml
    assert s.log_level == "debug"  # yaml still applies where env absent


def test_cli_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RECHROMA_DEVICE", "cuda")
    s = load_settings(config_path=None, device="cpu")
    assert s.device == "cpu"  # explicit override beats env


def test_yaml_used_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("RECHROMA_DEVICE", raising=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("device: cpu\n")
    s = load_settings(config_path=cfg)
    assert s.device == "cpu"
