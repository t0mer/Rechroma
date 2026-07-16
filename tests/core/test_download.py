import hashlib
from pathlib import Path

import pytest

from app.core.download import ensure_weights, resolve_url
from app.core.model_registry import ModelEntry


def _entry(sha: str, name: str = "m.pth") -> ModelEntry:
    return ModelEntry(name, sha, 4, "https://example.com/m.pth", "MIT", "test")


def test_resolve_url_uses_mirror():
    e = _entry("x")
    assert resolve_url(e, "https://mir.local/models/") == "https://mir.local/models/m.pth"
    assert resolve_url(e, "https://mir.local/models") == "https://mir.local/models/m.pth"
    assert resolve_url(e, None) == "https://example.com/m.pth"


def test_skips_when_present_and_valid(tmp_path):
    data = b"abcd"
    (tmp_path / "m.pth").write_bytes(data)
    e = _entry(hashlib.sha256(data).hexdigest())
    calls = []

    def fetch(url, dest):
        calls.append(url)

    p = ensure_weights(e, tmp_path, fetch=fetch)
    assert p == tmp_path / "m.pth"
    assert calls == []


def test_downloads_then_verifies(tmp_path):
    data = b"wxyz"
    e = _entry(hashlib.sha256(data).hexdigest())

    def fetch(url, dest):
        Path(dest).write_bytes(data)

    p = ensure_weights(e, tmp_path, fetch=fetch)
    assert p.read_bytes() == data


def test_redownloads_when_present_but_corrupt(tmp_path):
    good = b"good"
    (tmp_path / "m.pth").write_bytes(b"corrupt")
    e = _entry(hashlib.sha256(good).hexdigest())

    def fetch(url, dest):
        Path(dest).write_bytes(good)

    p = ensure_weights(e, tmp_path, fetch=fetch)
    assert p.read_bytes() == good


def test_checksum_mismatch_raises_and_cleans(tmp_path):
    e = _entry("0" * 64)

    def fetch(url, dest):
        Path(dest).write_bytes(b"bad!")

    with pytest.raises(RuntimeError, match="checksum"):
        ensure_weights(e, tmp_path, fetch=fetch)
    assert not (tmp_path / "m.pth.part").exists()
