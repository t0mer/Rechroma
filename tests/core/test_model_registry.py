import re

from app.core.model_registry import REGISTRY, ModelEntry, get_entry

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_artistic_and_stable_checksums_pinned():
    assert (
        get_entry("deoldify-artistic").sha256
        == "3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477"
    )
    assert (
        get_entry("deoldify-stable").sha256
        == "ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af"
    )


def test_entries_are_wellformed():
    seen = set()
    for name, e in REGISTRY.items():
        assert isinstance(e, ModelEntry)
        assert e.filename.endswith(".pth")
        assert e.filename not in seen, f"duplicate filename {e.filename}"
        seen.add(e.filename)
        assert HEX64.match(e.sha256) or e.sha256 == "", f"{name} bad sha"
        assert e.url.startswith("https://")
        assert e.size_bytes > 0


def test_get_entry_unknown_lists_names():
    try:
        get_entry("nope")
    except KeyError as exc:
        assert "deoldify-artistic" in str(exc)
    else:
        raise AssertionError("expected KeyError")
