"""Version string, read from the ``VERSION`` file (build-injected in images)."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def get_version() -> str:
    try:
        return _VERSION_FILE.read_text().strip()
    except OSError:
        return "0.0.0"


__version__ = get_version()
