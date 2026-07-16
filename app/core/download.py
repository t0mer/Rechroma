"""Lazy, checksum-verified weight downloads.

Weights are never baked into the image (CLAUDE.md §2): they are fetched to
``models_dir`` on first use and verified against the registry SHA-256. A file
that is already present and valid is reused (pre-seeding / air-gapped support).
``MODEL_BASE_URL`` redirects the source to a self-hosted mirror.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx
from loguru import logger

from .model_registry import ModelEntry

Fetcher = Callable[[str, Path], None]

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_url(entry: ModelEntry, base_url: str | None) -> str:
    """Return the download URL, honouring a ``MODEL_BASE_URL`` mirror override."""
    if base_url:
        return f"{base_url.rstrip('/')}/{entry.filename}"
    return entry.url


def _http_fetch(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(_CHUNK):
                f.write(chunk)


def ensure_weights(
    entry: ModelEntry,
    models_dir: Path,
    base_url: str | None = None,
    fetch: Fetcher = _http_fetch,
) -> Path:
    """Return the local path to ``entry``'s weights, downloading if necessary.

    Skips the download when the file is already present and its checksum matches.
    Downloads to a ``.part`` file, verifies the SHA-256, then atomically renames.
    Raises ``RuntimeError`` on a checksum mismatch (removing the partial file).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    final = models_dir / entry.filename

    if final.exists() and entry.sha256 and sha256_file(final) == entry.sha256:
        logger.debug("weights present and verified: {}", entry.filename)
        return final

    url = resolve_url(entry, base_url)
    part = final.with_suffix(final.suffix + ".part")
    logger.info("downloading {} from {}", entry.filename, url)
    try:
        fetch(url, part)
        got = sha256_file(part)
        if entry.sha256 and got != entry.sha256:
            raise RuntimeError(
                f"checksum mismatch for {entry.filename}: "
                f"expected {entry.sha256}, got {got}"
            )
        part.replace(final)
    finally:
        part.unlink(missing_ok=True)
    return final
