"""Pinned registry of model weights — the single source of truth.

Download logic, startup verification and the README credits table are all
derived from this module (CLAUDE.md §3b). Entries whose ``sha256`` is not yet
pinned carry an empty string; ``mirror_models.sh`` produces the checksums to
fill in. All weights are frozen upstream — no newer versions exist.
"""

from dataclasses import dataclass

MB = 1024 * 1024


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable weight file and everything needed to fetch/verify it."""

    filename: str
    sha256: str
    size_bytes: int
    url: str
    license: str
    role: str


REGISTRY: dict[str, ModelEntry] = {
    # --- Colorization ---------------------------------------------------------
    "deoldify-artistic": ModelEntry(
        filename="ColorizeArtistic_gen.pth",
        sha256="3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477",
        size_bytes=255144681,
        url="https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        license="MIT",
        role="DeOldify artistic colorizer (default on GPU)",
    ),
    "deoldify-stable": ModelEntry(
        filename="ColorizeStable_gen.pth",
        sha256="ca9cd7f43fb8b222c9a70f7b292e305a000694b0ff9d2ae4a6747b1a2e1ee5af",
        size_bytes=874066230,
        url="https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth",
        license="MIT",
        role="DeOldify stable colorizer (portraits/landscapes)",
    ),
    "ddcolor-tiny": ModelEntry(
        filename="ddcolor_paper_tiny.pth",
        sha256="",
        size_bytes=200 * MB,
        url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper_tiny.pth",
        license="Apache-2.0",
        role="DDColor tiny (default on CPU)",
    ),
    "ddcolor": ModelEntry(
        filename="ddcolor_modelscope.pth",
        sha256="",
        size_bytes=900 * MB,
        url="https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_modelscope.pth",
        license="Apache-2.0",
        role="DDColor full (GPU alt colorizer)",
    ),
    # --- Face restoration (GFPGAN + deps) ------------------------------------
    "gfpgan": ModelEntry(
        filename="GFPGANv1.4.pth",
        sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        size_bytes=348632874,
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        license="Apache-2.0",
        role="Face restoration",
    ),
    "gfpgan-detection": ModelEntry(
        filename="detection_Resnet50_Final.pth",
        sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        size_bytes=109497761,
        url="https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        license="MIT",
        role="Face detection (GFPGAN dependency, facexlib RetinaFace)",
    ),
    "gfpgan-parsing": ModelEntry(
        filename="parsing_parsenet.pth",
        sha256="3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2",
        size_bytes=85331193,
        url="https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        license="MIT",
        role="Face parsing (GFPGAN dependency, facexlib ParseNet)",
    ),
    # --- Upscale / denoise (Real-ESRGAN) -------------------------------------
    "realesrgan-x4plus": ModelEntry(
        filename="RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        size_bytes=67040989,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        license="BSD-3-Clause",
        role="4x upscale (GPU default)",
    ),
    "realesrgan-x2plus": ModelEntry(
        filename="RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        size_bytes=67061725,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        license="BSD-3-Clause",
        role="2x upscale",
    ),
    "realesr-general-x4v3": ModelEntry(
        filename="realesr-general-x4v3.pth",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        size_bytes=4885111,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        license="BSD-3-Clause",
        role="Lightweight upscale (CPU default)",
    ),
}


def get_entry(name: str) -> ModelEntry:
    """Return the registry entry for ``name`` or raise ``KeyError`` listing known names."""
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}") from e
