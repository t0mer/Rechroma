"""CLI smoke-test entrypoint for the colorization core.

Usage::

    python -m app.core.cli colorize IN OUT [--model artistic|stable]
        [--render-factor N] [--device auto|cuda|cpu]

Kept intentionally small — the REST API and Telegram bot (later slices) are the
real front doors; this exists so the core is runnable and demonstrable on its
own (CLAUDE.md §13 stage 1).
"""

import argparse
from pathlib import Path

from loguru import logger
from PIL import Image, UnidentifiedImageError

from ..config import load_settings
from .colorizer import DeOldifyColorizer
from .device import device_defaults, resolve_device
from .pipeline import PipelineOptions, build_steps, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.core.cli", description="Rechroma processing core")
    sub = parser.add_subparsers(dest="command", required=True)

    col = sub.add_parser("colorize", help="colorize a black & white image")
    col.add_argument("input", type=Path)
    col.add_argument("output", type=Path)
    col.add_argument("--model", choices=["artistic", "stable"], default="artistic")
    col.add_argument("--render-factor", type=int, default=None)
    col.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)

    proc = sub.add_parser("process", help="run a full pipeline preset (restore/colorize/upscale)")
    proc.add_argument("input", type=Path)
    proc.add_argument("output", type=Path)
    proc.add_argument("--preset", choices=["colorize", "restore", "full"], default="full")
    proc.add_argument("--model", choices=["artistic", "stable"], default="artistic")
    proc.add_argument("--render-factor", type=int, default=None)
    proc.add_argument("--upscale", type=int, choices=[2, 4], default=None)
    proc.add_argument("--no-restore", action="store_true", help="skip face restoration")
    proc.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    return parser


def _run_colorize(args: argparse.Namespace) -> int:
    settings = load_settings(device=args.device, render_factor=args.render_factor)
    try:
        device = resolve_device(settings.device)
    except RuntimeError as e:
        logger.error("{}", e)
        return 2
    render_factor = settings.render_factor or device_defaults(device).render_factor
    logger.info("device={} model={} render_factor={}", device.type, args.model, render_factor)

    try:
        image = Image.open(args.input)
        image.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        logger.error("cannot read input image {}: {}", args.input, e)
        return 1

    colorizer = DeOldifyColorizer(
        model=args.model,
        device=settings.device,
        models_dir=settings.models_dir,
        base_url=settings.model_base_url,
    )
    result = colorizer.colorize(image, render_factor=render_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    logger.info("wrote {}", args.output)
    return 0


def _run_process(args: argparse.Namespace) -> int:
    settings = load_settings(device=args.device)
    try:
        device = resolve_device(settings.device)
    except RuntimeError as e:
        logger.error("{}", e)
        return 2
    options = PipelineOptions(
        preset=args.preset,
        colorizer_model=args.model,
        render_factor=args.render_factor,
        upscale=args.upscale,
        restore_faces=not args.no_restore,
    )
    try:
        image = Image.open(args.input)
        image.load()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        logger.error("cannot read input image {}: {}", args.input, e)
        return 1

    steps = build_steps(options, settings.device, settings.models_dir, settings.model_base_url)
    logger.info("device={} steps={}", device.type, [s.name for s in steps])
    result = run_pipeline(steps, image.convert("RGB"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    logger.info("wrote {}", args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "colorize":
        return _run_colorize(args)
    if args.command == "process":
        return _run_process(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
