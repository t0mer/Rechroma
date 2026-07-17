"""aiogram 3 Telegram bot — a thin client over the shared JobService.

Feature parity with the web UI: send a photo, pick Colorize / Restore / Full via
an inline keyboard, and get the result back as an uncompressed document plus a
compressed preview. Progress is shown by editing a single status message. Access
is gated by the allowlist; per-chat rate limiting is enforced by the JobService
(CLAUDE.md §5). Long-polling by default; webhook mode if configured.
"""

import contextlib
import uuid
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from loguru import logger

from app.config import Settings
from app.core.pipeline import PipelineOptions
from app.jobs.models import JobStatus
from app.jobs.service import JobService, RateLimitError

from .access import is_allowed
from .chat_settings import ChatSettingsStore
from .runner import process_and_wait

_WELCOME = (
    "👋 *Rechroma* brings old photos back to life.\n\n"
    "Send me a black & white or faded photo (as a photo, or as a *file* for full "
    "quality) and pick what to do: Colorize, Restore, or Full.\n\n"
    "/settings — your defaults · /status — your queue · /help"
)
_HELP = (
    "*How to use Rechroma*\n"
    "1. Send a photo (or a file for best quality).\n"
    "2. Tap Colorize, Restore, or Full.\n"
    "3. Get back the restored image as a document + preview.\n\n"
    "Presets: *Colorize* (add colour) · *Restore* (faces + sharpen an already-colour "
    "photo) · *Full* (both).\n"
    "/settings changes your defaults · /status shows your queue position."
)


class BotContext:
    """Shared runtime pieces for handlers (avoids globals)."""

    def __init__(self, settings: Settings, service: JobService) -> None:
        self.settings = settings
        self.service = service
        self.chat_settings = ChatSettingsStore(settings.data_dir / "chat_settings.db")
        self.pending: dict[str, tuple[str, str]] = {}  # token -> (input path, kind)


def _preset_keyboard(token: str, kind: str = "image") -> InlineKeyboardMarkup:
    if kind == "video":
        # Video is colorize-only for v2.
        row = [InlineKeyboardButton(text="🎨 Colorize", callback_data=f"go:colorize:{token}")]
    else:
        row = [
            InlineKeyboardButton(text="🎨 Colorize", callback_data=f"go:colorize:{token}"),
            InlineKeyboardButton(text="✨ Restore", callback_data=f"go:restore:{token}"),
            InlineKeyboardButton(text="🌟 Full", callback_data=f"go:full:{token}"),
        ]
    return InlineKeyboardMarkup(inline_keyboard=[row])


def _classify_media(message: Message) -> tuple[str | None, str | None, int | None]:
    """Return (file_id, kind, size_bytes) for a photo/video/animation/document message."""
    if message.video:
        return message.video.file_id, "video", message.video.file_size
    if message.animation:
        return message.animation.file_id, "video", message.animation.file_size
    if message.photo:
        return message.photo[-1].file_id, "image", message.photo[-1].file_size
    if message.document:
        mime = message.document.mime_type or ""
        if mime.startswith("video/"):
            return message.document.file_id, "video", message.document.file_size
        if mime.startswith("image/"):
            return message.document.file_id, "image", message.document.file_size
    return None, None, None


def build_router(ctx: BotContext) -> Router:
    router = Router()

    def allowed(chat_id: int) -> bool:
        return is_allowed(chat_id, ctx.settings.allowed_chat_ids, ctx.settings.admin_chat_ids)

    async def _guard(message: Message) -> bool:
        if not allowed(message.chat.id):
            await message.answer(
                "Sorry, this bot is private. Ask the administrator to add your chat id "
                f"(`{message.chat.id}`) to the allowlist.",
                parse_mode="Markdown",
            )
            return False
        return True

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        if await _guard(message):
            await message.answer(_WELCOME, parse_mode="Markdown")

    @router.message(Command("help"))
    async def help_cmd(message: Message) -> None:
        if await _guard(message):
            await message.answer(_HELP, parse_mode="Markdown")

    @router.message(Command("settings"))
    async def settings_cmd(message: Message) -> None:
        if not await _guard(message):
            return
        cs = ctx.chat_settings.get(message.chat.id)
        rf = cs.render_factor if cs.render_factor is not None else "auto"
        await message.answer(
            f"*Your defaults*\nPreset: `{cs.preset}`\nModel: `{cs.model}`\n"
            f"Render factor: `{rf}`\n\n"
            "Change with: `/setpreset full|colorize|restore`, `/setmodel artistic|stable`, "
            "`/setrf 7-45|auto`",
            parse_mode="Markdown",
        )

    @router.message(Command("setpreset"))
    async def set_preset(message: Message) -> None:
        if not await _guard(message):
            return
        value = _arg(message.text)
        if value not in ("full", "colorize", "restore"):
            await message.answer("Usage: /setpreset full|colorize|restore")
            return
        ctx.chat_settings.set(message.chat.id, preset=value)
        await message.answer(f"Default preset set to `{value}`.", parse_mode="Markdown")

    @router.message(Command("setmodel"))
    async def set_model(message: Message) -> None:
        if not await _guard(message):
            return
        value = _arg(message.text)
        if value not in ("artistic", "stable"):
            await message.answer("Usage: /setmodel artistic|stable")
            return
        ctx.chat_settings.set(message.chat.id, model=value)
        await message.answer(f"Default model set to `{value}`.", parse_mode="Markdown")

    @router.message(Command("setrf"))
    async def set_rf(message: Message) -> None:
        if not await _guard(message):
            return
        value = _arg(message.text)
        rf: int | None
        if value == "auto":
            rf = None
        elif value.isdigit() and 7 <= int(value) <= 45:
            rf = int(value)
        else:
            await message.answer("Usage: /setrf 7-45|auto")
            return
        ctx.chat_settings.set(message.chat.id, render_factor=rf)
        await message.answer(f"Default render factor set to `{value}`.", parse_mode="Markdown")

    @router.message(Command("status"))
    async def status_cmd(message: Message) -> None:
        if not await _guard(message):
            return
        jobs = [
            j
            for j in ctx.service.store.list_jobs(limit=50)
            if j.source_ref == str(message.chat.id)
            and j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
        ]
        if not jobs:
            await message.answer("You have no active jobs.")
            return
        lines = []
        for j in jobs:
            pos = ctx.service.store.queue_position(j.id)
            lines.append(f"• `{j.id[:8]}` — {j.status}" + (f" (#{pos})" if pos else ""))
        await message.answer("*Your active jobs*\n" + "\n".join(lines), parse_mode="Markdown")

    @router.message(F.photo | F.video | F.animation | F.document)
    async def on_media(message: Message, bot: Bot) -> None:
        if not await _guard(message):
            return
        file_id, kind, size = _classify_media(message)
        if kind is None or file_id is None:
            await message.answer("Please send an image or a video.")
            return
        if kind == "video":
            if not ctx.settings.video_enabled:
                await message.answer("Video processing is disabled.")
                return
            limit = ctx.settings.telegram_video_max_mb * 1024 * 1024
            if size and size > limit:
                mb = ctx.settings.telegram_video_max_mb
                await message.answer(
                    f"That video is too large for Telegram (max {mb} MB). "
                    "Try a shorter clip, or use the web app."
                )
                return
        token = uuid.uuid4().hex[:12]
        ext = "mp4" if kind == "video" else "img"
        dest = ctx.settings.data_dir / "inputs" / f"tg_{token}.{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        await bot.download(file_id, destination=dest)
        ctx.pending[token] = (str(dest), kind)
        noun = "video" if kind == "video" else "photo"
        await message.answer(
            f"What should I do with this {noun}?", reply_markup=_preset_keyboard(token, kind)
        )

    @router.callback_query(F.data.startswith("go:"))
    async def on_choice(callback: CallbackQuery, bot: Bot) -> None:
        assert callback.message is not None
        chat_id = callback.message.chat.id
        if not allowed(chat_id):
            await callback.answer("Not allowed.", show_alert=True)
            return
        _, preset, token = callback.data.split(":", 2)  # type: ignore[union-attr]
        pending = ctx.pending.pop(token, None)
        if pending is None:
            await callback.answer("That upload expired, please resend it.", show_alert=True)
            return
        input_path, kind = pending
        await callback.answer()
        status_msg = await bot.send_message(chat_id, "Queued…")
        cs = ctx.chat_settings.get(chat_id)
        options = PipelineOptions(
            preset=preset,  # type: ignore[arg-type]
            colorizer_model=cs.model,  # type: ignore[arg-type]
            render_factor=cs.render_factor,
            upscale=2 if (kind == "image" and preset in ("restore", "full")) else None,
            restore_faces=kind == "image" and preset in ("restore", "full"),
        )

        async def on_status(job) -> None:  # type: ignore[no-untyped-def]
            text = {
                JobStatus.QUEUED: "Queued…",
                JobStatus.RUNNING: "Processing… ⏳",
                JobStatus.DONE: "Done ✅",
                JobStatus.FAILED: "Failed ❌",
            }.get(job.status, str(job.status))
            pos = ctx.service.store.queue_position(job.id)
            if job.status is JobStatus.QUEUED and pos:
                text = f"Queued (#{pos})…"
            elif job.status is JobStatus.RUNNING and job.progress > 0:
                text = f"Processing… {int(job.progress * 100)}%"
            with contextlib.suppress(Exception):  # editing is best-effort
                await bot.edit_message_text(text, chat_id=chat_id, message_id=status_msg.message_id)

        try:
            job = await process_and_wait(
                ctx.service, options, input_path, chat_id, on_status=on_status, kind=kind
            )
        except RateLimitError as e:
            await bot.edit_message_text(
                f"Rate limit: {e}", chat_id=chat_id, message_id=status_msg.message_id
            )
            return
        except Exception:  # noqa: BLE001
            logger.exception("telegram job failed")
            await bot.edit_message_text(
                "Something went wrong ❌", chat_id=chat_id, message_id=status_msg.message_id
            )
            return

        if job.status is JobStatus.FAILED or not job.result_path:
            await bot.edit_message_text(
                f"Sorry, processing failed: {job.error or 'unknown error'}",
                chat_id=chat_id,
                message_id=status_msg.message_id,
            )
            return
        if job.kind == "video":
            await bot.send_video(
                chat_id,
                FSInputFile(job.result_path, filename=f"rechroma_{job.id[:8]}.mp4"),
                caption="Colorized ✅",
            )
        else:
            data = Path(job.result_path).read_bytes()
            await bot.send_photo(chat_id, BufferedInputFile(data, "preview.png"), caption="Preview")
            await bot.send_document(
                chat_id, FSInputFile(job.result_path, filename=f"rechroma_{job.id[:8]}.png")
            )

    return router


def _arg(text: str | None) -> str:
    parts = (text or "").split(maxsplit=1)
    return parts[1].strip().lower() if len(parts) > 1 else ""


def build_dispatcher(ctx: BotContext) -> Dispatcher:
    dp = Dispatcher()
    dp.include_router(build_router(ctx))
    return dp


async def run_bot(settings: Settings, service: JobService) -> None:
    """Start the bot (long-polling, or webhook if ``telegram_webhook_url`` is set)."""
    if not settings.telegram_bot_token:
        logger.info("telegram_bot_token unset — Telegram bot disabled")
        return
    bot = Bot(settings.telegram_bot_token)
    ctx = BotContext(settings, service)
    dp = build_dispatcher(ctx)
    logger.info("starting Telegram bot (polling)")
    await dp.start_polling(bot, handle_signals=False)
