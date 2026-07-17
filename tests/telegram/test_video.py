from app.config import Settings
from app.jobs.service import JobService
from app.jobs.store import JobStore
from app.telegram.bot import BotContext, _preset_keyboard, build_dispatcher


def _ctx(tmp_path):
    settings = Settings(data_dir=tmp_path / "d", telegram_video_max_mb=5)
    service = JobService(JobStore(tmp_path / "j.db"), lambda job: "x")
    return BotContext(settings, service)


def test_dispatcher_still_builds_with_media_handler(tmp_path):
    dp = build_dispatcher(_ctx(tmp_path))
    router = dp.sub_routers[0]
    assert len(router.message.handlers) >= 7
    assert len(router.callback_query.handlers) >= 1


def test_video_keyboard_is_colorize_only():
    kb = _preset_keyboard("tok", kind="video")
    buttons = [b.text for row in kb.inline_keyboard for b in row]
    assert buttons == ["🎨 Colorize"]


def test_image_keyboard_has_three_presets():
    kb = _preset_keyboard("tok", kind="image")
    buttons = [b.text for row in kb.inline_keyboard for b in row]
    assert len(buttons) == 3


def test_context_carries_video_cap(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.settings.telegram_video_max_mb == 5
