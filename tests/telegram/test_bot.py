from app.config import Settings
from app.jobs.service import JobService
from app.jobs.store import JobStore
from app.telegram.bot import BotContext, build_dispatcher, build_router, run_bot


def _ctx(tmp_path) -> BotContext:
    settings = Settings(data_dir=tmp_path / "data", telegram_bot_token=None)
    service = JobService(JobStore(tmp_path / "j.db"), lambda job: "x")
    return BotContext(settings, service)


def test_dispatcher_builds_with_handlers(tmp_path):
    dp = build_dispatcher(_ctx(tmp_path))
    # message + callback handlers are registered on the included router
    router = dp.sub_routers[0]
    assert len(router.message.handlers) >= 6  # start, help, settings, set*, status, image
    assert len(router.callback_query.handlers) >= 1


def test_router_is_reusable(tmp_path):
    router = build_router(_ctx(tmp_path))
    assert router.message.handlers


async def test_run_bot_noop_without_token(tmp_path):
    # No token -> returns immediately without attempting a network connection.
    ctx = _ctx(tmp_path)
    await run_bot(ctx.settings, ctx.service)
