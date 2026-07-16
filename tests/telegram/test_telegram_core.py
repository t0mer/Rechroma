from app.core.pipeline import PipelineOptions
from app.jobs.models import JobStatus
from app.jobs.service import JobService
from app.jobs.store import JobStore
from app.telegram.access import is_allowed
from app.telegram.chat_settings import ChatSettingsStore
from app.telegram.runner import process_and_wait


def test_admin_always_allowed():
    assert is_allowed(5, allowed=[], admins=[5]) is True


def test_empty_allowlist_refuses_non_admin():
    assert is_allowed(9, allowed=[], admins=[5]) is False


def test_allowlist_membership():
    assert is_allowed(7, allowed=[7, 8], admins=[]) is True
    assert is_allowed(3, allowed=[7, 8], admins=[]) is False


def test_chat_settings_defaults_and_update(tmp_path):
    store = ChatSettingsStore(tmp_path / "cs.db")
    assert store.get(42).preset == "full"
    store.set(42, preset="colorize", render_factor=30)
    got = store.get(42)
    assert got.preset == "colorize"
    assert got.render_factor == 30
    assert got.model == "artistic"  # unchanged
    # persists across reopen
    assert ChatSettingsStore(tmp_path / "cs.db").get(42).preset == "colorize"


async def test_process_and_wait_reports_status_and_completes(tmp_path):
    def proc(job):
        return job.input_path + ".out"

    svc = JobService(JobStore(tmp_path / "j.db"), proc)
    await svc.start()
    seen: list[str] = []

    async def on_status(job):
        seen.append(str(job.status))

    final = await process_and_wait(
        svc, PipelineOptions(), "/tmp/x.png", chat_id=1, poll_interval=0.01, on_status=on_status
    )
    assert final.status is JobStatus.DONE
    assert final.result_path == "/tmp/x.png.out"
    assert str(JobStatus.DONE) in seen
    await svc.stop()
