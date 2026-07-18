from app.config import Settings
from app.core.engines.diffusion import DiffusionEngine


def _settings(tmp_path, **kw):
    return Settings(data_dir=tmp_path / "d", device="cpu", models_dir=tmp_path / "m", **kw)


def test_disabled_by_default(tmp_path):
    s = _settings(tmp_path)
    assert DiffusionEngine(s).check(s) == (False, "Diffusion engine is disabled")


def test_enabled_but_unavailable_reports_reason(tmp_path):
    # Enabled on a CPU box (no CUDA and/or no diffusers) must stay unavailable
    # with a non-empty, user-actionable reason.
    s = _settings(tmp_path, animate_diffusion_enabled=True)
    available, reason = DiffusionEngine(s).check(s)
    assert available is False
    assert reason  # e.g. "Requires a CUDA GPU" or "Install the 'diffusion' extra"


def test_requires_model_id(tmp_path):
    s = _settings(tmp_path, animate_diffusion_enabled=True, animate_diffusion_model="")
    available, reason = DiffusionEngine(s).check(s)
    assert available is False
    assert "animate_diffusion_model" in reason
