import pytest

from app.config import Settings
from app.core.engines import (
    ENGINE_NAMES,
    EngineUnavailable,
    build_engine,
    list_engine_infos,
    resolve_engine_name,
)


def _settings(tmp_path, **kw):
    return Settings(data_dir=tmp_path / "d", device="cpu", models_dir=tmp_path / "m", **kw)


def test_lists_all_engines_in_order(tmp_path):
    infos = list_engine_infos(_settings(tmp_path))
    assert [i.name for i in infos] == list(ENGINE_NAMES)


def test_tpsmm_available_diffusion_and_cloud_off_by_default(tmp_path):
    infos = {i.name: i for i in list_engine_infos(_settings(tmp_path))}
    assert infos["tpsmm"].available is True
    assert infos["diffusion"].available is False
    assert infos["cloud"].available is False
    assert infos["cloud"].requires_key is True
    assert infos["diffusion"].requires_gpu is True


def test_resolve_falls_back_to_default_when_requested_unavailable(tmp_path):
    s = _settings(tmp_path)
    assert resolve_engine_name("cloud", s) == "tpsmm"  # cloud off -> default
    assert resolve_engine_name(None, s) == "tpsmm"


def test_build_unavailable_engine_raises(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(EngineUnavailable):
        build_engine("cloud", s)
    with pytest.raises(EngineUnavailable):
        build_engine("bogus", s)


def test_build_tpsmm_succeeds(tmp_path):
    engine = build_engine("tpsmm", _settings(tmp_path))
    assert engine.name == "tpsmm"


def test_animate_disabled_makes_tpsmm_unavailable(tmp_path):
    s = _settings(tmp_path, animate_enabled=False)
    infos = {i.name: i for i in list_engine_infos(s)}
    assert infos["tpsmm"].available is False
