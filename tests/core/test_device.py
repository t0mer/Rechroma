import pytest
import torch

from app.core.device import device_defaults, resolve_device


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cuda_without_gpu_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_explicit_cpu_always_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_unknown_pref_errors():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_defaults_differ_by_device():
    assert device_defaults(torch.device("cuda")).render_factor == 35
    assert device_defaults(torch.device("cpu")).render_factor == 25
    assert device_defaults(torch.device("cuda")).max_resolution == 6000
    assert device_defaults(torch.device("cpu")).max_resolution == 4000
