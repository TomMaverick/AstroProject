import pytest
import torch

from astro_project.core.hardware import get_available_devices, resolve_device


@pytest.fixture(autouse=True)
def vram_cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_resolve_device_cpu_and_auto():
    cpu_dev = resolve_device("cpu")
    assert isinstance(cpu_dev, torch.device)
    assert cpu_dev.type == "cpu"
    t_cpu = torch.zeros((4, 4), device=cpu_dev)
    assert t_cpu.device == cpu_dev

    auto_dev = resolve_device("auto")
    assert isinstance(auto_dev, torch.device)
    if torch.cuda.is_available():
        assert auto_dev == torch.device("cuda", 0)
    else:
        assert auto_dev == torch.device("cpu")
    t_auto = torch.zeros((4, 4), device=auto_dev)
    assert t_auto.device == auto_dev


def test_resolve_device_cuda_indexing():
    dev = resolve_device("cuda")
    if torch.cuda.is_available():
        assert dev == torch.device("cuda", 0)
        assert dev.index == 0
        tensor = torch.ones((2, 2), device=dev)
        assert tensor.device == dev
    else:
        assert dev == torch.device("cpu")


def test_get_available_devices():
    devices = get_available_devices()
    assert isinstance(devices, list)
    assert "cpu" in devices
    if torch.cuda.is_available():
        assert "cuda:0" in devices
        if torch.cuda.device_count() > 1:
            assert "multi-gpu" in devices
