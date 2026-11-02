import torch


def resolve_device(device: torch.device | str | None = "auto") -> torch.device:
    """Resolves a device specification into an explicit torch.device with an index for CUDA."""
    if isinstance(device, torch.device):
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            idx = 0 if device.index is None else device.index
            return torch.device("cuda", idx)
        return device

    dev_str = "auto" if device is None else str(device).strip().lower()

    if dev_str in ("auto", "cuda", "gpu", "multi-gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        return torch.device("cpu")

    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        parsed = torch.device(dev_str)
        idx = 0 if parsed.index is None else parsed.index
        return torch.device("cuda", idx)

    return torch.device("cpu")


def get_available_devices() -> list[str]:
    """Returns a list of available compute devices ('cuda:0', optional 'multi-gpu', 'cpu')."""
    devices: list[str] = []
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for idx in range(count):
            devices.append(f"cuda:{idx}")
        if count > 1:
            devices.append("multi-gpu")
    devices.append("cpu")
    return devices
