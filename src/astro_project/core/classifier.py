import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_stamps(
    image: torch.Tensor,
    positions: list[tuple],
    stamp_size: int = 32,
) -> torch.Tensor:
    """
    Schneidet für jede (x, y) Koordinate in `positions` einen [stamp_size, stamp_size]
    Ausschnitt aus einem [Y, X] Bildtensor aus. Überschreitungen des Bildrandes werden
    mittels F.pad mit Nullen aufgefüllt. Gibt einen [N, 1, stamp_size, stamp_size] Tensor zurück.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D [Y, X] image tensor, got shape {image.shape}")

    height, width = image.shape
    half = stamp_size // 2

    if len(positions) == 0:
        return torch.empty(
            (0, 1, stamp_size, stamp_size), dtype=image.dtype, device=image.device
        )

    stamps = []
    for x, y in positions:
        xc = int(round(float(x)))
        yc = int(round(float(y)))

        y0 = yc - half
        y1 = y0 + stamp_size
        x0 = xc - half
        x1 = x0 + stamp_size

        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - height)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - width)

        sy0 = max(0, y0)
        sy1 = min(height, y1)
        sx0 = max(0, x0)
        sx1 = min(width, x1)

        patch = image[sy0:sy1, sx0:sx1]
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            # F.pad für 2D-Tensoren erwartet (pad_left, pad_right, pad_top, pad_bottom)
            patch = F.pad(
                patch,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0.0,
            )

        stamps.append(patch.unsqueeze(0))  # [1, stamp_size, stamp_size]

    return torch.stack(stamps, dim=0)  # [N, 1, stamp_size, stamp_size]


class RealBogusCNN(nn.Module):
    """
    CNN zur Real-Bogus-Klassifikation von [N, 1, 32, 32] Kandidaten-Stamps.
    Gibt Wahrscheinlichkeiten im Intervall [0.0, 1.0] mit Shape [N, 1] aus.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> [N, 16, 16, 16]

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> [N, 32, 8, 8]

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.flatten(out)
        out = self.relu3(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out
