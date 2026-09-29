import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    """Decode each vector in a sequence into a square RGB frame."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        image_size: int = 128,
        base_channels: int = 256,
    ) -> None:
        super().__init__()
        if image_size < 16 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 16")
        if base_channels < 8:
            raise ValueError("base_channels must be at least 8")

        self.image_size = int(image_size)
        self.output_channels = int(out_dim)
        self.initial_size = self.image_size // 16
        self.base_channels = int(base_channels)
        self.projection = nn.Linear(
            in_dim,
            self.base_channels * self.initial_size * self.initial_size,
        )

        channels = (
            self.base_channels,
            self.base_channels // 2,
            self.base_channels // 4,
            self.base_channels // 8,
            self.output_channels,
        )
        layers: list[nn.Module] = []
        for index, (input_channels, output_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            layers.append(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if index < len(channels) - 2:
                layers.extend((nn.BatchNorm2d(output_channels), nn.ReLU()))
        self.convolutions = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        projected = self.projection(x.reshape(-1, x.shape[-1]))
        projected = projected.reshape(
            -1,
            self.base_channels,
            self.initial_size,
            self.initial_size,
        )
        images = self.convolutions(projected)
        return images.reshape(
            *leading_shape,
            self.output_channels,
            self.image_size,
            self.image_size,
        )
