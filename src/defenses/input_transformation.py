import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)


class InputTransformation:
    """Implements various input transformation defenses."""

    def __init__(
            self,
            methods: list = ['jpeg', 'bit_depth', 'smoothing'],
            jpeg_quality: int = 75,
            bit_depth: int = 5,
            kernel_size: int = 3
    ):
        self.methods = methods
        self.jpeg_quality = jpeg_quality
        self.bit_depth = bit_depth
        self.kernel_size = kernel_size

        if 'smoothing' in methods:
            self.smooth_kernel = self._create_gaussian_kernel(kernel_size)

    def transform(
            self,
            x: torch.Tensor,
            method: Optional[str] = None
    ) -> torch.Tensor:
        """Apply input transformation."""
        if method is None:
            # Apply all specified methods
            result = x
            for m in self.methods:
                result = self._apply_method(result, m)
            return result

        return self._apply_method(x, method)

    def _apply_method(
            self,
            x: torch.Tensor,
            method: str
    ) -> torch.Tensor:
        """Apply specific transformation method."""
        if method == 'jpeg':
            return self._jpeg_compression(x)
        elif method == 'bit_depth':
            return self._reduce_bit_depth(x)
        elif method == 'smoothing':
            return self._spatial_smoothing(x)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

    def _jpeg_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate JPEG compression."""
        # Simplified JPEG-like compression using DCT
        batch_size, channels, height, width = x.shape
        block_size = 8

        # Pad if necessary
        pad_h = block_size - height % block_size if height % block_size != 0 else 0
        pad_w = block_size - width % block_size if width % block_size != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # Apply DCT and quantization
        result = x.clone()
        for i in range(0, height + pad_h, block_size):
            for j in range(0, width + pad_w, block_size):
                block = x[:, :, i:i + block_size, j:j + block_size]
                dct_block = torch.dct(block, dim=2)
                dct_block = torch.dct(dct_block, dim=3)

                # Quantize
                dct_block = torch.round(dct_block * self.jpeg_quality) / self.jpeg_quality

                # Inverse DCT
                idct_block = torch.idct(dct_block, dim=3)
                idct_block = torch.idct(idct_block, dim=2)
                result[:, :, i:i + block_size, j:j + block_size] = idct_block

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :height, :width]

        return torch.clamp(result, 0, 1)

    def _reduce_bit_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce the bit depth of the input."""
        max_val = 2 ** self.bit_depth - 1
        x_scaled = x * max_val
        x_quantized = torch.round(x_scaled) / max_val
        return x_quantized

    def _spatial_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing."""
        kernel = self.smooth_kernel.to(x.device)
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(
            x,
            kernel,
            padding=self.kernel_size // 2,
            groups=x.shape[1]
        )

    def _create_gaussian_kernel(self, size: int) -> torch.Tensor:
        """Create a Gaussian kernel."""
        sigma = size / 6
        coords = torch.arange(size).float() - (size - 1) / 2
        coords = coords.reshape(-1, 1)

        gaussian = torch.exp(-(coords ** 2 + coords.T ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()

        return gaussian.reshape(1, 1, size, size)