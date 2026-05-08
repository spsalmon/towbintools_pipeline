# gui/app_components/image_cache.py
import math
import threading

import matplotlib
import numpy as np
from towbintools.foundation import image_handling


class PointImageCache:
    """Thread-safe store of downsampled float32 channel arrays, keyed by (time_index, channel_idx)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[int, int], np.ndarray] = {}
        self._point: int | None = None

    def reset(self, point: int) -> None:
        with self._lock:
            self._data.clear()
            self._point = point

    def put(
        self, point: int, time_index: int, channel_idx: int, array: np.ndarray
    ) -> None:
        with self._lock:
            if self._point is None or point != self._point:
                return
            self._data[(time_index, channel_idx)] = array

    def get(self, time_index: int, channel_idx: int) -> np.ndarray | None:
        with self._lock:
            return self._data.get((time_index, channel_idx))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class ProgressTracker:
    """Thread-safe counter for background loader progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def reset(self, total: int) -> None:
        with self._lock:
            self._completed = 0
            self._total = total

    def increment(self) -> None:
        with self._lock:
            self._completed += 1

    def get(self) -> tuple[int, int]:
        with self._lock:
            return self._completed, self._total


MAX_DISPLAY_PX = 768

COLORMAPS: dict[str, np.ndarray] = {
    name: (
        matplotlib.colormaps.get_cmap(name)(np.linspace(0, 1, 256))[:, :3] * 255
    ).astype(np.uint8)
    for name in ("viridis", "magma", "autumn")
}


def downsample(img: np.ndarray) -> np.ndarray:
    """Stride-based downsample so the spatial dimensions fit within MAX_DISPLAY_PX."""
    step = max(1, math.ceil(max(img.shape[-2:]) / MAX_DISPLAY_PX))
    return img[..., ::step, ::step]


def apply_lut(img: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Convert a float32 [0, 1] 2-D array to an RGB uint8 array using a pre-built LUT."""
    lut = COLORMAPS[cmap_name]
    u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return lut[u8]


def alpha_composite(
    base: np.ndarray, overlay: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend two RGB uint8 arrays: result = base*(1-alpha) + overlay*alpha."""
    return np.clip(
        base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def composite_mask(
    base: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap_name: str = "autumn",
) -> np.ndarray:
    """Overlay a segmentation mask (non-zero pixels only) onto an RGB base image."""
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 0:
        mask_f = mask_f / mask_f.max()
    overlay = apply_lut(mask_f, cmap_name)
    visible = mask > 0
    result = base.copy()
    result[visible] = np.clip(
        base[visible].astype(np.float32) * (1.0 - alpha)
        + overlay[visible].astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    return result


def extract_channel(img: np.ndarray, channel: int) -> np.ndarray:
    """Extract a single spatial channel from a 2-D, 3-D, or 4-D array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[channel]
    if img.ndim == 4:
        return img[img.shape[0] // 2, channel]
    raise ValueError(f"Unexpected image dimensions: {img.ndim}")


def prepare_channel(channel_img: np.ndarray) -> np.ndarray:
    """Normalize a single channel to float32 [0, 1] and downsample for display."""
    normalized = image_handling.normalize_image(channel_img, dest_dtype=np.float32)
    return downsample(normalized)
