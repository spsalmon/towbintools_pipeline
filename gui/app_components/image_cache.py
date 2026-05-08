# gui/app_components/image_cache.py
import threading

import numpy as np


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
