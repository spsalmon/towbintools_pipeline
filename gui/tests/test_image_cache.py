# gui/tests/test_image_cache.py
import threading

import numpy as np
from app_components.image_cache import PointImageCache
from app_components.image_cache import ProgressTracker


class TestPointImageCache:
    def test_put_and_get(self):
        cache = PointImageCache()
        cache.reset(point=0)
        arr = np.zeros((10, 10), dtype=np.float32)
        cache.put(point=0, time_index=5, channel_idx=1, array=arr)
        result = cache.get(time_index=5, channel_idx=1)
        assert result is arr

    def test_miss_returns_none(self):
        cache = PointImageCache()
        cache.reset(point=0)
        assert cache.get(time_index=99, channel_idx=0) is None

    def test_stale_write_discarded(self):
        cache = PointImageCache()
        cache.reset(point=0)
        arr = np.zeros((10, 10), dtype=np.float32)
        cache.put(point=1, time_index=5, channel_idx=0, array=arr)  # wrong point
        assert cache.get(time_index=5, channel_idx=0) is None

    def test_reset_clears_data(self):
        cache = PointImageCache()
        cache.reset(point=0)
        arr = np.zeros((10, 10), dtype=np.float32)
        cache.put(point=0, time_index=0, channel_idx=0, array=arr)
        cache.reset(point=1)
        assert cache.get(time_index=0, channel_idx=0) is None

    def test_len(self):
        cache = PointImageCache()
        cache.reset(point=0)
        for i in range(5):
            cache.put(0, i, 0, np.zeros((4, 4), dtype=np.float32))
        assert len(cache) == 5

    def test_thread_safety(self):
        cache = PointImageCache()
        cache.reset(point=0)
        errors = []

        def writer(t):
            try:
                arr = np.full((4, 4), t, dtype=np.float32)
                cache.put(0, t, 0, arr)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(cache) == 50


class TestProgressTracker:
    def test_initial_state(self):
        tracker = ProgressTracker()
        completed, total = tracker.get()
        assert completed == 0
        assert total == 0

    def test_reset(self):
        tracker = ProgressTracker()
        tracker.reset(total=100)
        completed, total = tracker.get()
        assert completed == 0
        assert total == 100

    def test_increment(self):
        tracker = ProgressTracker()
        tracker.reset(total=10)
        tracker.increment()
        tracker.increment()
        completed, total = tracker.get()
        assert completed == 2
        assert total == 10

    def test_reset_clears_completed(self):
        tracker = ProgressTracker()
        tracker.reset(total=10)
        tracker.increment()
        tracker.reset(total=20)
        completed, total = tracker.get()
        assert completed == 0
        assert total == 20

    def test_thread_safety(self):
        tracker = ProgressTracker()
        tracker.reset(total=100)

        def inc():
            for _ in range(10):
                tracker.increment()

        threads = [threading.Thread(target=inc) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        completed, _ = tracker.get()
        assert completed == 100
