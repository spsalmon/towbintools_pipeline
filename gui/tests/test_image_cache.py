# gui/tests/test_image_cache.py
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
from app_components.image_cache import alpha_composite
from app_components.image_cache import apply_lut
from app_components.image_cache import BackgroundLoader
from app_components.image_cache import composite_mask
from app_components.image_cache import downsample
from app_components.image_cache import extract_channel
from app_components.image_cache import PointImageCache
from app_components.image_cache import prepare_channel
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
        for i in range(50):
            result = cache.get(time_index=i, channel_idx=0)
            assert result is not None and result[0, 0] == i


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


class TestDownsample:
    def test_large_image_reduced(self):
        img = np.zeros((2048, 2048), dtype=np.float32)
        result = downsample(img)
        assert max(result.shape) <= 768

    def test_small_image_unchanged(self):
        img = np.zeros((512, 512), dtype=np.float32)
        result = downsample(img)
        assert result.shape == (512, 512)

    def test_3d_input(self):
        img = np.zeros((4, 2048, 2048), dtype=np.float32)
        result = downsample(img)
        assert max(result.shape[-2:]) <= 768
        assert result.shape[0] == 4

    def test_exact_boundary(self):
        img = np.zeros((768, 768), dtype=np.float32)
        result = downsample(img)
        assert result.shape == (768, 768)


class TestApplyLut:
    def test_output_shape(self):
        img = np.random.rand(100, 100).astype(np.float32)
        rgb = apply_lut(img, "viridis")
        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8

    def test_clamps_below_zero(self):
        img = np.full((10, 10), -1.0, dtype=np.float32)
        rgb = apply_lut(img, "viridis")
        assert rgb.dtype == np.uint8

    def test_clamps_above_one(self):
        img = np.full((10, 10), 2.0, dtype=np.float32)
        rgb = apply_lut(img, "viridis")
        assert rgb.dtype == np.uint8

    def test_known_colormaps(self):
        img = np.zeros((4, 4), dtype=np.float32)
        for name in ("viridis", "magma", "autumn"):
            rgb = apply_lut(img, name)
            assert rgb.shape == (4, 4, 3)


class TestAlphaComposite:
    def test_output_shape_and_dtype(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = alpha_composite(base, overlay, alpha=0.5)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_midpoint_blend(self):
        base = np.zeros((4, 4, 3), dtype=np.uint8)
        overlay = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = alpha_composite(base, overlay, alpha=0.5)
        assert np.allclose(result, 100, atol=2)

    def test_alpha_zero_returns_base(self):
        base = np.full((4, 4, 3), 50, dtype=np.uint8)
        overlay = np.full((4, 4, 3), 200, dtype=np.uint8)
        result = alpha_composite(base, overlay, alpha=0.0)
        np.testing.assert_array_equal(result, base)


class TestCompositeMask:
    def test_zero_mask_unchanged(self):
        base = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint16)
        result = composite_mask(base, mask, alpha=0.5)
        np.testing.assert_array_equal(result, base)

    def test_nonzero_mask_changes_pixels(self):
        base = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint16)
        result = composite_mask(base, mask, alpha=1.0)
        assert not np.all(result == 0)

    def test_output_shape(self):
        base = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.uint16)
        result = composite_mask(base, mask)
        assert result.shape == (20, 20, 3)


class TestExtractChannel:
    def test_2d(self):
        img = np.ones((64, 64), dtype=np.uint16)
        result = extract_channel(img, channel=0)
        assert result.shape == (64, 64)

    def test_3d(self):
        img = np.zeros((4, 64, 64), dtype=np.uint16)
        img[2] = 1
        result = extract_channel(img, channel=2)
        assert result.shape == (64, 64)
        assert result[0, 0] == 1

    def test_4d_takes_middle_z(self):
        img = np.zeros((6, 4, 64, 64), dtype=np.uint16)
        z_mid = 3
        img[z_mid, 1] = 99
        result = extract_channel(img, channel=1)
        assert result.shape == (64, 64)
        assert result[0, 0] == 99

    def test_invalid_ndim_raises(self):
        img = np.zeros((2, 3, 4, 5, 6), dtype=np.uint16)
        with pytest.raises(ValueError):
            extract_channel(img, channel=0)


class TestPrepareChannel:
    def test_output_dtype_and_range(self):
        channel_img = np.array([[0, 100], [200, 65535]], dtype=np.uint16)
        result = prepare_channel(channel_img)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_large_image_downsampled(self):
        channel_img = np.random.randint(0, 65535, (2048, 2048), dtype=np.uint16)
        result = prepare_channel(channel_img)
        assert max(result.shape) <= 768


class TestBackgroundLoader:
    def _make_fake_tiff(self, n_channels=2, h=64, w=64):
        return np.random.randint(0, 65535, (n_channels, h, w), dtype=np.uint16)

    def test_populates_cache(self):
        cache = PointImageCache()
        tracker = ProgressTracker()
        n_frames = 3
        n_channels = 2
        fake_img = self._make_fake_tiff(n_channels)
        cache.reset(point=0)
        tracker.reset(total=n_frames)
        paths = [f"/fake/path/{i}.tif" for i in range(n_frames)]

        with patch(
            "app_components.image_cache.image_handling.read_tiff_file",
            return_value=fake_img,
        ):
            loader = BackgroundLoader(
                point=0,
                image_paths=paths,
                n_channels=n_channels,
                cache=cache,
                progress_tracker=tracker,
            )
            loader.wait()

        assert len(cache) == n_frames * n_channels
        for t in range(n_frames):
            for ch in range(n_channels):
                arr = cache.get(t, ch)
                assert arr is not None
                assert arr.dtype == np.float32
                assert max(arr.shape) <= 768

    def test_cancel_stops_workers(self):
        cache = PointImageCache()
        tracker = ProgressTracker()
        n_frames = 100

        def slow_read(path):
            time.sleep(0.05)
            return np.zeros((2, 64, 64), dtype=np.uint16)

        cache.reset(point=0)
        tracker.reset(total=n_frames)
        paths = [f"/fake/{i}.tif" for i in range(n_frames)]

        with patch(
            "app_components.image_cache.image_handling.read_tiff_file",
            side_effect=slow_read,
        ):
            loader = BackgroundLoader(
                point=0,
                image_paths=paths,
                n_channels=2,
                cache=cache,
                progress_tracker=tracker,
            )
            time.sleep(0.1)
            loader.cancel()
            loader.wait()

        assert len(cache) < (n_frames * 2) // 2  # far fewer than all frames

    def test_failed_reads_skipped(self):
        cache = PointImageCache()
        tracker = ProgressTracker()
        n_frames = 5
        fake_img = self._make_fake_tiff(1)
        cache.reset(point=0)
        tracker.reset(total=n_frames)
        paths = [f"/fake/{i}.tif" for i in range(n_frames)]

        call_count = {"n": 0}

        def sometimes_fail(path):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise OSError("simulated read failure")
            return fake_img

        with patch(
            "app_components.image_cache.image_handling.read_tiff_file",
            side_effect=sometimes_fail,
        ):
            loader = BackgroundLoader(
                point=0,
                image_paths=paths,
                n_channels=1,
                cache=cache,
                progress_tracker=tracker,
            )
            loader.wait()

        # Frames 0, 2, 4 succeed (3 out of 5); frames 1, 3 fail silently
        assert len(cache) == 3

    def test_stale_writes_discarded_after_cancel_and_reset(self):
        cache = PointImageCache()
        tracker = ProgressTracker()
        fake_img = self._make_fake_tiff(1)

        cache.reset(point=0)
        tracker.reset(total=5)
        paths = [f"/fake/{i}.tif" for i in range(5)]

        with patch(
            "app_components.image_cache.image_handling.read_tiff_file",
            return_value=fake_img,
        ):
            loader = BackgroundLoader(
                point=0,
                image_paths=paths,
                n_channels=1,
                cache=cache,
                progress_tracker=tracker,
            )
            loader.cancel()
            cache.reset(point=1)
            loader.wait()

        for t in range(5):
            assert cache.get(t, 0) is None

    def test_progress_tracker_incremented(self):
        cache = PointImageCache()
        tracker = ProgressTracker()
        n_frames = 4
        fake_img = self._make_fake_tiff(1)

        cache.reset(point=0)
        tracker.reset(total=n_frames)
        paths = [f"/fake/{i}.tif" for i in range(n_frames)]

        with patch(
            "app_components.image_cache.image_handling.read_tiff_file",
            return_value=fake_img,
        ):
            loader = BackgroundLoader(
                point=0,
                image_paths=paths,
                n_channels=1,
                cache=cache,
                progress_tracker=tracker,
            )
            loader.wait()

        completed, total = tracker.get()
        assert completed == n_frames
        assert total == n_frames
