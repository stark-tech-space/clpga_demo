import numpy as np
import pytest

from clpga_demo.cropper import CropRegion, calculate_crop


class TestCalculateCrop:
    def test_landscape_1080p_centered(self):
        """1920x1080 source with ball centered should produce 607x1080 crop."""
        crop = calculate_crop(
            ball_x=960.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        expected_w = int(1080 * 9 / 16)  # 607
        assert crop.width == expected_w
        assert crop.height == 1080
        # Centered: x should be (960 - 607/2) ~ 656
        assert crop.x == pytest.approx(960 - expected_w / 2, abs=1)
        assert crop.y == 0

    def test_edge_clamp_left(self):
        """Ball near left edge should clamp crop x to 0."""
        crop = calculate_crop(
            ball_x=50.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        assert crop.x == 0
        assert crop.width == int(1080 * 9 / 16)

    def test_edge_clamp_right(self):
        """Ball near right edge should clamp crop to not exceed frame."""
        crop = calculate_crop(
            ball_x=1900.0, ball_y=540.0,
            source_width=1920, source_height=1080,
        )
        assert crop.x + crop.width <= 1920

    def test_narrow_source_crops_vertically(self):
        """Source narrower than 9:16 should crop vertically instead."""
        # 360x640 source (9:16 already) — should use full width
        crop = calculate_crop(
            ball_x=180.0, ball_y=320.0,
            source_width=360, source_height=640,
        )
        assert crop.width == 360
        expected_h = int(360 * 16 / 9)  # 640
        assert crop.height == expected_h

    def test_very_narrow_source_crops_vertically(self):
        """Source much narrower than 9:16 should crop height to fit ratio."""
        # 200x800 source — too narrow, must crop vertically
        crop = calculate_crop(
            ball_x=100.0, ball_y=400.0,
            source_width=200, source_height=800,
        )
        assert crop.width == 200
        expected_h = int(200 * 16 / 9)  # 355
        assert crop.height == expected_h
        # Centered vertically on ball
        assert crop.y + crop.height <= 800

    def test_vertical_edge_clamp(self):
        """Ball near bottom of narrow source should clamp y."""
        crop = calculate_crop(
            ball_x=100.0, ball_y=790.0,
            source_width=200, source_height=800,
        )
        assert crop.y + crop.height <= 800
        assert crop.y >= 0


class TestCropRegion:
    def test_apply_extracts_region(self):
        """CropRegion.apply should extract the correct sub-array from a frame."""
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        crop = CropRegion(x=10, y=20, width=50, height=30)
        result = crop.apply(frame)
        assert result.shape == (30, 50, 3)
        np.testing.assert_array_equal(result, frame[20:50, 10:60, :])
