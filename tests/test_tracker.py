import numpy as np
import pytest

from clpga_demo.tracker import TrackResult, select_ball


class TestSelectBall:
    def test_single_detection_selected(self):
        """With one detection, it should be selected."""
        boxes = np.array([[100, 100, 200, 200, 1, 0.9, 0]])  # x1,y1,x2,y2,obj_id,score,cls
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result.obj_id == 1

    def test_centered_and_large_preferred(self):
        """A centered, large ball should score higher than a small corner one."""
        boxes = np.array([
            [0, 0, 20, 20, 1, 0.9, 0],          # small, top-left corner
            [900, 500, 1020, 580, 2, 0.9, 0],    # large, centered
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result.obj_id == 2

    def test_no_detections_returns_none(self):
        """Empty boxes should return None."""
        boxes = np.array([]).reshape(0, 7)
        result = select_ball(boxes, frame_width=1920, frame_height=1080)
        assert result is None

    def test_track_result_center(self):
        """TrackResult should compute center from bbox."""
        result = TrackResult(
            frame_idx=0,
            center_x=150.0,
            center_y=150.0,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            obj_id=1,
        )
        assert result.center_x == 150.0
        assert result.center_y == 150.0


class TestStickySelection:
    def test_sticky_follows_same_obj_id(self):
        """select_ball with preferred_obj_id should return that object if present."""
        boxes = np.array([
            [900, 500, 1020, 580, 2, 0.9, 0],  # centered, large
            [100, 100, 120, 120, 5, 0.9, 0],    # small, corner
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080, preferred_obj_id=5)
        assert result.obj_id == 5

    def test_sticky_falls_back_when_obj_lost(self):
        """If preferred_obj_id is not in boxes, re-evaluate heuristic."""
        boxes = np.array([
            [900, 500, 1020, 580, 2, 0.9, 0],
            [100, 100, 120, 120, 3, 0.9, 0],
        ])
        result = select_ball(boxes, frame_width=1920, frame_height=1080, preferred_obj_id=99)
        # preferred_obj_id 99 not found — should fall back to heuristic (obj_id=2)
        assert result.obj_id == 2
