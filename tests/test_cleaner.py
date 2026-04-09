import numpy as np
import pytest

from clpga_demo.cleaner import FrameCleaner, Corridor


class TestCorridorComputation:
    def test_corridor_size_scales_with_ball_size(self):
        """Corridor radius should be at least ball_size * corridor_multiplier."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0), (100.0, 100.0)]
        ball_sizes = [20.0, 20.0, 20.0, 20.0, 20.0]
        speeds = [0.0, 0.0, 0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert len(corridors) == 5
        for c in corridors:
            assert c.radius == 80.0
            assert c.center_x == 100.0
            assert c.center_y == 100.0

    def test_corridor_size_scales_with_speed(self):
        """Corridor radius should grow with ball speed."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0)]
        ball_sizes = [20.0]
        speeds = [50.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        # speed * corridor_speed_scale * radius_scale = 50 * 1.5 * 4.0 = 300
        assert corridors[0].radius == 300.0

    def test_corridor_clips_to_frame_bounds(self):
        """Corridor bbox should be clipped to frame edges."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(10.0, 10.0)]
        ball_sizes = [20.0]
        speeds = [0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0].x1 >= 0
        assert corridors[0].y1 >= 0

    def test_corridor_none_trajectory_point_skipped(self):
        """Frames where trajectory is None should produce a None corridor."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
            },
        )

        rough_trajectory = [(100.0, 100.0), None, (120.0, 100.0)]
        ball_sizes = [20.0, 0.0, 20.0]
        speeds = [0.0, 0.0, 0.0]

        corridors = cleaner.compute_corridors(rough_trajectory, ball_sizes, speeds, frame_w=640, frame_h=480)

        assert corridors[0] is not None
        assert corridors[1] is None
        assert corridors[2] is not None


class TestBallMaskIdentification:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_selects_closest_mask_to_trajectory(self):
        """Should pick the mask whose centroid is closest to trajectory point."""
        cleaner = self._make_cleaner()
        mask_near = np.zeros((480, 640), dtype=bool)
        mask_near[90:110, 90:110] = True
        mask_far = np.zeros((480, 640), dtype=bool)
        mask_far[290:310, 290:310] = True
        masks = [mask_near, mask_far]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 0

    def test_rejects_mask_with_bad_aspect_ratio(self):
        """Closest mask should be rejected if aspect ratio is too large."""
        cleaner = self._make_cleaner()
        mask_elongated = np.zeros((480, 640), dtype=bool)
        mask_elongated[99:100, 80:120] = True  # 1px tall, 40px wide
        mask_round = np.zeros((480, 640), dtype=bool)
        mask_round[190:210, 190:210] = True
        masks = [mask_elongated, mask_round]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 1

    def test_rejects_mask_with_bad_size_ratio(self):
        """Closest mask should be rejected if too large compared to median ball size."""
        cleaner = self._make_cleaner()
        mask_big = np.zeros((480, 640), dtype=bool)
        mask_big[50:150, 50:150] = True  # 100x100 vs median 20
        mask_right = np.zeros((480, 640), dtype=bool)
        mask_right[195:205, 195:205] = True
        masks = [mask_big, mask_right]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result == 1

    def test_returns_none_when_no_valid_masks(self):
        """Should return None if all masks fail validation."""
        cleaner = self._make_cleaner()
        mask_huge = np.zeros((480, 640), dtype=bool)
        mask_huge[0:200, 0:200] = True
        masks = [mask_huge]
        result = cleaner.identify_ball_mask(masks, (100.0, 100.0), 20.0)
        assert result is None

    def test_empty_masks_returns_none(self):
        """Should return None for empty mask list."""
        cleaner = self._make_cleaner()
        result = cleaner.identify_ball_mask([], (100.0, 100.0), 20.0)
        assert result is None


class TestQuadmaskGeneration:
    def _make_cleaner(self):
        return FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

    def test_ball_mask_region_is_255(self):
        """Ball mask pixels should be 255 (keep) in the quadmask."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[200:220, 200:220] = True
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[distractor_mask],
            corridor=corridor, frame_h=480, frame_w=640,
        )
        assert quadmask.shape == (480, 640)
        assert quadmask.dtype == np.uint8
        assert np.all(quadmask[90:110, 90:110] == 255)

    def test_distractor_region_is_0(self):
        """Distractor mask pixels (after dilation) should be 0 (remove)."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:160, 140:160] = True  # Inside corridor
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[distractor_mask],
            corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask[142:158, 142:158] == 0)

    def test_outside_corridor_is_255(self):
        """Everything outside the corridor should be 255 (keep)."""
        cleaner = self._make_cleaner()
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=None, distractor_masks=[], corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask[0:20, :] == 255)
        assert np.all(quadmask[180:, :] == 255)
        assert np.all(quadmask[:, 0:20] == 255)
        assert np.all(quadmask[:, 180:] == 255)

    def test_no_distractors_returns_all_255(self):
        """If no distractor masks, entire quadmask should be 255."""
        cleaner = self._make_cleaner()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[90:110, 90:110] = True
        corridor = Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180)
        quadmask = cleaner.generate_quadmask_frame(
            ball_mask=ball_mask, distractor_masks=[], corridor=corridor, frame_h=480, frame_w=640,
        )
        assert np.all(quadmask == 255)


class TestSegmentSplitting:
    def test_short_video_single_segment(self):
        segments = FrameCleaner.split_into_segments(total_frames=100, max_frames=180, overlap=16)
        assert segments == [(0, 100)]

    def test_exact_max_frames_single_segment(self):
        segments = FrameCleaner.split_into_segments(total_frames=180, max_frames=180, overlap=16)
        assert segments == [(0, 180)]

    def test_long_video_multiple_segments_with_overlap(self):
        segments = FrameCleaner.split_into_segments(total_frames=300, max_frames=180, overlap=16)
        assert segments[0] == (0, 180)
        assert segments[1][0] == 164  # 180 - 16
        assert segments[-1][1] == 300

    def test_segments_cover_all_frames(self):
        segments = FrameCleaner.split_into_segments(total_frames=500, max_frames=180, overlap=16)
        covered = set()
        for start, end in segments:
            covered.update(range(start, end))
        assert covered == set(range(500))


class TestOverlapBlending:
    def test_blend_two_segments_linear_crossfade(self):
        seg1 = np.full((20, 4, 4, 3), 100, dtype=np.uint8)
        seg2 = np.full((20, 4, 4, 3), 200, dtype=np.uint8)
        segments = [(0, 20), (12, 32)]
        result = FrameCleaner.blend_segments([seg1, seg2], segments, total_frames=32)
        assert result.shape == (32, 4, 4, 3)
        assert np.all(result[0] == 100)
        assert np.all(result[11] == 100)
        assert np.all(result[20] == 200)
        assert np.all(result[31] == 200)
        mid_val = result[16, 0, 0, 0]
        assert 140 <= mid_val <= 160

    def test_single_segment_no_blending(self):
        seg = np.full((10, 4, 4, 3), 128, dtype=np.uint8)
        result = FrameCleaner.blend_segments([seg], [(0, 10)], total_frames=10)
        assert np.array_equal(result, seg)


from unittest.mock import patch, MagicMock, call
from clpga_demo.scene_analyzer import SceneAnalysis


class TestCleanSegments:
    def test_clean_segments_calls_void_model(self):
        """clean_segments should invoke void_model.inpaint for each segment."""
        mock_void = MagicMock()
        mock_void.inpaint.return_value = np.zeros((10, 384, 672, 3), dtype=np.uint8)

        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=mock_void,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmasks = np.full((10, 384, 672), 255, dtype=np.uint8)
        # Mark some pixels for removal so it's not all-255
        quadmasks[:, 100:150, 200:250] = 0
        segments = [(0, 10)]

        result = cleaner.clean_segments(video_frames, quadmasks, segments, "golf course")

        assert len(result) == 1
        mock_void.inpaint.assert_called_once()

    def test_clean_segments_skips_all_255_quadmask(self):
        """If a segment's quadmask is all 255 (nothing to remove), skip void-model."""
        mock_void = MagicMock()

        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=mock_void,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 5,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((10, 384, 672, 3), dtype=np.uint8)
        quadmasks = np.full((10, 384, 672), 255, dtype=np.uint8)  # Nothing to remove
        segments = [(0, 10)]

        result = cleaner.clean_segments(video_frames, quadmasks, segments, "golf course")

        mock_void.inpaint.assert_not_called()
        assert np.array_equal(result[0], video_frames)


class TestGenerateQuadmasks:
    def test_generates_quadmask_array(self):
        """generate_quadmasks should return (T, H, W) uint8 array."""
        mock_sam = MagicMock()
        mock_result = MagicMock()
        ball_mask = np.zeros((480, 640), dtype=bool)
        ball_mask[95:105, 95:105] = True
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:160, 140:160] = True
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.stack([ball_mask, distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        corridors = [
            Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180),
            Corridor(center_x=100, center_y=100, radius=80, x1=20, y1=20, x2=180, y2=180),
            None,
        ]
        trajectory = [(100.0, 100.0), (100.0, 100.0), None]
        median_ball_size = 10.0

        quadmasks = cleaner.generate_quadmasks(video_frames, corridors, trajectory, median_ball_size)

        assert quadmasks.shape == (3, 480, 640)
        assert quadmasks.dtype == np.uint8
        assert np.all(quadmasks[2] == 255)

    def test_no_corridor_produces_all_255(self):
        """Frames with no corridor should have all-255 quadmask."""
        cleaner = FrameCleaner(
            sam3_model=None,
            void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0,
                "corridor_speed_scale": 1.5,
                "radius_scale": 4.0,
                "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0,
                "max_size_ratio": 2.0,
            },
        )

        video_frames = np.zeros((2, 100, 100, 3), dtype=np.uint8)
        corridors = [None, None]
        trajectory = [None, None]

        quadmasks = cleaner.generate_quadmasks(video_frames, corridors, trajectory, 10.0)

        assert np.all(quadmasks == 255)


class TestTargetedQuadmaskGeneration:
    def test_no_distractors_returns_all_255(self):
        """If Gemini found no distractors, all frames should be 255."""
        cleaner = FrameCleaner(
            sam3_model=None, void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )
        video_frames = np.zeros((5, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)] * 5
        analysis = SceneAnalysis(ball_bbox=(95, 95, 105, 105), distractors=[], scene_description="clean green")
        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)
        assert quadmasks.shape == (5, 480, 640)
        assert np.all(quadmasks == 255)

    def test_distractor_region_marked_for_removal(self):
        """Distractor mask from SAM3 should be marked as 0 in quadmask."""
        mock_sam = MagicMock()
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[140:200, 500:530] = True
        mock_result = MagicMock()
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.array([distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam, void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 0,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )
        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(300, 300, 250, 50, 50, 550, 550)]
        analysis = SceneAnalysis(
            ball_bbox=(290, 290, 310, 310),
            distractors=[{"label": "flagpole", "bbox": (490, 130, 540, 210)}],
            scene_description="green with flagpole",
        )
        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)
        assert np.any(quadmasks[0, 140:200, 500:530] == 0)

    def test_ball_protection_zone_stays_255(self):
        """Ball bbox region should remain 255 even if distractor overlaps."""
        mock_sam = MagicMock()
        distractor_mask = np.zeros((480, 640), dtype=bool)
        distractor_mask[90:120, 90:120] = True
        mock_result = MagicMock()
        mock_result.masks.data.cpu.return_value.numpy.return_value = np.array([distractor_mask])
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam, void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 0,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )
        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)]
        analysis = SceneAnalysis(
            ball_bbox=(95, 95, 105, 105),
            distractors=[{"label": "other ball", "bbox": (85, 85, 125, 125)}],
            scene_description="green",
        )
        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)
        # Ball protection zone center should be 255
        assert quadmasks[0, 100, 100] == 255

    def test_sam3_no_mask_skips_distractor(self):
        """If SAM3 returns no mask for a distractor, skip it gracefully."""
        mock_sam = MagicMock()
        mock_result = MagicMock()
        mock_result.masks = None
        mock_sam.return_value = [mock_result]

        cleaner = FrameCleaner(
            sam3_model=mock_sam, void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )
        video_frames = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        corridors = [Corridor(100, 100, 80, 20, 20, 180, 180)]
        analysis = SceneAnalysis(
            ball_bbox=(95, 95, 105, 105),
            distractors=[{"label": "person", "bbox": (300, 100, 500, 600)}],
            scene_description="green",
        )
        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)
        assert np.all(quadmasks == 255)

    def test_none_corridor_produces_all_255(self):
        """Frames with no corridor should have all-255 quadmask."""
        cleaner = FrameCleaner(
            sam3_model=None, void_model=None,
            corridor_config={
                "corridor_multiplier": 4.0, "corridor_speed_scale": 1.5,
                "radius_scale": 4.0, "mask_dilation_px": 3,
                "max_aspect_ratio": 2.0, "max_size_ratio": 2.0,
            },
        )
        video_frames = np.zeros((2, 100, 100, 3), dtype=np.uint8)
        corridors = [None, None]
        analysis = SceneAnalysis(
            ball_bbox=None,
            distractors=[{"label": "person", "bbox": (10, 10, 50, 50)}],
            scene_description="green",
        )
        quadmasks = cleaner.generate_quadmasks_targeted(video_frames, corridors, analysis, 10.0)
        assert np.all(quadmasks == 255)
