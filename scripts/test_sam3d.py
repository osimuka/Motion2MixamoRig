"""Focused SAM adapter tests; run with python -m unittest scripts.test_sam3d."""

from __future__ import annotations

import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from motion2mixamorig.extractors import DEFAULT_EXTRACTOR, get_extractor
from motion2mixamorig.extractors.base import SMPL_JOINT_NAMES, write_skeleton_npz
from motion2mixamorig.extractors.sam3d import (
    MainPersonTracker,
    Sam3dExtractor,
    infer_frame,
    map_keypoints,
    smooth_and_align,
)
from motion2mixamorig.human_motion import load_human_motion


class SamMappingTests(unittest.TestCase):
    def test_missing_dependency_error_names_actual_module(self):
        from motion2mixamorig.errors import classify_error

        missing = ModuleNotFoundError("No module named 'termcolor'", name="termcolor")
        self.assertEqual(classify_error(missing), ("PACKAGE_MISSING", str(missing)))
        package = ModuleNotFoundError(
            "No module named 'motion2mixamorig'", name="motion2mixamorig"
        )
        self.assertEqual(
            classify_error(package),
            ("PACKAGE_MISSING", "Motion2MixamoRig package not installed"),
        )

    def test_mapping_units_axes_laterality_and_camera_translation(self):
        points = np.arange(70 * 3, dtype=float).reshape(70, 3) / 100
        camera = np.array([1.0, 2.0, 3.0])
        mapped = map_keypoints(points, camera)
        converted = (points + camera) * [1, -1, -1]
        self.assertEqual(mapped.shape, (24, 3))
        for name, index in (
            ("left_hip", 9), ("right_hip", 10), ("left_wrist", 62),
            ("right_wrist", 41), ("neck", 69),
        ):
            np.testing.assert_allclose(mapped[SMPL_JOINT_NAMES.index(name)], converted[index])
        np.testing.assert_allclose(mapped[0], converted[[9, 10]].mean(axis=0))
        np.testing.assert_allclose(mapped[10], converted[[15, 16]].mean(axis=0))
        np.testing.assert_allclose(mapped[15], converted[[3, 4]].mean(axis=0))
        np.testing.assert_allclose(mapped[3], mapped[0] + (mapped[9] - mapped[0]) / 3)

    def test_invalid_predictions_rejected(self):
        for points, camera in (
            (np.zeros((69, 3)), np.zeros(3)),
            (np.full((70, 3), np.nan), np.zeros(3)),
            (np.zeros((70, 3)), np.ones((1, 3))),
            (np.zeros((70, 3)), np.array([0, np.inf, 0])),
        ):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                map_keypoints(points, camera)

    def test_smoothing_keeps_translation_and_shared_joint_offsets(self):
        joints = np.ones((60, 24, 3))
        joints[:, :, 0] = np.arange(60)[:, None] / 30
        joints[:, 0, 1] = 2
        result = smooth_and_align(joints, 30)
        np.testing.assert_allclose(result[0, 0], [0, 1, 0])
        self.assertGreater(result[-1, 0, 0], 1.8)
        np.testing.assert_allclose(result[:, 0, 1] - result[:, 7, 1], 1)

    def test_smoothing_reduces_jitter(self):
        joints = np.zeros((60, 24, 3))
        joints[:, :, 1] = np.arange(60)[:, None] % 2
        result = smooth_and_align(joints, 30)
        self.assertLess(float(np.std(np.diff(result[:, 0, 1]))), 0.1)

    def test_nonfinite_or_empty_motion_rejected(self):
        for joints, fps in (
            (np.zeros((0, 24, 3)), 30),
            (np.zeros((2, 24, 3)), 0),
            (np.zeros((2, 24, 3)), np.inf),
            (np.full((2, 24, 3), np.nan), 30),
        ):
            with self.assertRaises(ValueError):
                smooth_and_align(joints, fps)

    def test_npz_contract_and_60fps_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_skeleton_npz(
                Path(temp) / "skeleton_motion.npz", extractor="sam3d",
                joints_3d=np.zeros((120, 24, 3)), fps=60,
            )
            motion = load_human_motion(path)
            self.assertEqual(motion.joint_names, list(SMPL_JOINT_NAMES))
            self.assertEqual(motion.n_frames / motion.fps, 2.0)
            self.assertAlmostEqual(motion.timestamps[-1], 119 / 60)
            self.assertIsNone(motion.foot_contact)


class MainPersonTrackingTests(unittest.TestCase):
    def test_selects_dominant_person_and_follows_despite_larger_background_box(self):
        tracker = MainPersonTracker()
        with self.assertLogs("motion2mixamorig.extractors.sam3d", level="WARNING"):
            selected = tracker.select(np.array([[300, 0, 330, 60], [0, 0, 100, 200]]), 0)
        np.testing.assert_array_equal(selected, [[0, 0, 100, 200]])
        selected = tracker.select(np.array([[300, 0, 500, 300], [10, 0, 110, 200]]), 1)
        np.testing.assert_array_equal(selected, [[10, 0, 110, 200]])
        self.assertEqual(tracker.extra_detection_frames, 2)

    def test_similar_sized_initial_people_are_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "frame 0: ambiguous"):
            MainPersonTracker().select(np.array([[0, 0, 100, 200], [300, 0, 400, 200]]), 0)

    def test_lost_or_ambiguous_match_never_updates_track(self):
        for boxes, error in (
            (np.empty((0, 4)), "lost"),
            (np.array([[300, 0, 400, 200]]), "lost"),
            (np.array([[0, 0, 100, 200], [5, 0, 105, 200]]), "ambiguous"),
            (np.array([[0, 0, 300, 600]]), "lost"),
        ):
            with self.subTest(boxes=boxes):
                tracker = MainPersonTracker()
                tracker.select(np.array([[0, 0, 100, 200]]), 0)
                with self.assertRaisesRegex(ValueError, f"frame 62: .*{error}"):
                    tracker.select(boxes, 62)
                self.assertEqual(len(tracker.boxes), 1)

    def test_overlap_threshold_and_margin(self):
        tracker = MainPersonTracker()
        tracker.select(np.array([[0, 0, 100, 100]]), 0)
        # A 50-pixel shift has IoU 1/3 and is accepted.
        np.testing.assert_array_equal(
            tracker.select(np.array([[50, 0, 150, 100]]), 1), [[50, 0, 150, 100]],
        )
        with self.assertRaisesRegex(ValueError, "lost"):
            tracker.select(np.array([[110, 0, 210, 100]]), 2)
        tracker = MainPersonTracker()
        tracker.select(np.array([[0, 0, 100, 100]]), 0)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            tracker.select(np.array([[50, 0, 150, 100], [-60, 0, 40, 100]]), 1)

    def test_invalid_boxes_rejected(self):
        for boxes in (
            np.zeros((2, 4)), np.ones((4,)), np.ones((1, 3)),
            np.array([[0, 0, np.inf, 100]]),
        ):
            with self.assertRaisesRegex(ValueError, "invalid person boxes"):
                MainPersonTracker().select(boxes, 0)

    def test_pipeline_and_gvhmr_reject_unsupported_tracking(self):
        from motion2mixamorig.pipeline import run_pipeline

        with self.assertRaisesRegex(ValueError, "requires sam3d"):
            run_pipeline(Path("clip.mp4"), None, track_person=True)
        with self.assertRaisesRegex(ValueError, "without --skeleton"):
            run_pipeline(
                Path("clip.mp4"), None, extractor="sam3d", track_person=True,
                skeleton_npz=Path("saved.npz"),
            )
        with self.assertRaisesRegex(ValueError, "only by the sam3d"):
            get_extractor("gvhmr").extract(
                Path("clip.mp4"), Path("out.npz"), work_dir=Path("."), track_person=True,
            )


class SamInferenceTests(unittest.TestCase):
    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_tracker_passes_only_selected_box_to_estimator(self, boxes):
        boxes.return_value = np.array([[0, 0, 100, 200], [300, 0, 330, 60]])
        estimator = Mock()
        estimator.process_one_image.return_value = [{
            "pred_keypoints_3d": np.zeros((70, 3)), "pred_cam_t": np.ones(3),
        }]
        with self.assertLogs("motion2mixamorig.extractors.sam3d", level="WARNING"):
            infer_frame(
                np.zeros((8, 8, 3), dtype=np.uint8), estimator, Mock(), 0,
                tracker=MainPersonTracker(),
            )
        np.testing.assert_array_equal(
            estimator.process_one_image.call_args.kwargs["bboxes"], [[0, 0, 100, 200]],
        )

    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_missing_or_multiple_people_fail_instead_of_skipping_frames(self, boxes):
        for count in (0, 2):
            boxes.return_value = np.zeros((count, 4))
            with self.assertRaisesRegex(ValueError, "frame 17"):
                infer_frame(np.zeros((8, 8, 3), dtype=np.uint8), Mock(), Mock(), 17)

    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_rgb_and_body_only_estimator_contract(self, boxes):
        boxes.return_value = np.array([[0, 0, 8, 8]])
        estimator = Mock()
        estimator.process_one_image.return_value = [{
            "pred_keypoints_3d": np.zeros((308, 3)), "pred_cam_t": np.ones(3),
        }]
        frame = np.full((8, 8, 3), [1, 2, 3], dtype=np.uint8)
        result = infer_frame(frame, estimator, Mock(), 0)
        self.assertEqual(result.shape, (24, 3))
        args, kwargs = estimator.process_one_image.call_args
        np.testing.assert_array_equal(args[0][0, 0], [3, 2, 1])
        self.assertEqual(kwargs["inference_type"], "body")

    def test_default_and_native_finger_backend(self):
        self.assertEqual(DEFAULT_EXTRACTOR, "gvhmr")
        self.assertIsInstance(get_extractor("sam3d"), Sam3dExtractor)
        self.assertEqual(get_extractor("sam3d").hand_model, "sam3d_native")

    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_native_full_inference_and_left_right_finger_order(self, boxes):
        boxes.return_value = np.array([[0, 0, 8, 8]])
        points = np.arange(70 * 3, dtype=float).reshape(70, 3) / 100
        estimator = Mock()
        estimator.process_one_image.return_value = [{
            "pred_keypoints_3d": points, "pred_cam_t": np.ones(3),
        }]
        hands = []
        infer_frame(np.zeros((8, 8, 3), dtype=np.uint8), estimator, Mock(), 0, hand_frames=hands)
        self.assertEqual(estimator.process_one_image.call_args.kwargs["inference_type"], "full")
        self.assertEqual(np.stack(hands).shape, (1, 2, 21, 3))
        for side, wrist, base in ((0, 62, 42), (1, 41, 21)):
            np.testing.assert_allclose(hands[0][side, 0], points[wrist] * [1, -1, -1])
            for finger in range(5):
                for joint in range(4):
                    np.testing.assert_allclose(
                        hands[0][side, 1 + finger * 4 + joint],
                        points[base + finger * 4 + 3 - joint] * [1, -1, -1],
                        atol=1e-6,
                    )

    @patch("motion2mixamorig.extractors.sam3d.require_cuda")
    @patch("motion2mixamorig.extractors.sam3d.load_estimator")
    @patch("motion2mixamorig.extractors.sam3d.load_person_detector")
    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_native_still_hand_arrays_repeat_with_body(self, boxes, _det, estimator, _cuda):
        import cv2
        from motion2mixamorig.extractors.hamer import skeleton_has_hands

        boxes.return_value = np.array([[0, 0, 8, 8]])
        estimator.return_value.process_one_image.return_value = [{
            "pred_keypoints_3d": np.arange(70 * 3, dtype=float).reshape(70, 3) / 100,
            "pred_cam_t": np.ones(3),
        }]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "pose.png"
            cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
            for tracking in (False, True):
                with self.subTest(tracking=tracking):
                    output = Sam3dExtractor().extract(
                        image, root / "skeleton.npz", work_dir=root, device="cuda",
                        detect_hands=True, track_person=tracking,
                    )
                    self.assertTrue(skeleton_has_hands(output))
                    with np.load(output) as data:
                        hands = data["sam_hand_keypoints"]
                        self.assertEqual(hands.shape, (32, 2, 21, 3))
                        np.testing.assert_array_equal(hands[0], hands[-1])
                        self.assertEqual(str(data["hands"]), "sam3d_native")
                        self.assertNotIn("hand_pose", data.files)
                        self.assertEqual(
                            str(data["person_selection"]), "main_person" if tracking else "strict",
                        )
                        if tracking:
                            self.assertEqual(data["sam_person_boxes"].shape, (32, 4))
                            np.testing.assert_array_equal(data["sam_person_boxes"][0], [0, 0, 8, 8])
                            self.assertEqual(int(data["sam_extra_detection_frames"]), 0)

    @patch("motion2mixamorig.checks._count_people", side_effect=AssertionError("YOLO imported"))
    def test_decode_checks_can_avoid_gvhmr_detector(self, _count):
        import cv2
        from motion2mixamorig import checks

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8)))
            self.assertIsNone(checks.check_image(path, check_people=False))

    @patch("motion2mixamorig.extractors.sam3d.require_cuda")
    @patch("motion2mixamorig.extractors.sam3d.load_estimator")
    @patch("motion2mixamorig.extractors.sam3d.load_person_detector")
    @patch("motion2mixamorig.extractors.sam3d.infer_frame")
    def test_video_extract_preserves_every_frame_and_duration(self, infer, _det, _est, _cuda):
        import cv2

        infer.return_value = np.ones((24, 3))
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(120):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            result = Sam3dExtractor().extract(
                video, Path(temp) / "skeleton_motion.npz", work_dir=Path(temp), device="cuda",
                verbose=False,
            )
            motion = load_human_motion(result)
            self.assertEqual(infer.call_count, 120)
            self.assertEqual(motion.n_frames, 120)
            self.assertAlmostEqual(motion.n_frames / motion.fps, 2, places=3)

    @patch("motion2mixamorig.extractors.sam3d.require_cuda")
    @patch("motion2mixamorig.extractors.sam3d.load_estimator")
    @patch("motion2mixamorig.extractors.sam3d.load_person_detector")
    @patch("motion2mixamorig.extractors.sam3d.person_boxes")
    def test_video_native_hands_align_with_every_body_frame(self, boxes, _det, estimator, _cuda):
        import cv2

        boxes.side_effect = [
            np.array([[0, 0, 20, 32]]),
            np.array([[26, 0, 32, 16], [1, 0, 21, 32]]),
            np.array([[2, 0, 22, 32]]),
        ]
        points = np.arange(70 * 3, dtype=float).reshape(70, 3) / 100
        estimator.return_value.process_one_image.side_effect = [
            [{"pred_keypoints_3d": points + offset, "pred_cam_t": np.ones(3)}]
            for offset in (0, 1, 2)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            with self.assertLogs("motion2mixamorig.extractors.sam3d", level="WARNING"):
                output = Sam3dExtractor().extract(
                    video, root / "skeleton.npz", work_dir=root, device="cuda",
                    detect_hands=True, track_person=True,
                )
            with np.load(output) as data:
                hands = data["sam_hand_keypoints"]
                self.assertEqual(len(hands), len(data["joints_3d"]))
                self.assertEqual(hands.shape, (3, 2, 21, 3))
                np.testing.assert_allclose(hands[2] - hands[0], np.tile([2, -2, -2], (2, 21, 1)), atol=1e-6)
                self.assertEqual(str(data["person_selection"]), "main_person")
                np.testing.assert_array_equal(
                    data["sam_person_boxes"],
                    [[0, 0, 20, 32], [1, 0, 21, 32], [2, 0, 22, 32]],
                )
                self.assertEqual(int(data["sam_extra_detection_frames"]), 1)

    @patch("motion2mixamorig.checks._count_people", side_effect=AssertionError("YOLO imported"))
    @patch("motion2mixamorig.checks.check_smplx", side_effect=AssertionError("SMPL-X checked"))
    @patch("motion2mixamorig.checks.check_rig", return_value=None)
    @patch.object(Sam3dExtractor, "check", return_value=[])
    def test_cli_preflight_sam_avoids_smplx_and_yolo(self, _requirements, _rig, _smpl, _people):
        import cv2
        from motion2mixamorig.cli import build_parser, preflight

        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "still.png"
            rig = Path(temp) / "character.fbx"
            rig.write_bytes(b"test")
            cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
            args = build_parser().parse_args([
                "run", "--image", str(image), "--rig", str(rig),
                "--set-extractor", "sam3d", "--device", "cuda",
                "--track-person",
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                _, chosen_rig, _, chosen_image = preflight(args)
            self.assertEqual(chosen_rig, rig)
            self.assertEqual(chosen_image, image)


if __name__ == "__main__":
    unittest.main()
