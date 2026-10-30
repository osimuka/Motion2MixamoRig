"""Default mannequin and optional Space character regression tests."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motion2mixamorig.diagnose import _accessor, read_glb
from motion2mixamorig.extractors.base import SMPL_JOINT_NAMES, write_skeleton_npz
from motion2mixamorig.mixamo.fbx_skeleton import HUMAN_TO_MIXAMO
from motion2mixamorig.mixamo.kinematics import forward_kinematics, world_positions_m
from motion2mixamorig.mixamo.mannequin import NAME, build_mannequin
from motion2mixamorig.mixamo.skinned_mesh import rest_skin_error
from motion2mixamorig.mixamo.tpose_calibration import REQUIRED_BONES, calibrate_rig


class MannequinTests(unittest.TestCase):
    def test_glb_preserves_diffuse_samples_and_skin_weights(self):
        from motion2mixamorig.mixamo.gltf import write_glb

        with tempfile.TemporaryDirectory() as temp:
            asset = build_mannequin(Path("mannequin.json"))
            mesh = asset.meshes[0]
            mesh.face_colors = np.tile([0, 0, 128], (len(mesh.faces), 1))
            mesh.face_colors[1] = [255, 0, 0]
            output = write_glb(
                Path(temp) / "colored.glb", asset, [{}],
                [asset.rig.bones["mixamorig:Hips"].rest_world[:3, 3]], 30,
            )
            gltf, binary = read_glb(output)
            primitive = gltf["meshes"][0]["primitives"][0]
            attrs = primitive["attributes"]
            colors = _accessor(gltf, binary, attrs["COLOR_0"])
            expected_red = ((128 / 255 + 0.055) / 1.055) ** 2.4
            np.testing.assert_allclose(colors[:3], np.tile([expected_red, 0, 0], (3, 1)), atol=1e-6)
            np.testing.assert_allclose(colors[3:6], np.tile([0, 0, 1], (3, 1)))
            weights = _accessor(gltf, binary, attrs["WEIGHTS_0"])
            np.testing.assert_allclose(weights.sum(axis=1), 1)
            positions = _accessor(gltf, binary, attrs["POSITION"])
            np.testing.assert_allclose(positions, mesh.vertices[mesh.faces.reshape(-1)] * .01, atol=1e-6)
            self.assertEqual(len(colors), len(weights))
            material = gltf["materials"][primitive["material"]]["pbrMetallicRoughness"]
            self.assertEqual(material["baseColorFactor"], [1, 1, 1, 1])

    def test_tpose_rig_mesh_and_bind_weights(self):
        asset = build_mannequin(Path("mannequin.json"))
        rig = asset.rig
        self.assertTrue(set(REQUIRED_BONES) <= set(rig.order))
        for side, sign in (("Left", 1), ("Right", -1)):
            positions = [
                rig.bones[f"mixamorig:{side}{part}"].rest_world[:3, 3]
                for part in ("Arm", "ForeArm", "Hand")
            ]
            self.assertTrue(all(p[1] == 142 and p[2] == 0 for p in positions))
            self.assertGreater(sign * (positions[-1][0] - positions[0][0]), 0)
        mesh = asset.meshes[0]
        self.assertGreater(len(mesh.faces), 0)
        self.assertEqual(mesh.vertices[:, 1].min(), 0)
        np.testing.assert_allclose(mesh.weights.sum(axis=1), 1)
        self.assertLess(rest_skin_error(asset), 1e-8)
        points = mesh.vertices[mesh.faces]
        area = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
        self.assertTrue(np.all(area > 0))

    def test_pipeline_default_exports_real_skinned_animation_and_diagnostics(self):
        from motion2mixamorig.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            asset = build_mannequin(root / "mannequin.json")
            rest = world_positions_m(asset.rig, forward_kinematics(asset.rig, {}))
            human = np.stack([
                rest[HUMAN_TO_MIXAMO[name]] * [-1, 1, -1]
                for name in SMPL_JOINT_NAMES
            ])
            human[SMPL_JOINT_NAMES.index("left_hand")] = np.array([-0.86, 1.42, 0])
            human[SMPL_JOINT_NAMES.index("right_hand")] = np.array([0.86, 1.42, 0])
            motion = np.stack([human.copy() for _ in range(3)])
            for index in (SMPL_JOINT_NAMES.index("left_wrist"), SMPL_JOINT_NAMES.index("left_hand")):
                motion[1:, index, 1] += 0.15
                motion[1:, index, 0] += 0.05
            skeleton = write_skeleton_npz(
                root / "skeleton_motion.npz", extractor="test", joints_3d=motion, fps=30,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                output = run_pipeline(
                    None, None, image=root / "pose.png", skeleton_npz=skeleton,
                    output_dir=root / "output", no_preview=True, verbose=False,
                )
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["n_frames"], 3)
            self.assertTrue((output / f"{NAME}.json").is_file())
            self.assertTrue((output / "diagnostics.json").is_file())
            gltf, binary = read_glb(output / "mixamo_character.glb")
            self.assertEqual(gltf["asset"]["version"], "2.0")
            self.assertGreater(len(gltf["skins"][0]["joints"]), 20)
            attributes = gltf["meshes"][0]["primitives"][0]["attributes"]
            weights = _accessor(gltf, binary, attributes["WEIGHTS_0"])
            np.testing.assert_allclose(weights.sum(axis=1), 1)
            animation = gltf["animations"][0]
            rotations = [
                _accessor(gltf, binary, animation["samplers"][channel["sampler"]]["output"])
                for channel in animation["channels"] if channel["target"]["path"] == "rotation"
            ]
            self.assertTrue(any(not np.allclose(track[0], track[1]) for track in rotations))
            with np.load(output / "mixamo_rotations.npz") as data:
                self.assertEqual(data["quat_wxyz_rig"].shape[0], 3)
                self.assertIn("wxyz", str(data["convention"]))

    def test_upload_filters_accept_supported_extensions(self):
        import app
        from gradio.data_classes import FileData

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for component, suffixes in (
                (app.video_upload, app.VIDEO_SUFFIXES),
                (app.image_upload, app.IMAGE_SUFFIXES),
            ):
                self.assertEqual(component.file_types, sorted(suffixes))
                for suffix in sorted(suffixes):
                    with self.subTest(suffix=suffix):
                        source = root / f"source{suffix}"
                        source.write_bytes(b"upload filter test")
                        self.assertEqual(
                            component.preprocess(FileData(path=str(source))), str(source)
                        )
                unsupported = root / "source.txt"
                unsupported.write_text("not media")
                with self.assertRaises(app.gr.Error):
                    component.preprocess(FileData(path=str(unsupported)))

    def test_space_optional_and_uploaded_character_selection(self):
        import app

        for custom, configured in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(custom=custom, configured=configured), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                image, rig = root / "pose.png", root / "character.fbx"
                image.write_bytes(b"image")
                rig.write_bytes(b"fbx")
                captured = []

                def pipeline(video, chosen_rig, **kwargs):
                    self.assertEqual(kwargs["detect_hands"], custom)
                    self.assertEqual(kwargs["hand_smooth_s"], 0.12)
                    captured.append(chosen_rig)
                    out = kwargs["output_dir"]
                    out.mkdir()
                    (out / "mixamo_character.glb").write_bytes(b"test")
                    (out / "diagnostics.json").write_text("{}")
                    return out

                with patch.object(app, "require_cuda"), \
                     patch.dict(app.os.environ, {"M2MR_DEFAULT_RIG": str(rig) if configured else ""}), \
                     patch.object(app.checks, "check_image", return_value=None), \
                     patch.object(app.checks, "check_rig", return_value=None) as validate, \
                     patch.object(app, "run_pipeline", side_effect=pipeline), \
                     patch.object(app, "GRADIO_TEMP_DIR", root / "cache"):
                    status, archive = app.generate(
                        "Still image", None, str(image), str(rig) if custom else None,
                        detect_hands=custom, hand_smooth_s=0.12,
                    )
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0] is None, not (custom or configured))
                if custom:
                    self.assertEqual(captured[0].name, "character.fbx")
                elif configured:
                    self.assertEqual(captured[0], rig)
                self.assertEqual(validate.call_count, int(custom or configured))
                self.assertIn("diagnostics.json", status)
                self.assertTrue(Path(archive).is_file())

    def test_missing_configured_default_does_not_silently_fall_back(self):
        import app

        with tempfile.TemporaryDirectory() as temp:
            missing = str(Path(temp) / "missing.fbx")
            with patch.object(app, "require_cuda"), \
                 patch.dict(app.os.environ, {"M2MR_DEFAULT_RIG": missing}), \
                 patch.object(app, "run_pipeline") as pipeline:
                with self.assertRaisesRegex(app.gr.Error, "default character is missing"):
                    app.generate("Video", None, None, None)
            pipeline.assert_not_called()

    def test_invalid_uploaded_character_does_not_fall_back_to_mannequin(self):
        import app

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rig = root / "broken.fbx"
            rig.write_bytes(b"not an FBX")
            with patch.object(app, "require_cuda"), \
                 patch.object(app.checks, "check_rig", return_value="invalid FBX"), \
                 patch.object(app, "run_pipeline") as pipeline:
                with self.assertRaisesRegex(app.gr.Error, "invalid FBX"):
                    app.generate("Video", None, None, str(rig))
            pipeline.assert_not_called()


class NativeSamHandTests(unittest.TestCase):
    def setUp(self):
        from motion2mixamorig.mixamo.skinned_mesh import load_character_asset

        self.rig_path = Path(__file__).resolve().parents[1] / "private_assets/default_mannequin.fbx"
        self.asset = load_character_asset(self.rig_path, textures=False)
        self.world = forward_kinematics(self.asset.rig, {})
        self.points = np.zeros((2, 21, 3))
        for side_index, side in enumerate(("Left", "Right")):
            self.points[side_index, 0] = self.world[f"mixamorig:{side}Hand"][:3, 3]
            for finger_index, finger in enumerate(("Thumb", "Index", "Middle", "Ring", "Pinky")):
                base = 1 + finger_index * 4
                for joint in range(3):
                    self.points[side_index, base + joint] = self.world[
                        f"mixamorig:{side}Hand{finger}{joint + 1}"
                    ][:3, 3]
                self.points[side_index, base + 3] = (
                    2 * self.points[side_index, base + 2] - self.points[side_index, base + 1]
                )

    def write_motion(self, root, hand_points):
        rest = world_positions_m(self.asset.rig, self.world)
        human = np.stack([rest[HUMAN_TO_MIXAMO[name]] * [-1, 1, -1] for name in SMPL_JOINT_NAMES])
        return write_skeleton_npz(
            root / "skeleton.npz", extractor="sam3d", fps=30,
            joints_3d=np.repeat(human[None], len(hand_points), axis=0),
            sam_hand_keypoints=hand_points, hands=np.array("sam3d_native"),
        )

    def bent(self):
        from scipy.spatial.transform import Rotation

        points = self.points.copy()
        for side in range(2):
            normal = np.cross(points[side, 9] - points[side, 0], points[side, 5] - points[side, 17])
            normal /= np.linalg.norm(normal)
            rotation = Rotation.from_rotvec(normal * .7).as_matrix()
            pivot = points[side, 6].copy()
            points[side, 7:9] = (points[side, 7:9] - pivot) @ rotation.T + pivot
        return points

    def test_neutral_bent_and_camera_invariant_rotations(self):
        from scipy.spatial.transform import Rotation
        from motion2mixamorig.mixamo.sam_hand_retarget import load_sam_hands

        points = np.stack([self.points, self.bent()])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hands, mapping = load_sam_hands(self.write_motion(root, points), self.asset.rig, smooth_s=0)
            self.assertEqual(len(mapping.bones), 30)
            for rotation in mapping.finger_rotations(hands.pose[0], hands.valid[0]).values():
                np.testing.assert_allclose(rotation, np.eye(3), atol=1e-6)
            for side in ("Left", "Right"):
                rotation = mapping.finger_rotations(hands.pose[1], hands.valid[1])[
                    f"mixamorig:{side}HandIndex2"
                ]
                self.assertFalse(np.allclose(rotation, np.eye(3)))
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                self.assertAlmostEqual(np.linalg.det(rotation), 1)
            camera = Rotation.from_euler("xyz", [.4, .2, -.8]).as_matrix()
            transformed = points @ camera.T + [4, 5, 6]
            changed, _ = load_sam_hands(self.write_motion(root, transformed), self.asset.rig, smooth_s=0)
            np.testing.assert_allclose(changed.pose, hands.pose, atol=1e-6)

    def test_invalid_hands_and_rigs_report_errors(self):
        from motion2mixamorig.mixamo.sam_hand_retarget import load_sam_hands

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.write_motion(root, np.zeros((1, 2, 21, 3)))
            with self.assertRaisesRegex(ValueError, "no geometrically usable"):
                load_sam_hands(output, self.asset.rig)
            output = self.write_motion(root, self.points[None])
            with self.assertRaisesRegex(ValueError, "finger bones"):
                load_sam_hands(output, build_mannequin(root / "rig.json").rig)
            with self.assertRaisesRegex(ValueError, "smoothing"):
                load_sam_hands(output, self.asset.rig, smooth_s=-1)
            malformed = np.full((1, 2, 21, 3), np.nan)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_sam_hands(self.write_motion(root, malformed), self.asset.rig)

    def test_body_only_sam_reuse_cannot_silently_run_hamer(self):
        from motion2mixamorig.pipeline import run_pipeline
        from motion2mixamorig.cli import build_parser, preflight

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            skeleton = write_skeleton_npz(
                root / "body.npz", extractor="sam3d", fps=30,
                joints_3d=np.zeros((1, 24, 3)),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "no saved native hands"):
                    run_pipeline(
                        None, self.rig_path, image=root / "pose.png", skeleton_npz=skeleton,
                        output_dir=root / "output", no_preview=True, verbose=False, detect_hands=True,
                    )
                args = build_parser().parse_args([
                    "run", "--skeleton", str(skeleton), "--rig", str(self.rig_path),
                    "--set-extractor", "sam3d", "--detect-hands",
                ])
                with self.assertRaises(SystemExit):
                    preflight(args)

    def test_native_fingers_reach_npz_and_glb_with_skeleton_reuse(self):
        from motion2mixamorig.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            skeleton = self.write_motion(root, np.stack([self.points, self.bent(), self.points]))
            with contextlib.redirect_stdout(io.StringIO()):
                output = run_pipeline(
                    None, self.rig_path, image=root / "pose.png", skeleton_npz=skeleton,
                    output_dir=root / "output", no_preview=True, verbose=False,
                    detect_hands=True, hand_smooth_s=0,
                )
            with np.load(output / "mixamo_rotations.npz") as data:
                names = list(data["bone_names"])
                for side in ("Left", "Right"):
                    bone = names.index(f"mixamorig:{side}HandIndex2")
                    self.assertFalse(np.allclose(data["quat_wxyz_rig"][0, bone], data["quat_wxyz_rig"][1, bone]))
            gltf, binary = read_glb(output / "mixamo_character.glb")
            for side in ("Left", "Right"):
                node = next(i for i, n in enumerate(gltf["nodes"]) if n["name"] == f"mixamorig:{side}HandIndex2")
                animation = gltf["animations"][0]
                channel = next(c for c in animation["channels"] if c["target"] == {"node": node, "path": "rotation"})
                track = _accessor(gltf, binary, animation["samplers"][channel["sampler"]]["output"])
                self.assertFalse(np.allclose(track[0], track[1]))
            self.assertTrue((output / "diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
