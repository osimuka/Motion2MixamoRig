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

    def test_space_optional_and_uploaded_character_selection(self):
        import app

        for custom in (False, True):
            with self.subTest(custom=custom), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                image, rig = root / "pose.png", root / "character.fbx"
                image.write_bytes(b"image")
                rig.write_bytes(b"fbx")
                captured = []

                def pipeline(video, chosen_rig, **kwargs):
                    captured.append(chosen_rig)
                    out = kwargs["output_dir"]
                    out.mkdir()
                    (out / "mixamo_character.glb").write_bytes(b"test")
                    (out / "diagnostics.json").write_text("{}")
                    return out

                with patch.object(app, "require_cuda"), \
                     patch.object(app.checks, "check_image", return_value=None), \
                     patch.object(app.checks, "check_rig", return_value=None) as validate, \
                     patch.object(app, "run_pipeline", side_effect=pipeline), \
                     patch.object(app, "GRADIO_TEMP_DIR", root / "cache"):
                    status, archive = app.generate(
                        "Still image", None, str(image), str(rig) if custom else None,
                    )
                self.assertEqual(len(captured), 1)
                self.assertEqual(captured[0] is None, not custom)
                self.assertEqual(validate.call_count, int(custom))
                self.assertIn("diagnostics.json", status)
                self.assertTrue(Path(archive).is_file())

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


if __name__ == "__main__":
    unittest.main()
