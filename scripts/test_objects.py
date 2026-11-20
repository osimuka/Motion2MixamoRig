"""Focused tests for optional video object reconstruction and export."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np
import trimesh
from PIL import Image

from motion2mixamorig.diagnose import _accessor, read_glb
from motion2mixamorig.object_export import prepare_object_mesh, write_object_glb
from motion2mixamorig.object_sam2 import mask_bounding_box
from motion2mixamorig.object_tracking import (
    ObjectRuntimes,
    ObjectSelection,
    capture_object,
    object_runtimes,
    validate_video_times,
)


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("The test video writer could not start.")
    try:
        for index in range(4):
            image = np.full((24, 32, 3), 30 + index * 10, dtype=np.uint8)
            image[4:16, 5:20] = [20, 40, 220]
            writer.write(image)
    finally:
        writer.release()


class ObjectExportTests(unittest.TestCase):
    def test_texture_atlas_matches_face_colors_across_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = trimesh.creation.box(extents=[2, 1, 1])
            mesh.visual.vertex_colors = np.where(
                np.asarray(mesh.vertices)[:, :1] < 0,
                [255, 0, 0, 255], [0, 0, 255, 255],
            )
            source = root / "source.glb"
            source.write_bytes(mesh.export(file_type="glb"))
            _, static_glb = prepare_object_mesh(source, root / "mesh", 0.2)
            gltf, binary = read_glb(static_glb)
            primitive = gltf["meshes"][0]["primitives"][0]
            positions = _accessor(gltf, binary, primitive["attributes"]["POSITION"])
            uv = _accessor(gltf, binary, primitive["attributes"]["TEXCOORD_0"])
            view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
            start = view.get("byteOffset", 0)
            with Image.open(io.BytesIO(binary[start:start + view["byteLength"]])) as image:
                pixels = np.asarray(image.convert("RGB"))
            x = np.floor(uv[:, 0] * pixels.shape[1]).astype(int)
            y = np.floor(uv[:, 1] * pixels.shape[0]).astype(int)
            red_fraction = (positions[:, 0].reshape(-1, 3) < 0).mean(axis=1)
            expected = np.rint(np.column_stack([
                red_fraction * 255, np.zeros(len(red_fraction)), (1 - red_fraction) * 255,
            ])).astype(np.uint8)
            np.testing.assert_array_equal(pixels[y, x], np.repeat(expected, 3, axis=0))

    def test_textured_glb_preserves_pose_timing_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
            mesh.visual.vertex_colors = np.tile([240, 40, 10, 255], (len(mesh.vertices), 1))
            source = root / "source.glb"
            source.write_bytes(mesh.export(file_type="glb"))
            obj, static_glb = prepare_object_mesh(source, root / "mesh", 0.2)
            self.assertTrue(obj.is_file())
            self.assertEqual(len(list(obj.parent.glob("*.mtl"))), 1)
            self.assertEqual(len(list(obj.parent.glob("*.png"))), 1)

            times = np.arange(4, dtype=np.float64) / 60
            poses = np.repeat(np.eye(4)[None], 4, axis=0)
            poses[:, :3, 3] = [[1, 2, 3], [0, 0, 0], [4, 5, 6], [7, 8, 9]]
            valid = np.array([True, False, True, True])
            poses[1] = np.nan
            output = write_object_glb(static_glb, root / "animated.glb", times, poses, valid)
            gltf, binary = read_glb(output)

            primitive = gltf["meshes"][0]["primitives"][0]
            positions = _accessor(gltf, binary, primitive["attributes"]["POSITION"])
            self.assertAlmostEqual(float(np.ptp(positions, axis=0).max()), 0.2, places=5)
            self.assertIn("TEXCOORD_0", primitive["attributes"])
            self.assertEqual(gltf["images"][0]["mimeType"], "image/png")
            view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
            start = view.get("byteOffset", 0)
            png = binary[start:start + view["byteLength"]]
            with Image.open(io.BytesIO(png)) as image:
                self.assertEqual(image.convert("RGB").getpixel((0, 0)), (240, 40, 10))
            animation = gltf["animations"][0]
            channels = {channel["target"]["path"]: channel for channel in animation["channels"]}

            def track(name):
                sampler = animation["samplers"][channels[name]["sampler"]]
                return _accessor(gltf, binary, sampler["output"])

            time_sampler = animation["samplers"][channels["translation"]["sampler"]]
            np.testing.assert_allclose(
                _accessor(gltf, binary, time_sampler["input"]).ravel(), times, atol=1e-7
            )
            np.testing.assert_allclose(track("translation")[[0, 2]], [[1, -2, -3], [4, -5, -6]])
            np.testing.assert_allclose(track("scale")[:, 0], [1, 0, 1, 1])
            self.assertEqual(animation["samplers"][channels["scale"]["sampler"]]["interpolation"], "STEP")
            np.testing.assert_allclose(np.abs(track("rotation")[0]), [1, 0, 0, 0], atol=1e-6)

    def test_rejects_bad_mesh_and_invalid_pose_tracks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = trimesh.creation.box()
            mesh.visual.vertex_colors = [255, 0, 0, 255]
            source = root / "source.glb"
            source.write_bytes(mesh.export(file_type="glb"))
            with self.assertRaisesRegex(ValueError, "size"):
                prepare_object_mesh(source, root / "mesh", 0)
            _, static_glb = prepare_object_mesh(source, root / "mesh", 0.2)
            with self.assertRaisesRegex(ValueError, "no valid"):
                write_object_glb(
                    static_glb, root / "empty.glb", np.array([0.0]),
                    np.full((1, 4, 4), np.nan), np.array([False]),
                )
            pose = np.eye(4)[None]
            pose[0, 0, 0] = 2
            with self.assertRaisesRegex(ValueError, "rotation"):
                write_object_glb(
                    static_glb, root / "bad.glb", np.array([0.0]), pose, np.array([True])
                )


class ObjectTrackingTests(unittest.TestCase):
    def test_mask_box_uses_pixel_edges_and_reports_empty_masks(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 3:7] = True
        self.assertEqual(mask_bounding_box(mask), (3, 2, 7, 5))
        self.assertIsNone(mask_bounding_box(np.zeros((8, 10), dtype=bool)))

    def test_sam2_propagates_both_directions_and_keeps_empty_frames(self):
        from motion2mixamorig.object_sam2 import segment_video

        class Tensor:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, index):
                return Tensor(self.value[index])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        def result(index):
            mask = np.full((1, 1, 6, 8), -1.0)
            if index != 2:
                mask[0, 0, 1:5, 2:6] = 1
            return index, [1], Tensor(mask)

        class Predictor:
            def init_state(self, **kwargs):
                self.assert_offload = kwargs["offload_state_to_cpu"] and kwargs["offload_video_to_cpu"]
                return {"num_frames": 3, "video_height": 6, "video_width": 8}

            def add_new_points_or_box(self, **kwargs):
                self.box = kwargs["box"]
                return result(1)

            def propagate_in_video(self, state, *, start_frame_idx, reverse):
                self.start = start_frame_idx
                return iter([result(1), result(0 if reverse else 2)])

        predictor = Predictor()
        sam2 = ModuleType("sam2")
        build = ModuleType("sam2.build_sam")
        build.build_sam2_video_predictor = lambda *args, **kwargs: predictor
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), inference_mode=nullcontext)
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            sys.modules, {"torch": torch, "sam2": sam2, "sam2.build_sam": build}
        ):
            root = Path(temp)
            segment_video(
                root, 3, 1, np.array([2, 1, 6, 5]), root / "weights",
                root / "masks.npz", root / "seed.png",
            )
            with np.load(root / "masks.npz") as data:
                np.testing.assert_array_equal(data["visible"], [True, True, False])
                np.testing.assert_array_equal(data["boxes"][0], [2, 1, 6, 5])
                self.assertTrue(np.isnan(data["boxes"][2]).all())
            self.assertTrue((root / "seed.png").is_file())
            self.assertTrue(predictor.assert_offload)
            self.assertEqual(predictor.start, 1)

    def test_megapose_estimates_only_visible_frames(self):
        from motion2mixamorig.object_megapose import estimate_video

        class Tensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.eye(4, dtype=np.float32)[None]

        class CudaValue:
            def cuda(self):
                return self

        class Observation(CudaValue):
            @staticmethod
            def from_numpy(rgb, depth, camera):
                self.assertEqual(rgb.shape, (6, 8, 3))
                self.assertIsNone(depth)
                self.assertAlmostEqual(float(camera[0, 0]), 10)
                return Observation()

        class Estimator(CudaValue):
            def __init__(self):
                self.calls = 0
                self.renderer = SimpleNamespace(stopped=False)
                self.renderer.stop = lambda: setattr(self.renderer, "stopped", True)
                self.refiner_model = SimpleNamespace(renderer=self.renderer)

            def run_inference_pipeline(self, observation, **kwargs):
                self.calls += 1
                return SimpleNamespace(poses=Tensor()), None

        estimator = Estimator()
        class ObjectDatasetModule(ModuleType):
            def __getattribute__(self, name):
                if name == "RigidObject":
                    # Simulate MegaPose's Slurm import changing the renderer device.
                    os.environ["EGL_VISIBLE_DEVICES"] = "2"
                return super().__getattribute__(name)

        modules = {}
        for name in (
            "megapose", "megapose.datasets", "megapose.datasets.object_dataset",
            "megapose.datasets.scene_dataset", "megapose.inference", "megapose.inference.types",
            "megapose.inference.utils", "megapose.utils", "megapose.utils.load_model",
        ):
            modules[name] = ModuleType(name)
        modules["megapose.datasets.object_dataset"] = ObjectDatasetModule(
            "megapose.datasets.object_dataset"
        )
        modules["megapose.datasets.object_dataset"].RigidObject = lambda **kwargs: kwargs
        modules["megapose.datasets.object_dataset"].RigidObjectDataset = lambda objects: objects
        modules["megapose.datasets.scene_dataset"].ObjectData = lambda **kwargs: kwargs
        modules["megapose.inference.types"].ObservationTensor = Observation
        modules["megapose.inference.utils"].make_detections_from_object_data = lambda objects: CudaValue()
        modules["megapose.utils.load_model"].NAMED_MODELS = {
            "megapose-1.0-RGB": {"inference_parameters": {}},
        }
        def load_model(*args, **kwargs):
            self.assertEqual(os.environ["LOCAL_RANK"], "0")
            self.assertEqual(os.environ["EGL_VISIBLE_DEVICES"], "7")
            return estimator

        modules["megapose.utils.load_model"].load_named_model = load_model
        modules["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True), no_grad=nullcontext,
        )
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), \
             patch.dict(os.environ, {
                 "CONDA_PREFIX": "previous", "LOCAL_RANK": "4", "EGL_VISIBLE_DEVICES": "7",
             }):
            root = Path(temp)
            for index in range(3):
                self.assertTrue(cv2.imwrite(str(root / f"{index:05d}.jpg"), np.zeros((6, 8, 3), np.uint8)))
            np.savez_compressed(
                root / "masks.npz",
                boxes=[[1, 1, 5, 5], [np.nan] * 4, [1, 1, 5, 5]],
                visible=[True, False, True],
            )
            estimate_video(root, root / "masks.npz", root / "object.obj", root / "poses.npz")
            with np.load(root / "poses.npz") as data:
                np.testing.assert_array_equal(data["valid"], [True, False, True])
                self.assertTrue(np.isnan(data["camera_from_object"][1]).all())
                np.testing.assert_allclose(data["camera_from_object"][0], np.eye(4))
            self.assertEqual(estimator.calls, 2)
            self.assertTrue(estimator.renderer.stopped)

    def test_source_times_must_match_character_frame_timing(self):
        times = np.arange(60, dtype=np.float64) / 60
        validate_video_times(times + 3.0, times)
        with self.assertRaisesRegex(ValueError, "frame"):
            validate_video_times(times[:-1], times)
        with self.assertRaisesRegex(ValueError, "constant-frame-rate"):
            validate_video_times(times + np.linspace(0, 0.1, len(times)), times)

    def test_runtime_keeps_virtualenv_python_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            triposr = root / "TripoSR"
            triposr.mkdir()
            (triposr / "run.py").write_text("")
            data = root / "megapose-data"
            for name in ("coarse-rgb-906902141", "refiner-rgb-653307694"):
                model = data / "megapose-models" / name
                model.mkdir(parents=True)
                (model / "config.yaml").write_text("")
                (model / "checkpoint.pth.tar").write_bytes(b"test")
            with patch.dict(os.environ, {
                "M2MR_SAM2_PYTHON": str(python), "M2MR_TRIPOSR_PYTHON": str(python),
                "M2MR_TRIPOSR_DIR": str(triposr), "M2MR_MEGAPOSE_PYTHON": str(python),
                "MEGAPOSE_DATA_DIR": str(data),
            }):
                result = object_runtimes()
            self.assertEqual(result.sam2_python, python)
            self.assertTrue(result.sam2_python.is_symlink())

    def test_nonzero_cuda_requires_explicit_egl_device(self):
        from motion2mixamorig.object_tracking import _object_gpu_devices, _run_stage

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_object_gpu_devices(), ("0", "0"))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "EGL_VISIBLE_DEVICES"):
                _object_gpu_devices()
        with patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "0", "EGL_VISIBLE_DEVICES": "",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "numeric EGL"):
                _object_gpu_devices()
        with patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "2,3", "EGL_VISIBLE_DEVICES": "7",
        }, clear=True):
            cuda, egl = _object_gpu_devices()
            with patch("motion2mixamorig.object_tracking.subprocess.run") as run:
                run.return_value.returncode = 0
                _run_stage(
                    "MegaPose", ["worker"], cuda_visible=cuda,
                    worker_env={"EGL_VISIBLE_DEVICES": egl},
                )
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(run.call_args.kwargs["env"]["EGL_VISIBLE_DEVICES"], "7")

    def test_mock_model_stages_preserve_video_frames_and_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.mp4"
            _video(video)
            skeleton = root / "skeleton.npz"
            times = np.arange(4, dtype=np.float64) / 30
            np.savez_compressed(skeleton, timestamps=times, fps=np.float32(30))
            output = root / "output"
            output.mkdir()
            selection = ObjectSelection(1, (4, 3, 21, 17), 32, 24, 0.25)
            stages = []

            def fake_stage(label, command, **kwargs):
                stages.append(label)

                def argument(name):
                    return Path(command[command.index(name) + 1])

                if label == "SAM 2":
                    boxes = np.tile([5, 4, 20, 16], (4, 1)).astype(np.float32)
                    boxes[2] = np.nan
                    np.savez_compressed(argument("--output"), boxes=boxes, visible=[True, True, False, True])
                    mask = np.zeros((24, 32), dtype=np.uint8)
                    mask[4:16, 5:20] = 255
                    self.assertTrue(cv2.imwrite(str(argument("--seed-mask")), mask))
                elif label == "TripoSR":
                    mesh = trimesh.creation.box(extents=[2, 1, 1])
                    mesh.visual.vertex_colors = [220, 40, 20, 255]
                    (argument("--output-dir") / "0" / "mesh.glb").write_bytes(mesh.export(file_type="glb"))
                else:
                    self.assertEqual(kwargs["worker_env"], {"EGL_VISIBLE_DEVICES": "0"})
                    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
                    poses[:, :3, 3] = [1, 2, 3]
                    poses[2] = np.nan
                    np.savez_compressed(
                        argument("--output"), camera_from_object=poses,
                        valid=[True, True, False, True], intrinsics=np.eye(3),
                    )

            runtimes = ObjectRuntimes(Path("sam"), Path("tripo"), root, Path("mega"))
            with patch("huggingface_hub.hf_hub_download", return_value="checkpoint"), \
                 patch("huggingface_hub.snapshot_download", return_value="weights"), \
                 patch("motion2mixamorig.object_tracking._run_stage", side_effect=fake_stage), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "EGL_VISIBLE_DEVICES": "0"}):
                capture_object(video, skeleton, selection, output, runtimes)

            self.assertEqual(stages, ["SAM 2", "TripoSR", "MegaPose"])
            with np.load(output / "object_motion.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["frame_indices"], np.arange(4))
                np.testing.assert_allclose(data["timestamps"], times)
                np.testing.assert_array_equal(data["valid"], [True, True, False, True])
                self.assertTrue(np.isnan(data["camera_from_object"][2]).all())
            metadata = json.loads((output / "object_metadata.json").read_text())
            self.assertEqual(metadata["missing_frame_count"], 1)
            self.assertEqual(metadata["selection_frame"], 1)
            self.assertIn("Not attached or aligned", metadata["warning"])
            gltf, binary = read_glb(output / "object.glb")
            channel = next(c for c in gltf["animations"][0]["channels"] if c["target"]["path"] == "scale")
            sampler = gltf["animations"][0]["samplers"][channel["sampler"]]
            np.testing.assert_array_equal(_accessor(gltf, binary, sampler["output"])[:, 0], [1, 1, 0, 1])

    def test_space_selection_and_optional_bundle(self):
        import app

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "clip.mp4"
            _video(video)
            _, state, _ = app.preview_object_frame(str(video), 1)
            _, state, _ = app.select_object_point(str(video), 1, state, SimpleNamespace(index=(5, 4)))
            _, state, _ = app.select_object_point(str(video), 1, state, SimpleNamespace(index=(20, 16)))
            selection = app._object_selection(str(video), 1, state, 25)
            self.assertEqual(selection.box, (5, 4, 20, 16))
            self.assertEqual(selection.longest_side_m, 0.25)
            with self.assertRaisesRegex(app.gr.Error, "Load the object frame"):
                app._object_selection(str(video), 2, state, 25)

            progress_values = []

            def pipeline(*args, **kwargs):
                out = kwargs["output_dir"]
                out.mkdir()
                (out / "mixamo_character.glb").write_bytes(b"character")
                kwargs["progress"].emit("extract", 0.15, "Extracting human motion")
                kwargs["progress"].emit("done", 1.0, "Completed")

            def fake_capture(source, skeleton, picked, out, runtimes, **kwargs):
                self.assertEqual(picked, selection)
                kwargs["progress"](0.05, "Preparing frames")
                kwargs["progress"](1.0, "Object capture finished")
                for name in ("object.glb", "object_motion.npz", "object_metadata.json"):
                    (out / name).write_bytes(b"object")

            with patch.object(app, "require_cuda"), \
                 patch.object(app.checks, "check_video", return_value=None), \
                 patch.object(app, "run_pipeline", side_effect=pipeline), \
                 patch.object(app, "_release_cuda_memory"), \
                 patch.object(app, "GRADIO_TEMP_DIR", root / "cache"), \
                 patch("motion2mixamorig.object_tracking.object_runtimes", return_value=object()), \
                 patch("motion2mixamorig.object_tracking.capture_object", side_effect=fake_capture):
                status, archive = app.generate(
                    "Video", str(video), None, None, capture_held_object=True,
                    object_frame=1, object_state=state, object_size_cm=25,
                    progress=lambda value, desc: progress_values.append(value),
                )
            self.assertIn("not attached or aligned", status)
            np.testing.assert_allclose(progress_values, [0.075, 0.5, 0.525, 1.0])
            with ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {
                    "mixamo_character.glb", "object.glb", "object_motion.npz", "object_metadata.json",
                })
            incomplete = root / "incomplete"
            incomplete.mkdir()
            (incomplete / "mixamo_character.glb").write_bytes(b"character")
            with self.assertRaisesRegex(RuntimeError, "object.glb"):
                app._zip_outputs(incomplete, include_object=True)


if __name__ == "__main__":
    unittest.main()
