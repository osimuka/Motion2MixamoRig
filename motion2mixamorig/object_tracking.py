"""Orchestrate the optional, separately installed video-object models."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import paths
from .object_export import prepare_object_mesh, write_object_glb

SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
TRIPOSR_MODEL = "stabilityai/TripoSR"
TRIPOSR_REVISION = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
OBJECT_FILES = ("object.glb", "object_motion.npz", "object_metadata.json")


@dataclass(frozen=True)
class ObjectSelection:
    frame: int
    box: tuple[int, int, int, int]
    width: int
    height: int
    longest_side_m: float


@dataclass(frozen=True)
class ObjectRuntimes:
    sam2_python: Path
    triposr_python: Path
    triposr_dir: Path
    megapose_python: Path


def _object_gpu_devices() -> tuple[str, str]:
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if not cuda.isdecimal():
        raise RuntimeError(
            "MegaPose requires a numeric CUDA_VISIBLE_DEVICES index for one NVIDIA GPU; "
            "UUID-based device selection is not supported by the upstream runtime."
        )
    egl = os.environ.get("EGL_VISIBLE_DEVICES")
    if egl is None:
        if int(cuda) != 0:
            raise RuntimeError(
                "For a nonzero CUDA device, set EGL_VISIBLE_DEVICES to the matching "
                "EGL renderer device index before enabling object capture."
            )
        egl = "0"
    egl = egl.strip()
    if not egl.isdecimal():
        raise RuntimeError("EGL_VISIBLE_DEVICES must be one numeric EGL device index.")
    return cuda, egl


def object_runtimes() -> ObjectRuntimes:
    def configured(name: str, *, executable: bool = False) -> Path:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"Object capture is not configured: set {name}. See docs/huggingface-space.md.")
        path = Path(value).expanduser()
        if not path.exists() or (executable and (not path.is_file() or not os.access(path, os.X_OK))):
            raise RuntimeError(f"Object capture setting {name} is not a usable path: {path}")
        return path.absolute()

    sam2 = configured("M2MR_SAM2_PYTHON", executable=True)
    tripo_python = configured("M2MR_TRIPOSR_PYTHON", executable=True)
    tripo_dir = configured("M2MR_TRIPOSR_DIR")
    megapose = configured("M2MR_MEGAPOSE_PYTHON", executable=True)
    if not (tripo_dir / "run.py").is_file():
        raise RuntimeError("M2MR_TRIPOSR_DIR must point to a TripoSR checkout containing run.py.")
    data = configured("MEGAPOSE_DATA_DIR")
    for model in ("coarse-rgb-906902141", "refiner-rgb-653307694"):
        directory = data / "megapose-models" / model
        if not (directory / "config.yaml").is_file() or not (directory / "checkpoint.pth.tar").is_file():
            raise RuntimeError(
                f"MegaPose model files are missing in {directory}; follow the object setup guide."
            )
    if not shutil.which("ffprobe"):
        raise RuntimeError("Object capture needs ffprobe to verify the source video timing.")
    _object_gpu_devices()
    return ObjectRuntimes(sam2, tripo_python, tripo_dir, megapose)


def validate_video_times(source_times: np.ndarray, character_times: np.ndarray) -> None:
    source = np.asarray(source_times, dtype=np.float64)
    character = np.asarray(character_times, dtype=np.float64)
    if source.ndim != 1 or character.ndim != 1 or not len(source) or len(source) != len(character):
        raise ValueError("Object and character frame counts differ; re-export the source video.")
    if not np.isfinite(source).all() or not np.isfinite(character).all():
        raise ValueError("Source video has invalid frame timestamps.")
    source = source - source[0]
    if len(source) > 1:
        if np.any(np.diff(source) <= 0) or np.any(np.diff(character) <= 0):
            raise ValueError("Source video has non-increasing frame timestamps.")
        tolerance = max(0.0015, float(np.median(np.diff(character))) * 0.05)
        if np.max(np.abs(source - (character - character[0]))) > tolerance:
            raise ValueError(
                "Object capture requires a constant-frame-rate video synchronized with the "
                "character. Re-export the video at a constant frame rate."
            )


def _source_timestamps(video: Path) -> np.ndarray:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        frames = json.loads(result.stdout)["frames"]
        return np.array([float(frame["best_effort_timestamp_time"]) for frame in frames])
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot read source frame timestamps with ffprobe; re-export the video."
        ) from exc


def _decode_frames(video: Path, directory: Path, count: int, selection: ObjectSelection) -> None:
    import cv2

    directory.mkdir()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot decode the source video for object capture.")
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= count:
                raise ValueError("The object video has more frames than the character.")
            if frame.shape[:2] != (selection.height, selection.width):
                raise ValueError("Video dimensions changed after selecting the object.")
            if not cv2.imwrite(
                str(directory / f"{index:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not write object-tracking frame {index}.")
            index += 1
        if index != count:
            raise ValueError("The object video has fewer decoded frames than the character.")
    finally:
        capture.release()


def _make_triposr_input(frame_path: Path, mask_path: Path, output: Path) -> None:
    import cv2
    from .object_sam2 import mask_bounding_box

    frame = cv2.imread(str(frame_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask is None or frame.shape[:2] != mask.shape:
        raise ValueError("The selected object frame and mask do not match.")
    box = mask_bounding_box(mask > 0)
    if box is None:
        raise ValueError("The selected object mask is empty.")
    x0, y0, x1, y1 = box
    crop, alpha = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    scale = 0.85 * 512 / max(crop.shape[:2])
    width = max(1, round(crop.shape[1] * scale))
    height = max(1, round(crop.shape[0] * scale))
    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    image = np.full((512, 512, 3), 127, dtype=np.uint8)
    x, y = (512 - width) // 2, (512 - height) // 2
    image[y:y + height, x:x + width] = np.where(alpha[..., None], crop, 127)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError("Could not save the selected object image for reconstruction.")


def _run_stage(
    label: str, command: list[str], *, cuda_visible: str,
    worker_env: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if worker_env:
        env.update(worker_env)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    try:
        result = subprocess.run(command, env=env, check=False, timeout=7200)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{label} could not complete; check the Space logs and runtime setup.") from exc
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); see the Space logs.")


def capture_object(
    video: Path,
    skeleton: Path,
    selection: ObjectSelection,
    output_dir: Path,
    runtimes: ObjectRuntimes,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    with np.load(skeleton, allow_pickle=False) as data:
        times = np.asarray(data["timestamps"], dtype=np.float64)
        fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("The character skeleton has an invalid frame rate.")
    validate_video_times(_source_timestamps(video), times)
    x0, y0, x1, y1 = selection.box
    if (
        selection.frame < 0 or selection.frame >= len(times)
        or selection.width <= 0 or selection.height <= 0
        or not (0 <= x0 < x1 <= selection.width and 0 <= y0 < y1 <= selection.height)
        or x1 - x0 < 2 or y1 - y0 < 2
        or not np.isfinite(selection.longest_side_m) or selection.longest_side_m <= 0
    ):
        raise ValueError("Invalid object frame, box, or size; select the object again.")

    def update(value: float, message: str) -> None:
        if progress is not None:
            progress(value, message)

    cuda_visible, egl_visible = _object_gpu_devices()
    workers = Path(__file__).parent
    with tempfile.TemporaryDirectory(prefix="m2mr-object-", dir=output_dir.parent) as temp:
        root = Path(temp)
        frames = root / "frames"
        update(0.05, "Preparing frames for object capture")
        _decode_frames(video, frames, len(times), selection)
        update(0.10, "Segmenting the selected object with SAM 2")
        checkpoint = hf_hub_download(
            SAM2_MODEL, "sam2.1_hiera_tiny.pt", revision=SAM2_REVISION,
            cache_dir=paths.WEIGHTS / "objects" / "sam2",
        )
        masks, seed = root / "masks.npz", root / "seed_mask.png"
        _run_stage("SAM 2", [
            str(runtimes.sam2_python), "-u", str(workers / "object_sam2.py"),
            "--frames", str(frames), "--frame-count", str(len(times)),
            "--selected-frame", str(selection.frame), "--box", *(str(v) for v in selection.box),
            "--checkpoint", str(checkpoint), "--output", str(masks), "--seed-mask", str(seed),
        ], cuda_visible=cuda_visible)

        update(0.35, "Reconstructing the object with TripoSR")
        tripo_input = root / "triposr_input.png"
        _make_triposr_input(frames / f"{selection.frame:05d}.jpg", seed, tripo_input)
        weights = snapshot_download(
            TRIPOSR_MODEL, revision=TRIPOSR_REVISION,
            allow_patterns=["config.yaml", "model.ckpt"],
            cache_dir=paths.WEIGHTS / "objects" / "triposr",
        )
        tripo_out = root / "triposr"
        (tripo_out / "0").mkdir(parents=True)
        _run_stage("TripoSR", [
            str(runtimes.triposr_python), "-u", str(runtimes.triposr_dir / "run.py"),
            str(tripo_input), "--device", "cuda:0", "--no-remove-bg",
            "--pretrained-model-name-or-path", str(weights),
            "--model-save-format", "glb", "--output-dir", str(tripo_out),
        ], cuda_visible=cuda_visible)
        obj, static_glb = prepare_object_mesh(
            tripo_out / "0" / "mesh.glb", root / "mesh", selection.longest_side_m,
        )

        update(0.55, "Estimating object poses with MegaPose")
        predictions = root / "poses.npz"
        _run_stage("MegaPose", [
            str(runtimes.megapose_python), "-u", str(workers / "object_megapose.py"),
            "--frames", str(frames), "--masks", str(masks),
            "--mesh", str(obj), "--output", str(predictions),
        ], cuda_visible=cuda_visible, worker_env={"EGL_VISIBLE_DEVICES": egl_visible})
        with np.load(masks, allow_pickle=False) as data:
            boxes = np.asarray(data["boxes"], dtype=np.float32)
            visible = np.asarray(data["visible"], dtype=bool)
        with np.load(predictions, allow_pickle=False) as data:
            poses = np.asarray(data["camera_from_object"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=bool)
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        if (
            boxes.shape != (len(times), 4) or visible.shape != (len(times),)
            or poses.shape != (len(times), 4, 4) or valid.shape != (len(times),)
            or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or np.any(valid & ~visible)
        ):
            raise ValueError("Object model output has inconsistent frame or camera data.")

        update(0.95, "Writing the separate animated object")
        write_object_glb(static_glb, output_dir / "object.glb", times, poses, valid)
        convention = (
            "camera_from_object: 4x4 object-to-OpenCV-camera transform, metres; "
            "camera x right, y down, z forward. object.glb converts camera axes "
            "with diag(1,-1,-1) to Y-up. Not aligned or attached to the character."
        )
        np.savez_compressed(
            output_dir / "object_motion.npz",
            timestamps=times, frame_indices=np.arange(len(times)), fps=np.float32(fps),
            camera_from_object=poses, valid=valid, boxes_xyxy=boxes,
            camera_intrinsics=intrinsics, convention=np.array(convention),
        )
        (output_dir / "object_metadata.json").write_text(json.dumps({
            "schema": "m2mr-object-v1",
            "selection_frame": selection.frame,
            "selection_box_xyxy": list(selection.box),
            "object_longest_side_m": selection.longest_side_m,
            "fps": fps,
            "frame_count": len(times),
            "estimated_frame_count": int(valid.sum()),
            "missing_frame_count": int((~valid).sum()),
            "camera_intrinsics": intrinsics.tolist(),
            "convention": convention,
            "warning": (
                "Approximate shape, size, camera intrinsics and per-frame poses. "
                "Missing masks hide the object in the GLB; a nonempty mask does not "
                "guarantee correct tracking. Not attached or aligned to the character."
            ),
        }, indent=2) + "\n", encoding="utf-8")
        update(1.0, "Object capture finished")
