"""SAM 3D Body frames -> approximate 24-joint camera-aligned motion.

SAM is an image model, not a world-motion tracker. Use a fixed, level camera:
camera motion is not removed and spine/collar positions are interpolated.
The output uses the same joint-name contract as GVHMR, not the SMPL-X model.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import paths
from ..extract import playback_fps
from .base import CheckResult, SMPL_JOINT_NAMES, write_skeleton_npz

MODEL_ID = "facebook/sam-3d-body-dinov3"
MODEL_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
MODEL_HINT = (
    f"request access at https://huggingface.co/{MODEL_ID}, accept Meta's terms, "
    "then set HF_TOKEN as a Space secret (never paste it into source code)"
)
CAMERA_CONVENTION = (
    "SAM camera metres transformed by diag(1,-1,-1) to Y-up; initial pelvis XZ "
    "at origin, initial lowest ankle/toe at Y=0. Fixed level camera assumed. "
    "No camera-motion compensation or gravity recovery. Spine/collars approximated "
    "from landmarks; fingers not animated."
)


class BodyEstimator(Protocol):
    def process_one_image(
        self, img: np.ndarray, *, bboxes: np.ndarray, inference_type: str
    ) -> list[dict[str, np.ndarray]]: ...


def map_keypoints(keypoints: np.ndarray, camera_translation: np.ndarray) -> np.ndarray:
    """Map the upstream MHR70 landmark order (metres) into the shared contract."""
    points = np.asarray(keypoints, dtype=np.float64)
    translation = np.asarray(camera_translation, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 70 or points.shape[1] != 3:
        raise ValueError(f"SAM keypoints must be (70 or more, 3), got {points.shape}")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("SAM camera translation must contain three finite values")
    if not np.isfinite(points[:70]).all():
        raise ValueError("SAM returned non-finite keypoints")
    # Upstream keypoints exclude pred_cam_t; both are already in metres.
    points = (points[:70] + translation) * np.array([1.0, -1.0, -1.0])
    pelvis = (points[9] + points[10]) / 2
    chest = (points[5] + points[6]) / 2
    mapped = {
        "pelvis": pelvis,
        "spine1": pelvis + (chest - pelvis) / 3,
        "spine2": pelvis + (chest - pelvis) * 2 / 3,
        "spine3": chest,
        "neck": points[69],
        "head": (points[3] + points[4]) / 2,
        "left_collar": chest + (points[5] - chest) / 3,
        "right_collar": chest + (points[6] - chest) / 3,
        "left_foot": (points[15] + points[16]) / 2,
        "right_foot": (points[18] + points[19]) / 2,
        "left_hand": (points[49] + points[61]) / 2,
        "right_hand": (points[28] + points[40]) / 2,
    }
    for name, index in (
        ("left_hip", 9), ("right_hip", 10),
        ("left_knee", 11), ("right_knee", 12),
        ("left_ankle", 13), ("right_ankle", 14),
        ("left_shoulder", 5), ("right_shoulder", 6),
        ("left_elbow", 7), ("right_elbow", 8),
        ("left_wrist", 62), ("right_wrist", 41),
    ):
        mapped[name] = points[index]
    return np.stack([mapped[name] for name in SMPL_JOINT_NAMES])


def smooth_and_align(joints: np.ndarray, fps: float) -> np.ndarray:
    """Offline 40 ms Gaussian smoothing; preserve translation in one camera frame."""
    from scipy.ndimage import gaussian_filter1d

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3) or len(joints) == 0:
        raise ValueError("SAM motion must contain one or more frames of 24 joints")
    if not np.isfinite(joints).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("SAM motion and FPS must be finite, with FPS positive")
    smoothed = gaussian_filter1d(joints, sigma=0.04 * fps, axis=0, mode="nearest")
    offset = smoothed[0, 0].copy()
    offset[1] = smoothed[0, [7, 8, 10, 11], 1].min()
    return (smoothed - offset).astype(np.float32)


def require_cuda(device: str) -> None:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "SAM 3D Body requires CUDA. Select an NVIDIA GPU in Space Settings "
            "or run the CLI with --device cuda on a CUDA-enabled machine."
        )


def load_estimator() -> BodyEstimator:
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    cache = paths.WEIGHTS / "sam3d"
    try:
        files = {
            name: hf_hub_download(
                MODEL_ID, name, revision=MODEL_REVISION, cache_dir=cache
            )
            for name in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt")
        }
    except GatedRepoError as exc:
        raise RuntimeError(f"SAM 3D Body model access is required: {MODEL_HINT}") from exc
    torch.hub.set_dir(str(paths.WEIGHTS / "torch"))
    model, config = load_sam_3d_body(
        checkpoint_path=files["model.ckpt"],
        mhr_path=files["assets/mhr_model.pt"],
        device="cuda",
    )
    return SAM3DBodyEstimator(model, config)


def load_person_detector(device: str):
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    torch.hub.set_dir(str(paths.WEIGHTS / "torch"))
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    return fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)


def person_boxes(frame: np.ndarray, detector, device: str) -> np.ndarray:
    import torch

    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(device).float() / 255
    with torch.inference_mode():
        prediction = detector([tensor])[0]
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = (labels == 1) & (scores >= 0.7) & (area >= 0.015 * frame.shape[0] * frame.shape[1])
    return boxes[keep].astype(np.float32)


def infer_frame(frame: np.ndarray, estimator: BodyEstimator, detector, index: int) -> np.ndarray:
    import cv2

    boxes = person_boxes(frame, detector, "cuda")
    if len(boxes) != 1:
        raise ValueError(
            f"SAM frame {index}: detected {len(boxes)} people; exactly one clearly "
            "visible person is required in every frame. Trim or replace the clip."
        )
    predictions = estimator.process_one_image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=boxes, inference_type="body"
    )
    if len(predictions) != 1:
        raise ValueError(f"SAM frame {index}: expected one pose, got {len(predictions)}")
    prediction = predictions[0]
    return map_keypoints(prediction["pred_keypoints_3d"], prediction["pred_cam_t"])


class Sam3dExtractor:
    name = "sam3d"
    weights_note = (
        f"SAM 3D Body downloads from {MODEL_ID} into weights/sam3d/ on first use; "
        "Meta approval, HF_TOKEN and an NVIDIA GPU are required. No SMPL-X needed."
    )

    def check(self, detect_hands: bool = False) -> list[CheckResult]:
        results = [
            CheckResult(
                f"python module      {module}",
                importlib.util.find_spec(module) is not None,
                "install the SAM runtime: see docs/huggingface-space.md",
                section="module",
            )
            for module in ("sam_3d_body", "torch", "torchvision", "huggingface_hub")
        ]
        if importlib.util.find_spec("torch") is not None:
            import torch

            results.append(CheckResult(
                "SAM CUDA runtime",
                torch.cuda.is_available(),
                "select an NVIDIA GPU in Space Settings; CPU and MPS are unsupported",
            ))
        return results

    def extract(
        self,
        source: Path,
        output_npz: Path,
        *,
        work_dir: Path,
        device: str = "cpu",
        verbose: bool = True,
        detect_hands: bool = False,
    ) -> Path:
        import cv2

        if detect_hands:
            raise ValueError("sam3d does not yet support Mixamo finger animation; omit --detect-hands")
        require_cuda(device)
        estimator = load_estimator()
        detector = load_person_detector(device)
        source = Path(source)
        is_still = paths.is_image(source)
        frames: list[np.ndarray] = []
        if is_still:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Cannot decode still image: {source}")
            pose = infer_frame(image, estimator, detector, 0)
            frames = [pose] * 32
            fps = 30.0
        else:
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f"Cannot decode video: {source}")
                expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                native_fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(native_fps) or native_fps <= 0 or expected <= 0:
                    raise ValueError("Video has invalid timing; re-export as a constant-frame-rate MP4")
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    frames.append(infer_frame(image, estimator, detector, len(frames)))
                    if verbose and len(frames) % 30 == 0:
                        print(f"SAM 3D Body: {len(frames)}/{expected} frames", flush=True)
                if not frames or abs(len(frames) - expected) > 1:
                    raise ValueError(
                        f"Video decoded {len(frames)} of {expected} frames; re-export the clip"
                    )
                fps = playback_fps(source, len(frames), fallback=native_fps)
            finally:
                capture.release()
        joints = smooth_and_align(np.stack(frames), fps)
        return write_skeleton_npz(
            output_npz,
            extractor=self.name,
            joints_3d=joints,
            fps=fps,
            root_translation=joints[:, 0],
            source_is_image=np.bool_(is_still),
            video=np.array(str(source)) if not is_still else None,
            image=np.array(str(source)) if is_still else None,
            coordinate_convention=np.array(CAMERA_CONVENTION),
            model_id=np.array(MODEL_ID),
            model_revision=np.array(MODEL_REVISION),
        )
