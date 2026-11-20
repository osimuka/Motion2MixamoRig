"""Isolated MegaPose worker; transforms are object-to-OpenCV-camera metres."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def estimate_video(frames: Path, masks: Path, mesh: Path, output: Path) -> None:
    os.environ["CONDA_PREFIX"] = sys.prefix
    os.environ["LOCAL_RANK"] = "0"
    requested_egl = os.environ.get("EGL_VISIBLE_DEVICES")
    import cv2
    import torch
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.datasets.scene_dataset import ObjectData
    from megapose.inference.types import ObservationTensor
    from megapose.inference.utils import make_detections_from_object_data
    from megapose.utils.load_model import NAMED_MODELS, load_named_model

    # MegaPose's Slurm initialization can overwrite an explicitly selected EGL device.
    if requested_egl is not None:
        os.environ["EGL_VISIBLE_DEVICES"] = requested_egl
    if not torch.cuda.is_available():
        raise RuntimeError("Object pose estimation requires an NVIDIA CUDA GPU.")
    with np.load(masks, allow_pickle=False) as data:
        boxes = np.asarray(data["boxes"], dtype=np.float32)
        visible = np.asarray(data["visible"], dtype=bool)
    if boxes.shape != (len(visible), 4) or not visible.any():
        raise ValueError("Object segmentation produced no usable boxes.")
    first = cv2.imread(str(frames / "00000.jpg"))
    if first is None:
        raise ValueError("Cannot read the first object-tracking frame.")
    height, width = first.shape[:2]
    focal = float(np.hypot(width, height))
    intrinsics = np.array([
        [focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1],
    ], dtype=np.float32)
    dataset = RigidObjectDataset([
        RigidObject(label="selected_object", mesh_path=mesh, mesh_units="m"),
    ])
    model_name = "megapose-1.0-RGB"
    estimator = load_named_model(model_name, dataset, n_workers=1, bsz_images=16).cuda()
    poses = np.full((len(visible), 4, 4), np.nan, dtype=np.float32)
    valid = np.zeros(len(visible), dtype=bool)
    try:
        with torch.no_grad():
            for index in np.flatnonzero(visible):
                frame = cv2.imread(str(frames / f"{index:05d}.jpg"))
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError(f"Cannot read object-tracking frame {index}.")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                observation = ObservationTensor.from_numpy(rgb, None, intrinsics).cuda()
                detections = make_detections_from_object_data([
                    ObjectData(label="selected_object", bbox_modal=boxes[index]),
                ]).cuda()
                result, _ = estimator.run_inference_pipeline(
                    observation, detections=detections,
                    **NAMED_MODELS[model_name]["inference_parameters"],
                )
                predicted = result.poses.detach().cpu().numpy()
                if predicted.shape != (1, 4, 4):
                    raise ValueError(
                        f"MegaPose returned {len(predicted)} poses for frame {index}, expected one."
                    )
                poses[index] = predicted[0]
                valid[index] = True
                if (index + 1) % 30 == 0:
                    print(f"MegaPose: processed frame {index + 1}/{len(visible)}", flush=True)
    finally:
        estimator.refiner_model.renderer.stop()
    np.savez_compressed(output, camera_from_object=poses, valid=valid, intrinsics=intrinsics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    estimate_video(args.frames, args.masks, args.mesh, args.output)


if __name__ == "__main__":
    main()
