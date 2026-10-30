"""Native SAM MHR70 hand landmarks -> rig-relative Mixamo finger swings.

Palm-relative directions remove camera/wrist orientation. Landmark positions
constrain curl and splay, not axial finger twist or occlusion confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .hand_retarget import (
    HAND_JOINT_NAMES, SIDES, HandMapping, HandMotion, _hand_frame,
    finger_bone, smooth_hand_pose,
)
from .kinematics import MixamoRig, forward_kinematics, rotation_aligning

LOGGER = logging.getLogger(__name__)
FINGER_BASES = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}


def load_sam_hands(
    path: Path, rig: MixamoRig, *, smooth_s: float = 0.08,
) -> tuple[HandMotion, HandMapping] | None:
    """Build rig-local rotations; never interpret SAM parameters as MANO pose."""
    with np.load(path, allow_pickle=False) as data:
        if "sam_hand_keypoints" not in data.files:
            return None
        points = np.asarray(data["sam_hand_keypoints"], dtype=np.float64)
        fps = float(data["fps"])
        frame_count = len(data["joints_3d"])
    if points.shape != (frame_count, 2, 21, 3) or frame_count == 0:
        raise ValueError("SAM hands must be (body frames, 2, 21, 3)")
    if not np.isfinite(points).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("SAM hand landmarks and positive FPS must be finite")
    if not np.isfinite(smooth_s) or smooth_s < 0:
        raise ValueError("Finger smoothing must be finite and nonnegative")

    world = forward_kinematics(rig, {})
    pose = np.tile(np.eye(3), (frame_count, 2, 15, 1, 1))
    valid = np.zeros((frame_count, 2), dtype=bool)
    bones: list[str] = []
    basis = {}
    for side_index, side in enumerate(SIDES):
        names = [finger_bone(side, joint) for joint in HAND_JOINT_NAMES]
        if any(name not in world for name in names):
            raise ValueError(
                f"Native SAM hands require all 15 {side.lower()} Mixamo finger bones; "
                "upload a finger-rigged character or disable finger animation."
            )
        hand = world[f"mixamorig:{side}Hand"][:3, 3]
        rest_palm = _hand_frame(
            hand, world[finger_bone(side, "middle1")][:3, 3],
            world[finger_bone(side, "index1")][:3, 3],
            world[finger_bone(side, "pinky1")][:3, 3],
        )
        bones.extend(names)
        basis[side] = np.eye(3)
        for frame in range(frame_count):
            landmarks = points[frame, side_index]
            along = landmarks[9] - landmarks[0]
            lateral = landmarks[5] - landmarks[17]
            if np.linalg.norm(along) < 1e-6 or np.linalg.norm(np.cross(along, lateral)) < 1e-8:
                continue
            palm = _hand_frame(landmarks[0], landmarks[9], landmarks[5], landmarks[17])
            alignment = rest_palm @ palm.T
            frame_valid = True
            for finger, base in FINGER_BASES.items():
                cumulative = np.eye(3)
                for joint in range(1, 4):
                    name = finger_bone(side, f"{finger}{joint}")
                    if joint < 3:
                        rest_direction = (
                            world[finger_bone(side, f"{finger}{joint + 1}")][:3, 3]
                            - world[name][:3, 3]
                        )
                    else:
                        rest_direction = (
                            world[name][:3, 3]
                            - world[finger_bone(side, f"{finger}2")][:3, 3]
                        )
                    observed = landmarks[base + joint] - landmarks[base + joint - 1]
                    if min(np.linalg.norm(rest_direction), np.linalg.norm(observed)) < 1e-6:
                        frame_valid = False
                        break
                    updated = rotation_aligning(
                        cumulative @ rest_direction, alignment @ observed,
                    ) @ cumulative
                    rest_rotation = world[name][:3, :3]
                    delta = rest_rotation.T @ cumulative.T @ updated @ rest_rotation
                    pose[frame, side_index, HAND_JOINT_NAMES.index(f"{finger}{joint}")] = delta
                    cumulative = updated
                if not frame_valid:
                    break
            valid[frame, side_index] = frame_valid

    if not valid.any():
        raise ValueError("SAM returned no geometrically usable hands; disable fingers or use a clearer clip.")
    if not valid.all():
        LOGGER.warning(
            "SAM hands: %d of %d hand-frames have degenerate geometry; "
            "short gaps may be smoothed and unfilled gaps remain at rest. "
            "Validity is geometric, not visibility confidence.",
            int((~valid).sum()), valid.size,
        )
    pose, valid = smooth_hand_pose(pose, valid, fps, smooth_s)
    # Rotations already use rig-local axes, so the shared mapping is identity.
    mapping = HandMapping(rig, basis, {name: np.eye(3) for name in bones}, tuple(bones))
    motion = HandMotion(pose, valid, None, fps, float(smooth_s))
    return motion, mapping
